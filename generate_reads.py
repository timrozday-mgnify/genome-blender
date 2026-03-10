#!/usr/bin/env python3
"""Generate simulated WGS reads from reference genomes.

Takes reference genomes via a CSV input table and produces FASTQ reads
with configurable fragment size distributions, GC bias, and read modes
(single-end or paired-end).
"""

from __future__ import annotations

import csv
import logging
import math
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

import click.core
import pysam
import torch
import typer
import yaml
from Bio import SeqIO
from Bio.Seq import Seq
from pyro.distributions import Categorical as PyroCategorical
from pyro.distributions import LogNormal, NegativeBinomial, Poisson
from pyro.distributions.hmm import DiscreteHMM

logger = logging.getLogger(__name__)

_inner_progress: Progress | None = None
_BASES = "ACGT"

app = typer.Typer()


@contextmanager
def progress_task(total: int, description: str):
    """Add a sub-task to the inner progress display.

    Yields a callable that advances the task by one step.
    """
    if _inner_progress is None:
        yield lambda: None
        return
    task_id = _inner_progress.add_task(description, total=total)
    try:
        yield lambda: _inner_progress.advance(task_id)
    finally:
        _inner_progress.remove_task(task_id)


class ErrorModel(str, Enum):
    none = "none"
    illumina = "illumina"
    pacbio = "pacbio"
    nanopore = "nanopore"


class QualityCalibrationModel(str, Enum):
    phred = "phred"
    log_linear = "log-linear"
    sigmoid = "sigmoid"


# Mapping from YAML key names to the enum classes that need conversion
_ENUM_PARAMS: dict[str, type[Enum]] = {
    "error_model": ErrorModel,
    "quality_calibration_model": QualityCalibrationModel,
}


def _load_yaml_config(config_path: Path) -> dict[str, object]:
    """Load a YAML config file and normalise keys to snake_case.

    Parameters
    ----------
    config_path : Path
        Path to the YAML configuration file.

    Returns
    -------
    dict mapping parameter names (snake_case) to values.

    Raises
    ------
    typer.BadParameter
        If the file doesn't contain a YAML mapping.

    """
    with open(config_path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise typer.BadParameter(
            f"Config file must contain a YAML mapping, got {type(data).__name__}"
        )
    # Normalise kebab-case keys to snake_case
    return {k.replace("-", "_"): v for k, v in data.items()}


def _apply_yaml_config(
    ctx: typer.Context,
    config: dict[str, object],
) -> None:
    """Override default-valued CLI parameters with YAML config values.

    Only parameters whose source is DEFAULT (i.e. not explicitly provided
    on the command line) are overridden. Enum-typed parameters are
    converted from their string values.

    Parameters
    ----------
    ctx : typer.Context
        Typer/Click invocation context.
    config : dict
        Parsed YAML config (snake_case keys).

    """
    for key, value in config.items():
        source = ctx.get_parameter_source(key)
        if source is not click.core.ParameterSource.COMMANDLINE:
            if key in _ENUM_PARAMS and isinstance(value, str):
                value = _ENUM_PARAMS[key](value)
            elif key == "input_csv" and isinstance(value, str):
                value = Path(value)
            ctx.params[key] = value


@dataclass
class Fragment:
    """A DNA fragment excised from a genome."""

    genome_id: str
    contig_id: str
    start: int  # 0-based
    end: int  # half-open
    strand: str  # '+' or '-'
    sequence: str


@dataclass
class Read:
    """A sequencing read with name, sequence, and quality string."""

    name: str
    sequence: str
    quality: str
    cigar: list[tuple[int, int]] | None = None  # pysam-style CIGAR tuples


@dataclass
class ReadBatch:
    """Container for generated reads, either single-end or paired-end.

    Exactly one of ``single`` or ``paired`` is set; the other is ``None``.
    """

    single: list[Read] | None = None
    paired: list[tuple[Read, Read]] | None = None

    @property
    def is_paired(self) -> bool:
        """Return True if this batch contains paired-end reads."""
        return self.paired is not None


@dataclass
class QualityCalibration:
    """Base for quality-score-to-error-rate calibration models."""

    name: str

    def __call__(self, q_scores: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class PhredCalibration(QualityCalibration):
    """Theoretical Phred: P = 10^(-Q/10)."""

    name: str = "phred"

    def __call__(self, q_scores: torch.Tensor) -> torch.Tensor:
        return 10.0 ** (-q_scores.float() / 10.0)


@dataclass
class LogLinearCalibration(QualityCalibration):
    """DADA2-style: P = clamp(10^(intercept + slope*Q), floor, ceiling)."""

    name: str = "log-linear"
    intercept: float = -0.3
    slope: float = -0.08
    floor: float = 1e-7
    ceiling: float = 0.5

    def __call__(self, q_scores: torch.Tensor) -> torch.Tensor:
        log10_p = self.intercept + self.slope * q_scores.float()
        return (10.0 ** log10_p).clamp(min=self.floor, max=self.ceiling)


@dataclass
class SigmoidCalibration(QualityCalibration):
    """Logistic sigmoid: P = floor + (ceiling-floor) * sigma(-k*(Q-mid))."""

    name: str = "sigmoid"
    steepness: float = 0.25
    midpoint: float = 15.0
    floor: float = 1e-6
    ceiling: float = 0.5

    def __call__(self, q_scores: torch.Tensor) -> torch.Tensor:
        x = -self.steepness * (q_scores.float() - self.midpoint)
        return self.floor + (self.ceiling - self.floor) * torch.sigmoid(x)


# Noise scales for per-run variability perturbation of calibration parameters
_QCAL_NOISE_SCALES: dict[str, float] = {
    "intercept": 0.15,
    "slope": 0.01,
    "steepness": 0.05,
    "midpoint": 2.0,
}


def build_quality_calibration(
    model_name: str,
    variability: float,
    rng: torch.Generator,
    *,
    intercept: float = -0.3,
    slope: float = -0.08,
    floor: float = 1e-7,
    ceiling: float = 0.5,
    steepness: float = 0.25,
    midpoint: float = 15.0,
) -> QualityCalibration:
    """Build a quality-calibration model, optionally with per-run noise.

    Parameters
    ----------
    model_name : str
        One of ``"phred"``, ``"log-linear"``, ``"sigmoid"``.
    variability : float
        Multiplier for per-run parameter noise. 0 means no perturbation.
    rng : torch.Generator
        Random number generator for reproducible noise.
    intercept, slope : float
        Parameters for the log-linear model.
    floor, ceiling : float
        Hard bounds on predicted error probability (log-linear and sigmoid).
    steepness, midpoint : float
        Parameters for the sigmoid model.

    Returns
    -------
    QualityCalibration

    """
    if variability > 0:
        noise = torch.randn(4, generator=rng)
        intercept += variability * _QCAL_NOISE_SCALES["intercept"] * noise[0].item()
        slope += variability * _QCAL_NOISE_SCALES["slope"] * noise[1].item()
        steepness += variability * _QCAL_NOISE_SCALES["steepness"] * noise[2].item()
        midpoint += variability * _QCAL_NOISE_SCALES["midpoint"] * noise[3].item()

    if model_name == "phred":
        cal = PhredCalibration()
        logger.info("Quality calibration: phred (theoretical)")
        return cal

    if model_name == "log-linear":
        cal = LogLinearCalibration(
            intercept=intercept, slope=slope, floor=floor, ceiling=ceiling,
        )
        logger.info(
            "Quality calibration: log-linear (intercept=%.4f, slope=%.4f, "
            "floor=%.1e, ceiling=%.2f)",
            intercept, slope, floor, ceiling,
        )
        return cal

    if model_name == "sigmoid":
        cal = SigmoidCalibration(
            steepness=steepness, midpoint=midpoint, floor=floor, ceiling=ceiling,
        )
        logger.info(
            "Quality calibration: sigmoid (steepness=%.4f, midpoint=%.2f, "
            "floor=%.1e, ceiling=%.2f)",
            steepness, midpoint, floor, ceiling,
        )
        return cal

    raise ValueError(f"Unknown quality calibration model: {model_name!r}")


@dataclass
class ErrorModelProfile:
    """HMM-based sequencing error model parameters."""

    name: str
    num_states: int
    initial_logits: torch.Tensor  # (num_states,)
    transition_logits: torch.Tensor  # (num_states, num_states)
    emission_logits: torch.Tensor  # (num_states, num_quality_values)
    substitution_ratio: float  # fraction of errors that are substitutions
    insertion_ratio: float  # fraction of errors that are insertions
    deletion_ratio: float  # fraction of errors that are deletions


def _build_emission_logits(
    num_states: int,
    quality_peaks: list[float],
    quality_spreads: list[float],
    num_quality_values: int = 94,
) -> torch.Tensor:
    """Build emission logits for HMM states.

    Each state emits quality scores centred around a peak value with
    a Gaussian-shaped distribution over the quality range.

    Parameters
    ----------
    num_states : int
    quality_peaks : list of float
        Centre quality value for each state.
    quality_spreads : list of float
        Standard deviation of emission distribution for each state.
    num_quality_values : int
        Number of possible quality values (0 to num_quality_values-1).

    Returns
    -------
    torch.Tensor of shape (num_states, num_quality_values)

    """
    q_values = torch.arange(num_quality_values, dtype=torch.float64)
    logits = torch.zeros(num_states, num_quality_values, dtype=torch.float64)
    for s in range(num_states):
        logits[s] = -0.5 * ((q_values - quality_peaks[s]) / quality_spreads[s]) ** 2
    return logits


def _build_sticky_transitions(
    num_states: int,
    self_logit: float = 3.0,
    neighbour_logit: float = 0.5,
) -> torch.Tensor:
    """Build transition logits favouring self-transitions and nearby states.

    Parameters
    ----------
    num_states : int
    self_logit : float
        Logit for staying in the same state.
    neighbour_logit : float
        Logit for transitioning to adjacent states.

    Returns
    -------
    torch.Tensor of shape (num_states, num_states)

    """
    # Start with a low baseline
    logits = torch.full(
        (num_states, num_states), -2.0, dtype=torch.float64
    )
    for s in range(num_states):
        logits[s, s] = self_logit
        if s > 0:
            logits[s, s - 1] = neighbour_logit
        if s < num_states - 1:
            logits[s, s + 1] = neighbour_logit
    return logits


def default_illumina_profile() -> ErrorModelProfile:
    """Return a default Illumina-like error model profile.

    States represent quality zones: very-high (Q37), high (Q33),
    medium (Q25), low (Q15), very-low (Q5). Errors are predominantly
    substitutions (~80%).
    """
    num_states = 5
    quality_peaks = [37.0, 33.0, 25.0, 15.0, 5.0]
    quality_spreads = [3.0, 3.0, 4.0, 4.0, 3.0]

    # Initial state: start in high-quality zone
    initial_logits = torch.tensor(
        [2.0, 1.0, -1.0, -3.0, -5.0], dtype=torch.float64
    )

    transition_logits = _build_sticky_transitions(
        num_states, self_logit=3.5, neighbour_logit=0.5
    )
    emission_logits = _build_emission_logits(
        num_states, quality_peaks, quality_spreads
    )

    return ErrorModelProfile(
        name="illumina",
        num_states=num_states,
        initial_logits=initial_logits,
        transition_logits=transition_logits,
        emission_logits=emission_logits,
        substitution_ratio=0.80,
        insertion_ratio=0.10,
        deletion_ratio=0.10,
    )


def default_pacbio_profile() -> ErrorModelProfile:
    """Return a default PacBio-like error model profile.

    States represent quality zones centred around Q10-Q20. Errors
    are predominantly insertions and deletions (~85%).
    """
    num_states = 5
    quality_peaks = [20.0, 15.0, 10.0, 7.0, 3.0]
    quality_spreads = [3.0, 3.0, 3.0, 3.0, 2.0]

    initial_logits = torch.tensor(
        [1.0, 2.0, 1.0, -1.0, -3.0], dtype=torch.float64
    )

    transition_logits = _build_sticky_transitions(
        num_states, self_logit=3.0, neighbour_logit=0.8
    )
    emission_logits = _build_emission_logits(
        num_states, quality_peaks, quality_spreads
    )

    return ErrorModelProfile(
        name="pacbio",
        num_states=num_states,
        initial_logits=initial_logits,
        transition_logits=transition_logits,
        emission_logits=emission_logits,
        substitution_ratio=0.15,
        insertion_ratio=0.40,
        deletion_ratio=0.45,
    )


def default_nanopore_profile() -> ErrorModelProfile:
    """Return a default Oxford Nanopore-like error model profile.

    States represent quality zones centred around Q7-Q15. Errors
    are predominantly insertions and deletions (~80%).
    """
    num_states = 5
    quality_peaks = [15.0, 12.0, 9.0, 6.0, 3.0]
    quality_spreads = [3.0, 3.0, 3.0, 3.0, 2.0]

    initial_logits = torch.tensor(
        [0.5, 1.5, 1.0, -0.5, -2.0], dtype=torch.float64
    )

    transition_logits = _build_sticky_transitions(
        num_states, self_logit=2.5, neighbour_logit=1.0
    )
    emission_logits = _build_emission_logits(
        num_states, quality_peaks, quality_spreads
    )

    return ErrorModelProfile(
        name="nanopore",
        num_states=num_states,
        initial_logits=initial_logits,
        transition_logits=transition_logits,
        emission_logits=emission_logits,
        substitution_ratio=0.20,
        insertion_ratio=0.35,
        deletion_ratio=0.45,
    )


def batch_sample_quality_scores(
    profile: ErrorModelProfile,
    read_lengths: list[int],
) -> list[torch.Tensor]:
    """Sample quality scores for a batch of reads from the HMM.

    Constructs a single DiscreteHMM at the maximum read length and
    samples all sequences in one call, then truncates each to its
    actual length.

    Parameters
    ----------
    profile : ErrorModelProfile
        HMM parameters and error ratios.
    read_lengths : list of int
        Length of each read to generate quality scores for.

    Returns
    -------
    list of torch.Tensor, each of shape (read_length,) with integer
    quality values.

    """
    if not read_lengths:
        return []

    max_len = max(read_lengths)
    batch_size = len(read_lengths)

    hmm = DiscreteHMM(
        initial_logits=profile.initial_logits,
        transition_logits=profile.transition_logits,
        observation_dist=PyroCategorical(
            logits=profile.emission_logits
        ),
        duration=max_len,
    )

    # Shape: (batch_size, max_len)
    all_scores = hmm.sample(sample_shape=(batch_size,))

    return [
        all_scores[i, :length]
        for i, length in enumerate(read_lengths)
    ]

def apply_errors_to_sequence(
    sequence: str,
    quality_scores: torch.Tensor,
    profile: ErrorModelProfile,
    rng: torch.Generator,
    calibration: QualityCalibration | None = None,
    error_rate_scale: float = 1.0,
) -> tuple[str, str, list[tuple[int, int]]]:
    """Apply errors to a sequence based on quality scores.

    All random draws (error coin flips, error type selection,
    substitution/insertion base choices) are made in vectorised
    batches up front.  Only the sequential assembly of the output
    sequence and CIGAR string loops over reference positions.

    Parameters
    ----------
    sequence : str
        Original read sequence.
    quality_scores : torch.Tensor
        Quality scores (one per reference position).
    profile : ErrorModelProfile
        Error type ratios.
    rng : torch.Generator
        Random number generator.

    Returns
    -------
    modified_sequence : str
    quality_string : str
        Phred+33 encoded quality string for the modified sequence.
    cigar_tuples : list of (op, length)
        pysam-style CIGAR tuples.

    """
    ref_len = len(sequence)
    q_scores = quality_scores[:ref_len]

    # --- Vectorised random draws ---
    if calibration is not None:
        p_error = calibration(q_scores)
    else:
        p_error = 10.0 ** (-q_scores.float() / 10.0)
    if error_rate_scale != 1.0:
        p_error = (p_error * error_rate_scale).clamp(max=1.0)
    is_error = torch.rand(ref_len, generator=rng) < p_error

    error_type_weights = torch.tensor(
        [profile.substitution_ratio,
         profile.insertion_ratio,
         profile.deletion_ratio],
        dtype=torch.float64,
    )
    error_types = torch.multinomial(
        error_type_weights.expand(ref_len, -1), 1,
        generator=rng,
    ).squeeze(-1)

    # Pre-draw random bases for substitutions (index into 3
    # alternatives) and insertions (index into all 4 bases)
    sub_choices = torch.randint(
        0, 3, (ref_len,), generator=rng
    )
    ins_bases = torch.randint(
        0, 4, (ref_len,), generator=rng
    )

    # Move to Python lists once for the assembly loop
    is_error_list = is_error.tolist()
    error_types_list = error_types.tolist()
    sub_choices_list = sub_choices.tolist()
    ins_bases_list = ins_bases.tolist()
    q_int = q_scores.to(torch.int64).tolist()

    # --- Sequential assembly (insertions/deletions shift positions) ---
    modified_bases: list[str] = []
    qual_chars: list[str] = []
    cigar_ops: list[int] = []

    for pos in range(ref_len):
        q_char = chr(q_int[pos] + 33)

        if not is_error_list[pos]:
            modified_bases.append(sequence[pos])
            qual_chars.append(q_char)
            cigar_ops.append(0)  # M
        else:
            etype = error_types_list[pos]
            if etype == 0:
                # Substitution
                original = sequence[pos].upper()
                alts = [b for b in _BASES if b != original]
                modified_bases.append(alts[sub_choices_list[pos]])
                qual_chars.append(q_char)
                cigar_ops.append(0)  # M
            elif etype == 1:
                # Insertion then current base
                modified_bases.append(_BASES[ins_bases_list[pos]])
                qual_chars.append(q_char)
                cigar_ops.append(1)  # I
                modified_bases.append(sequence[pos])
                qual_chars.append(q_char)
                cigar_ops.append(0)  # M
            else:
                # Deletion
                cigar_ops.append(2)  # D

    # Run-length encode CIGAR ops
    cigar_tuples: list[tuple[int, int]] = []
    if cigar_ops:
        current_op = cigar_ops[0]
        current_len = 1
        for op in cigar_ops[1:]:
            if op == current_op:
                current_len += 1
            else:
                cigar_tuples.append((current_op, current_len))
                current_op = op
                current_len = 1
        cigar_tuples.append((current_op, current_len))

    return "".join(modified_bases), "".join(qual_chars), cigar_tuples


def load_genomes(
    csv_path: Path,
) -> tuple[dict[str, list], dict[str, float]]:
    """Parse input CSV and load FASTA files.

    Parameters
    ----------
    csv_path : Path
        CSV with columns: genome_id, fasta_path, abundance.

    Returns
    -------
    genomes : dict mapping genome_id to list of Bio.SeqRecord
    abundances : dict mapping genome_id to normalised abundance

    """
    genomes: dict[str, list] = {}
    raw_abundances: dict[str, float] = {}

    # Pre-scan to count rows for progress bar
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    logger.debug("Found %d entries in %s", len(rows), csv_path)

    with progress_task(len(rows), "Loading genomes") as step:
        for row in rows:
            genome_id = row["genome_id"]
            fasta_path = Path(row["fasta_path"])
            abundance = float(row["abundance"])

            logger.debug("Parsing FASTA %s for genome %s", fasta_path, genome_id)
            records = list(SeqIO.parse(fasta_path, "fasta"))
            if not records:
                logger.warning("No sequences found in %s", fasta_path)
                step()
                continue

            genomes[genome_id] = records
            raw_abundances[genome_id] = abundance
            logger.info(
                "Loaded %s: %d contigs from %s",
                genome_id,
                len(records),
                fasta_path,
            )
            for rec in records:
                logger.debug(
                    "  contig %s: %d bp, GC=%.1f%%",
                    rec.id,
                    len(rec.seq),
                    _gc_fraction(str(rec.seq)) * 100,
                )
            step()

    # Normalise abundances
    total = sum(raw_abundances.values())
    if total == 0:
        raise ValueError("Total abundance is zero; check input CSV")
    abundances = {gid: a / total for gid, a in raw_abundances.items()}

    return genomes, abundances


def _nb_params_from_mean_variance(
    mean: float, variance: float
) -> tuple[str, dict[str, float]]:
    """Convert mean/variance to NegativeBinomial or Poisson parameters.

    Returns
    -------
    dist_name : 'nb' or 'poisson'
    params : dict with keys appropriate for the distribution

    """
    if variance <= mean:
        logger.info(
            "Fragment variance (%.1f) <= mean (%.1f); "
            "falling back to Poisson distribution",
            variance,
            mean,
        )
        return "poisson", {"rate": mean}

    # Pyro NB parameterisation: mean = r*p/(1-p), var = r*p/(1-p)^2
    # Solving: r = mu^2 / (sigma^2 - mu), p = 1 - mu / sigma^2
    r = mean**2 / (variance - mean)
    p = 1.0 - mean / variance
    return "nb", {"total_count": r, "probs": p}


def _lognormal_params_from_mean_variance(
    mean: float, variance: float
) -> tuple[float, float]:
    """Convert desired mean/variance to LogNormal mu_ln, sigma_ln."""
    sigma2_ln = math.log(1 + variance / mean**2)
    mu_ln = math.log(mean) - sigma2_ln / 2
    sigma_ln = math.sqrt(sigma2_ln)
    return mu_ln, sigma_ln


def _gc_fraction(sequence: str) -> float:
    """Compute GC fraction of a sequence."""
    if not sequence:
        return 0.5
    upper = sequence.upper()
    gc = upper.count("G") + upper.count("C")
    return gc / len(sequence)


def _reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return str(Seq(sequence).reverse_complement())


def sample_fragments(
    genomes: dict[str, list],
    abundances: dict[str, float],
    num_fragments: int,
    fragment_mean: float,
    fragment_variance: float,
    gc_bias_strength: float,
    rng: torch.Generator,
) -> list[Fragment]:
    """Sample DNA fragments from genomes with optional GC bias.

    Parameters
    ----------
    genomes : dict mapping genome_id to list of SeqRecord
    abundances : dict mapping genome_id to normalised abundance
    num_fragments : int
        Number of fragments to generate.
    fragment_mean : float
        Mean fragment length.
    fragment_variance : float
        Variance of fragment length distribution.
    gc_bias_strength : float
        Strength of GC bias filtering. 0 = no bias.
    rng : torch.Generator
        Seeded random number generator.

    Returns
    -------
    list of Fragment

    """
    # Allocate fragments per genome proportional to abundance
    genome_ids = list(abundances.keys())
    abundance_values = torch.tensor(
        [abundances[gid] for gid in genome_ids], dtype=torch.float64
    )
    counts_float = abundance_values * num_fragments
    counts = counts_float.floor().long()
    # Distribute remainder by largest fractional parts
    remainder = num_fragments - counts.sum().item()
    fractional = counts_float - counts.float()
    if remainder > 0:
        _, top_idx = fractional.topk(int(remainder))
        for idx in top_idx:
            counts[idx] += 1

    # Set up fragment length distribution
    if fragment_variance == 0:
        frag_dist = None
        logger.debug(
            "Fragment length distribution: fixed at %d",
            int(fragment_mean),
        )
    else:
        dist_name, dist_params = _nb_params_from_mean_variance(
            fragment_mean, fragment_variance
        )
        if dist_name == "nb":
            frag_dist = NegativeBinomial(
                total_count=dist_params["total_count"],
                probs=dist_params["probs"],
            )
            logger.debug(
                "Fragment length distribution: NegativeBinomial"
                "(r=%.2f, p=%.4f)",
                dist_params["total_count"], dist_params["probs"],
            )
        else:
            frag_dist = Poisson(rate=dist_params["rate"])
            logger.debug(
                "Fragment length distribution: Poisson(rate=%.2f)",
                dist_params["rate"],
            )

    for gid in genome_ids:
        logger.debug(
            "  %s: target %d fragments (abundance=%.4f)",
            gid, counts[genome_ids.index(gid)].item(), abundances[gid],
        )

    fragments: list[Fragment] = []

    with progress_task(num_fragments, "Sampling fragments") as step:
        for genome_idx, genome_id in enumerate(genome_ids):
            n_frags = counts[genome_idx].item()
            if n_frags == 0:
                continue

            records = genomes[genome_id]
            contig_lengths = torch.tensor(
                [len(r.seq) for r in records], dtype=torch.float64
            )
            contig_weights = contig_lengths / contig_lengths.sum()

            accepted = 0
            rejected = 0
            max_attempts = n_frags * 20
            attempts = 0

            while accepted < n_frags and attempts < max_attempts:
                attempts += 1

                # Pick random contig weighted by length
                contig_idx = torch.multinomial(
                    contig_weights, 1, generator=rng
                ).item()
                record = records[contig_idx]
                contig_len = len(record.seq)

                # Sample fragment length
                if frag_dist is None:
                    frag_len = max(1, int(fragment_mean))
                else:
                    frag_len = int(
                        frag_dist.sample().clamp(min=1).item()
                    )
                if frag_len > contig_len:
                    rejected += 1
                    continue

                # Pick random start position
                max_start = contig_len - frag_len
                start = torch.randint(
                    0, max_start + 1, (1,), generator=rng
                ).item()
                end = start + frag_len

                # Pick random strand
                strand = (
                    "+"
                    if torch.rand(1, generator=rng).item() < 0.5
                    else "-"
                )

                # Extract sequence
                seq_str = str(record.seq[start:end])
                if strand == "-":
                    seq_str = _reverse_complement(seq_str)

                # GC bias accept/reject
                if gc_bias_strength > 0:
                    gc = _gc_fraction(seq_str)
                    p_keep = math.exp(
                        -gc_bias_strength * (gc - 0.5) ** 2
                    )
                    if torch.rand(1, generator=rng).item() > p_keep:
                        rejected += 1
                        continue

                fragments.append(
                    Fragment(
                        genome_id=genome_id,
                        contig_id=record.id,
                        start=start,
                        end=end,
                        strand=strand,
                        sequence=seq_str,
                    )
                )
                accepted += 1
                step()

            logger.debug(
                "  %s: accepted %d/%d fragments (%d rejected, "
                "%d attempts)",
                genome_id, accepted, n_frags, rejected, attempts,
            )

            if accepted < n_frags:
                logger.warning(
                    "Only generated %d/%d fragments for %s "
                    "(GC bias may be too strong or contigs too "
                    "short)",
                    accepted,
                    n_frags,
                    genome_id,
                )

    return fragments


def amplicon_fragments(
    genomes: dict[str, list],
    abundances: dict[str, float],
    num_fragments: int,
    rng: torch.Generator,
) -> list[Fragment]:
    """Create fragments from input sequences treated as amplicons.

    Each sequence record is used directly as a fragment (no shearing).
    Fragments are replicated proportionally to genome abundance to reach
    the requested total, modelling PCR amplification.

    Parameters
    ----------
    genomes : dict mapping genome_id to list of SeqRecord
    abundances : dict mapping genome_id to normalised abundance
    num_fragments : int
        Total number of fragments to produce.
    rng : torch.Generator
        Seeded random number generator (used for shuffling).

    Returns
    -------
    list of Fragment

    """
    # Build flat list of (genome_id, record) pairs with per-amplicon weights
    amplicons: list[tuple[str, any]] = []
    weights: list[float] = []
    for genome_id, records in genomes.items():
        for record in records:
            amplicons.append((genome_id, record))
            weights.append(abundances[genome_id])

    if not amplicons:
        return []

    # Normalise weights (they should already sum to ~1 across genomes,
    # but a genome with multiple records splits its weight evenly here)
    weight_tensor = torch.tensor(weights, dtype=torch.float64)
    weight_tensor = weight_tensor / weight_tensor.sum()

    # Largest-remainder allocation
    counts_float = weight_tensor * num_fragments
    counts = counts_float.floor().long()
    remainder = num_fragments - counts.sum().item()
    if remainder > 0:
        fractional = counts_float - counts.float()
        _, top_idx = fractional.topk(int(remainder))
        for idx in top_idx:
            counts[idx] += 1

    # Build fragment list
    fragments: list[Fragment] = []
    logger.debug("Amplicon allocation:")
    with progress_task(
        num_fragments, "Building amplicon fragments",
    ) as step:
        for i, (genome_id, record) in enumerate(amplicons):
            n = counts[i].item()
            if n == 0:
                continue
            logger.debug(
                "  %s:%s (%d bp) -> %d copies",
                genome_id, record.id, len(record.seq), n,
            )
            seq_str = str(record.seq)
            for _ in range(n):
                fragments.append(
                    Fragment(
                        genome_id=genome_id,
                        contig_id=record.id,
                        start=0,
                        end=len(record.seq),
                        strand="+",
                        sequence=seq_str,
                    )
                )
                step()

    # Shuffle so reads aren't grouped by amplicon
    n = len(fragments)
    indices = torch.randperm(n, generator=rng).tolist()
    fragments = [fragments[i] for i in indices]

    return fragments


def _generate_long_read(
    frag: Fragment, global_idx: int, read_len: int,
) -> Read:
    """Generate a long read spanning the entire fragment."""
    base_name = (
        f"{frag.genome_id}:{frag.contig_id}:"
        f"{frag.start}-{frag.end}:{frag.strand}"
    )
    seq = frag.sequence
    qual = "I" * len(seq)  # Q40 Phred+33
    return Read(
        name=f"{base_name} read_{global_idx}",
        sequence=seq,
        quality=qual,
    )


def _generate_se_read(
    frag: Fragment, global_idx: int, read_len: int,
) -> Read:
    """Generate a single-end read from a fragment."""
    base_name = (
        f"{frag.genome_id}:{frag.contig_id}:"
        f"{frag.start}-{frag.end}:{frag.strand}"
    )
    seq = frag.sequence[:read_len]
    qual = "I" * len(seq)  # Q40 Phred+33
    return Read(
        name=f"{base_name} read_{global_idx}",
        sequence=seq,
        quality=qual,
    )


def _generate_pe_read(
    frag: Fragment, global_idx: int, read_len: int,
) -> tuple[Read, Read]:
    """Generate a paired-end read pair from a fragment."""
    base_name = (
        f"{frag.genome_id}:{frag.contig_id}:"
        f"{frag.start}-{frag.end}:{frag.strand}"
    )
    frag_len = len(frag.sequence)
    r1_len = min(read_len, frag_len)
    r2_len = min(read_len, frag_len)

    r1_seq = frag.sequence[:r1_len]
    r2_seq = _reverse_complement(frag.sequence[-r2_len:])

    return (
        Read(
            name=f"{base_name}/1 read_{global_idx}",
            sequence=r1_seq,
            quality="I" * len(r1_seq),
        ),
        Read(
            name=f"{base_name}/2 read_{global_idx}",
            sequence=r2_seq,
            quality="I" * len(r2_seq),
        ),
    )


def generate_reads(
    fragments: list[Fragment],
    read_length_mean: float,
    read_length_variance: float,
    paired_end: bool,
    rng: torch.Generator,
    read_index_offset: int = 0,
    long_read: bool = False,
) -> ReadBatch:
    """Generate reads from fragments.

    Parameters
    ----------
    fragments : list of Fragment
    read_length_mean : float
        Mean read length.
    read_length_variance : float
        Variance of read length distribution.
    paired_end : bool
        If True, generate paired-end reads.
    rng : torch.Generator
        Seeded random number generator.
    read_index_offset : int
        Starting index for read naming, used when generating
        reads in chunks to ensure globally unique names.
    long_read : bool
        If True, each read spans the entire fragment (read
        length parameters are ignored).

    Returns
    -------
    ReadBatch

    """
    # rng is accepted for API consistency; torch uses global RNG state
    # which is seeded in main() via torch.manual_seed()
    _ = rng

    # Select read generator and set up length sampling
    if long_read:
        mode = "long"
        generate_read = _generate_long_read
        batch_key = "single"
        read_len_dist = None
    elif paired_end:
        mode = "paired-end"
        generate_read = _generate_pe_read
        batch_key = "paired"
    else:
        mode = "single-end"
        generate_read = _generate_se_read
        batch_key = "single"

    if not long_read:
        if read_length_variance == 0:
            read_len_dist = None
            logger.debug(
                "Read length distribution: fixed at %d",
                int(read_length_mean),
            )
        else:
            mu_ln, sigma_ln = (
                _lognormal_params_from_mean_variance(
                    read_length_mean, read_length_variance,
                )
            )
            read_len_dist = LogNormal(
                loc=mu_ln, scale=sigma_ln,
            )
            logger.debug(
                "Read length distribution: "
                "LogNormal(mu=%.4f, sigma=%.4f)",
                mu_ln, sigma_ln,
            )

    reads: list = []

    with progress_task(
        len(fragments), f"Generating {mode} reads",
    ) as step:
        for i, frag in enumerate(fragments):
            global_idx = read_index_offset + i
            frag_len = len(frag.sequence)
            if long_read:
                read_len = frag_len
            elif read_len_dist is None:
                read_len = min(
                    max(1, int(read_length_mean)), frag_len,
                )
            else:
                sampled_len = int(
                    read_len_dist.sample().clamp(min=1).item()
                )
                read_len = min(sampled_len, frag_len)
            reads.append(
                generate_read(frag, global_idx, read_len),
            )
            step()

    logger.debug("Generated %d %s reads", len(reads), mode)
    return ReadBatch(**{batch_key: reads})


def apply_error_model(
    read_batch: ReadBatch,
    profile: ErrorModelProfile | None,
    rng: torch.Generator,
    calibration: QualityCalibration | None = None,
    error_rate_scale: float = 1.0,
) -> ReadBatch:
    """Apply HMM-based sequencing error model to reads.

    If profile is None, returns reads unchanged (backwards compatible).
    Otherwise, samples quality scores from the HMM in a single batch
    and applies errors (substitutions, insertions, deletions) based on
    those quality scores.

    Parameters
    ----------
    read_batch : ReadBatch
        Generated reads (single-end or paired-end).
    profile : ErrorModelProfile or None
        Error model parameters. None means no errors applied.
    rng : torch.Generator
        Random number generator.

    Returns
    -------
    ReadBatch

    """
    if profile is None:
        logger.debug("No error model; skipping error application")
        return read_batch

    logger.info("Applying %s error model...", profile.name)

    # Collect all individual reads and their lengths for batch
    # quality score sampling
    flat_reads: list[Read] = []
    if read_batch.is_paired:
        assert read_batch.paired is not None
        for r1, r2 in read_batch.paired:
            flat_reads.append(r1)
            flat_reads.append(r2)
    else:
        assert read_batch.single is not None
        flat_reads = list(read_batch.single)

    read_lengths = [len(r.sequence) for r in flat_reads]
    logger.debug(
        "Sampling HMM quality scores for %d reads "
        "(max length %d)...",
        len(read_lengths),
        max(read_lengths) if read_lengths else 0,
    )
    all_q_scores = batch_sample_quality_scores(
        profile, read_lengths
    )
    logger.debug("Quality score sampling complete")

    # Apply errors using the pre-sampled quality scores
    modified: list[Read] = []
    n_errors_total = 0
    n_bases_total = 0
    with progress_task(len(flat_reads), "Applying errors") as step:
        for read, q_scores in zip(flat_reads, all_q_scores):
            new_seq, new_qual, cigar = apply_errors_to_sequence(
                read.sequence, q_scores, profile, rng, calibration,
                error_rate_scale,
            )
            n_bases_total += len(read.sequence)
            n_errors_total += sum(
                length for op, length in cigar if op != 0
            )
            modified.append(Read(
                name=read.name,
                sequence=new_seq,
                quality=new_qual,
                cigar=cigar,
            ))
            step()

    if n_bases_total > 0:
        logger.debug(
            "Error model summary: %d/%d bases affected (%.2f%%)",
            n_errors_total,
            n_bases_total,
            n_errors_total / n_bases_total * 100,
        )

    # Re-pack into ReadBatch
    batch_key = "paired" if read_batch.is_paired else "single"
    if read_batch.is_paired:
        result = [
            (modified[i], modified[i + 1])
            for i in range(0, len(modified), 2)
        ]
    else:
        result = modified
    return ReadBatch(**{batch_key: result})


def write_fastq(
    reads: list[Read],
    output_path: Path,
    append: bool = False,
) -> None:
    """Write reads to a FASTQ file with Phred+33 encoding.

    Parameters
    ----------
    reads : list of Read
    output_path : Path
        Output FASTQ file path.
    append : bool
        If True, append to existing file instead of overwriting.

    """
    mode = "a" if append else "w"
    logger.debug(
        "%s %d reads to %s",
        "Appending" if append else "Writing",
        len(reads),
        output_path,
    )
    with open(output_path, mode) as fh:
        with progress_task(
            len(reads), f"Writing {output_path.name}",
        ) as step:
            for read in reads:
                fh.write(f"@{read.name}\n")
                fh.write(f"{read.sequence}\n")
                fh.write("+\n")
                fh.write(f"{read.quality}\n")
                step()

    logger.info(
        "%s %d reads to %s",
        "Appended" if append else "Wrote",
        len(reads),
        output_path,
    )


def build_bam_header(
    genomes: dict[str, list],
) -> tuple[pysam.AlignmentHeader, dict[str, int]]:
    """Build a BAM header and reference name index from genomes.

    Parameters
    ----------
    genomes : dict mapping genome_id to list of Bio.SeqRecord
        Used to build the BAM header with contig lengths.

    Returns
    -------
    tuple of (AlignmentHeader, dict mapping ref_name to ref_id)

    """
    ref_names: list[str] = []
    ref_lengths: list[int] = []
    ref_name_to_idx: dict[str, int] = {}

    for genome_id, records in genomes.items():
        for record in records:
            name = f"{genome_id}:{record.id}"
            ref_name_to_idx[name] = len(ref_names)
            ref_names.append(name)
            ref_lengths.append(len(record.seq))

    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "SQ": [
            {"SN": name, "LN": length}
            for name, length in zip(ref_names, ref_lengths)
        ],
    })
    logger.debug(
        "BAM header: %d reference sequences", len(ref_names),
    )
    return header, ref_name_to_idx


def _ref_consumed(cigar: list[tuple[int, int]]) -> int:
    """Return the number of reference bases consumed by a CIGAR."""
    # M=0 and D=2 consume reference bases
    return sum(length for op, length in cigar if op in (0, 2))


def _bam_fields_for_read(
    read: Read,
    frag: Fragment,
    is_reverse: bool,
) -> tuple[str, str, list[tuple[int, int]], int]:
    """Compute BAM-ready sequence, quality, CIGAR, and ref_start.

    SAM stores query_sequence on the forward strand. When the read
    maps to the reverse strand, sequence and quality must be reversed
    and complemented, and the CIGAR must be reversed. The
    reference_start is computed from the fragment end for
    reverse-strand reads.

    Returns
    -------
    tuple of (query_sequence, quality, cigar, reference_start)

    """
    cigar = (
        read.cigar if read.cigar is not None
        else [(0, len(read.sequence))]
    )
    seq = read.sequence
    qual = read.quality
    ref_start = frag.start
    if is_reverse:
        seq = _reverse_complement(seq)
        qual = qual[::-1]
        cigar = list(reversed(cigar))
        ref_start = frag.end - _ref_consumed(cigar)
    return seq, qual, cigar, ref_start


def write_bam_chunk(
    bam: pysam.AlignmentFile,
    header: pysam.AlignmentHeader,
    ref_name_to_idx: dict[str, int],
    fragments: list[Fragment],
    read_batch: ReadBatch,
) -> None:
    """Write a chunk of ground-truth alignments to an open BAM file.

    Parameters
    ----------
    bam : pysam.AlignmentFile
        Open BAM file for writing.
    header : pysam.AlignmentHeader
        BAM header (needed to create AlignedSegment objects).
    ref_name_to_idx : dict mapping ref_name to ref_id
        Reference name to index mapping.
    fragments : list of Fragment
        Source fragments (one per read or read-pair).
    read_batch : ReadBatch
        Generated reads (single-end or paired-end).

    """
    def write_pe(i: int, frag: Fragment, ref_id: int) -> None:
        assert read_batch.paired is not None
        r1, r2 = read_batch.paired[i]
        qname = r1.name.split("/")[0]
        tlen = frag.end - frag.start

        r1_is_reverse = frag.strand == "-"
        r2_is_reverse = not r1_is_reverse

        r1_seq, r1_qual, r1_cigar, r1_start = (
            _bam_fields_for_read(r1, frag, r1_is_reverse)
        )
        r2_seq, r2_qual, r2_cigar, r2_start = (
            _bam_fields_for_read(r2, frag, r2_is_reverse)
        )

        if r1_start <= r2_start:
            r1_tlen, r2_tlen = tlen, -tlen
        else:
            r1_tlen, r2_tlen = -tlen, tlen

        a1 = pysam.AlignedSegment(header)
        a1.query_name = qname
        a1.query_sequence = r1_seq
        a1.flag = 0
        a1.is_paired = True
        a1.is_proper_pair = True
        a1.is_read1 = True
        a1.is_reverse = r1_is_reverse
        a1.mate_is_reverse = r2_is_reverse
        a1.reference_id = ref_id
        a1.reference_start = r1_start
        a1.cigar = r1_cigar
        a1.mapping_quality = 255
        a1.query_qualities = (
            pysam.qualitystring_to_array(r1_qual)
        )
        a1.next_reference_id = ref_id
        a1.next_reference_start = r2_start
        a1.template_length = r1_tlen

        a2 = pysam.AlignedSegment(header)
        a2.query_name = qname
        a2.query_sequence = r2_seq
        a2.flag = 0
        a2.is_paired = True
        a2.is_proper_pair = True
        a2.is_read2 = True
        a2.is_reverse = r2_is_reverse
        a2.mate_is_reverse = r1_is_reverse
        a2.reference_id = ref_id
        a2.reference_start = r2_start
        a2.cigar = r2_cigar
        a2.mapping_quality = 255
        a2.query_qualities = (
            pysam.qualitystring_to_array(r2_qual)
        )
        a2.next_reference_id = ref_id
        a2.next_reference_start = r1_start
        a2.template_length = r2_tlen

        bam.write(a1)
        bam.write(a2)

    def write_se(i: int, frag: Fragment, ref_id: int) -> None:
        assert read_batch.single is not None
        read = read_batch.single[i]
        is_reverse = frag.strand == "-"
        seq, qual, cigar, ref_start = (
            _bam_fields_for_read(read, frag, is_reverse)
        )
        a = pysam.AlignedSegment(header)
        a.query_name = read.name
        a.query_sequence = seq
        a.flag = 0
        a.is_reverse = is_reverse
        a.reference_id = ref_id
        a.reference_start = ref_start
        a.cigar = cigar
        a.mapping_quality = 255
        a.query_qualities = (
            pysam.qualitystring_to_array(qual)
        )
        bam.write(a)

    write_alignment = write_pe if read_batch.is_paired else write_se

    with progress_task(len(fragments), "Writing BAM") as step:
        for i, frag in enumerate(fragments):
            ref_name = f"{frag.genome_id}:{frag.contig_id}"
            ref_id = ref_name_to_idx[ref_name]
            write_alignment(i, frag, ref_id)
            step()


def write_bam(
    fragments: list[Fragment],
    read_batch: ReadBatch,
    genomes: dict[str, list],
    output_path: Path,
) -> None:
    """Write ground-truth alignments to a BAM file.

    Convenience wrapper that builds the header and writes all
    fragments in a single pass.

    Parameters
    ----------
    fragments : list of Fragment
        Source fragments (one per read or read-pair).
    read_batch : ReadBatch
        Generated reads (single-end or paired-end).
    genomes : dict mapping genome_id to list of Bio.SeqRecord
        Used to build the BAM header with contig lengths.
    output_path : Path
        Output BAM file path.

    """
    header, ref_name_to_idx = build_bam_header(genomes)
    with pysam.AlignmentFile(output_path, "wb", header=header) as bam:
        write_bam_chunk(
            bam, header, ref_name_to_idx, fragments, read_batch,
        )
    logger.info("Wrote ground-truth BAM to %s", output_path)


@app.command()
def main(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option(
        help="YAML config file (CLI options override config values)",
    )] = None,
    verbose: Annotated[bool, typer.Option(
        "--verbose/--no-verbose",
        help="Enable verbose (DEBUG) logging",
    )] = False,
    no_ansi: Annotated[bool, typer.Option(
        "--no-ansi",
        help="Disable ANSI escape codes (progress bars, colours)",
    )] = False,
    input_csv: Annotated[Path | None, typer.Option(
        help="CSV with columns: genome_id, fasta_path, abundance",
    )] = None,
    num_reads: Annotated[int | None, typer.Option(
        help="Total number of reads to generate",
    )] = None,
    output_prefix: Annotated[str | None, typer.Option(
        help="Output file prefix",
    )] = None,
    fragment_mean: Annotated[float, typer.Option(help="Mean fragment length")] = 300.0,
    fragment_variance: Annotated[float, typer.Option(
        help="Variance of fragment length (Negative Binomial)",
    )] = 300.0,
    read_length_mean: Annotated[float, typer.Option(
        help="Mean read length (LogNormal)",
    )] = 150.0,
    read_length_variance: Annotated[float, typer.Option(
        help="Variance of read length (LogNormal)",
    )] = 10.0,
    gc_bias_strength: Annotated[float, typer.Option(
        help="GC bias strength; 0 = no bias",
    )] = 0.0,
    paired_end: Annotated[bool, typer.Option(
        "--paired-end/--single-end",
        help="Generate paired-end or single-end reads (default: single-end)",
    )] = False,
    seed: Annotated[int | None, typer.Option(
        help="Random seed for reproducibility",
    )] = None,
    error_model: Annotated[ErrorModel, typer.Option(
        help="Sequencing error model profile",
        case_sensitive=False,
    )] = ErrorModel.none,
    quality_calibration_model: Annotated[QualityCalibrationModel, typer.Option(
        help="Quality-score-to-error-rate calibration model",
        case_sensitive=False,
    )] = QualityCalibrationModel.phred,
    qcal_variability: Annotated[float, typer.Option(
        help="Per-run noise multiplier for calibration parameters; 0 = no noise",
    )] = 0.0,
    qcal_intercept: Annotated[float, typer.Option(
        help="Log-linear model intercept (log10 scale)",
    )] = -0.3,
    qcal_slope: Annotated[float, typer.Option(
        help="Log-linear model slope",
    )] = -0.08,
    qcal_floor: Annotated[float, typer.Option(
        help="Minimum error probability (log-linear and sigmoid)",
    )] = 1e-7,
    qcal_ceiling: Annotated[float, typer.Option(
        help="Maximum error probability (log-linear and sigmoid)",
    )] = 0.5,
    qcal_steepness: Annotated[float, typer.Option(
        help="Sigmoid model steepness",
    )] = 0.25,
    qcal_midpoint: Annotated[float, typer.Option(
        help="Sigmoid model midpoint (Q-score at inflection)",
    )] = 15.0,
    error_rate_scale: Annotated[float, typer.Option(
        help="Multiplier applied to error probabilities after quality "
        "calibration; <1 reduces errors, >1 increases them",
    )] = 1.0,
    long_read: Annotated[bool, typer.Option(
        "--long-read/--no-long-read",
        help="Sequence entire fragments (read length params ignored)",
    )] = False,
    amplicon: Annotated[bool, typer.Option(
        "--amplicon/--no-amplicon",
        help="Treat input sequences as amplicons (no shearing); "
        "replicate proportionally to abundance",
    )] = False,
    chunk_size: Annotated[int, typer.Option(
        help="Number of fragments to process per chunk to limit "
        "memory usage",
    )] = 100_000,
) -> None:
    """Generate simulated WGS reads from reference genomes."""
    # Apply YAML config: values from the file fill in anything not
    # explicitly provided on the command line.
    if config is not None:
        _apply_yaml_config(ctx, _load_yaml_config(config))
        # Re-read parameters that may have been overridden by config.
        # ctx.params holds raw click values, so enum/Path types need
        # explicit conversion.
        p = ctx.params
        no_ansi = p.get("no_ansi", False)
        input_csv = Path(p["input_csv"]) if p.get("input_csv") else None
        num_reads = p.get("num_reads")
        output_prefix = p.get("output_prefix")
        fragment_mean = p["fragment_mean"]
        fragment_variance = p["fragment_variance"]
        read_length_mean = p["read_length_mean"]
        read_length_variance = p["read_length_variance"]
        gc_bias_strength = p["gc_bias_strength"]
        paired_end = p["paired_end"]
        seed = p.get("seed")
        error_model = ErrorModel(p["error_model"])
        quality_calibration_model = QualityCalibrationModel(
            p["quality_calibration_model"]
        )
        qcal_variability = p["qcal_variability"]
        qcal_intercept = p["qcal_intercept"]
        qcal_slope = p["qcal_slope"]
        qcal_floor = p["qcal_floor"]
        qcal_ceiling = p["qcal_ceiling"]
        qcal_steepness = p["qcal_steepness"]
        qcal_midpoint = p["qcal_midpoint"]
        error_rate_scale = p.get("error_rate_scale", 1.0)
        long_read = p.get("long_read", False)
        amplicon = p["amplicon"]
        chunk_size = p.get("chunk_size", 1_000_000)

    # Validate required parameters
    if input_csv is None:
        raise typer.BadParameter("--input-csv is required (via CLI or config)")
    if num_reads is None:
        raise typer.BadParameter("--num-reads is required (via CLI or config)")
    if output_prefix is None:
        raise typer.BadParameter("--output-prefix is required (via CLI or config)")
    if long_read and paired_end:
        raise typer.BadParameter(
            "--long-read and --paired-end are mutually exclusive"
        )

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.debug("Verbose logging enabled")

    # Log resolved parameters
    logger.debug("Parameters:")
    logger.debug("  input_csv       = %s", input_csv)
    logger.debug("  num_reads       = %d", num_reads)
    logger.debug("  output_prefix   = %s", output_prefix)
    logger.debug("  fragment_mean   = %.1f", fragment_mean)
    logger.debug("  fragment_var    = %.1f", fragment_variance)
    logger.debug("  read_len_mean   = %.1f", read_length_mean)
    logger.debug("  read_len_var    = %.1f", read_length_variance)
    logger.debug("  gc_bias         = %.2f", gc_bias_strength)
    logger.debug("  paired_end      = %s", paired_end)
    logger.debug("  long_read       = %s", long_read)
    logger.debug("  amplicon        = %s", amplicon)
    logger.debug("  chunk_size      = %d", chunk_size)
    logger.debug("  error_model     = %s", error_model.value)
    logger.debug("  quality_cal     = %s", quality_calibration_model.value)
    logger.debug("  seed            = %s", seed)

    # Set up RNG
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        logger.info("Random seed: %d", seed)
    else:
        rng.seed()
        logger.info("Using random seed")

    # Resolve error model profile
    profile_map = {
        "illumina": default_illumina_profile,
        "pacbio": default_pacbio_profile,
        "nanopore": default_nanopore_profile,
    }
    error_model_str = error_model.value
    profile = profile_map[error_model_str]() if error_model_str != "none" else None
    if profile is not None:
        logger.debug(
            "Error profile: %d HMM states, sub=%.0f%% ins=%.0f%% del=%.0f%%",
            profile.num_states,
            profile.substitution_ratio * 100,
            profile.insertion_ratio * 100,
            profile.deletion_ratio * 100,
        )

    # Build quality calibration model
    calibration = build_quality_calibration(
        model_name=quality_calibration_model.value,
        variability=qcal_variability,
        rng=rng,
        intercept=qcal_intercept,
        slope=qcal_slope,
        floor=qcal_floor,
        ceiling=qcal_ceiling,
        steepness=qcal_steepness,
        midpoint=qcal_midpoint,
    )

    # Determine whether to show progress bars
    try:
        is_tty = sys.stderr.isatty()
    except ValueError:
        is_tty = False
    disable_progress = no_ansi or not is_tty

    progress_columns = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.elapsed:.1f}s"),
    )
    outer_progress = Progress(*progress_columns)
    inner_progress = Progress(*progress_columns)

    global _inner_progress  # noqa: PLW0603

    pipeline_kwargs = dict(
        input_csv=input_csv,
        num_reads=num_reads,
        output_prefix=output_prefix,
        fragment_mean=fragment_mean,
        fragment_variance=fragment_variance,
        read_length_mean=read_length_mean,
        read_length_variance=read_length_variance,
        gc_bias_strength=gc_bias_strength,
        paired_end=paired_end,
        long_read=long_read,
        amplicon=amplicon,
        chunk_size=chunk_size,
        rng=rng,
        profile=profile,
        calibration=calibration,
        error_rate_scale=error_rate_scale,
    )

    if disable_progress:
        _inner_progress = None
        _run_pipeline(
            **pipeline_kwargs,
            outer_progress=None,
            inner_progress=None,
        )
    else:
        # Suppress log output while the Live display is active;
        # progress bars convey status instead.
        saved_level = logging.root.level
        logging.root.setLevel(logging.WARNING)
        with Live(
            Group(outer_progress, inner_progress),
            refresh_per_second=10,
        ):
            _inner_progress = inner_progress
            _run_pipeline(
                **pipeline_kwargs,
                outer_progress=outer_progress,
                inner_progress=inner_progress,
            )
            _inner_progress = None
        logging.root.setLevel(saved_level)

    logger.info("Done.")


def _run_pipeline(
    *,
    input_csv: Path,
    num_reads: int,
    output_prefix: str,
    fragment_mean: float,
    fragment_variance: float,
    read_length_mean: float,
    read_length_variance: float,
    gc_bias_strength: float,
    paired_end: bool,
    long_read: bool,
    amplicon: bool,
    chunk_size: int,
    rng: torch.Generator,
    profile: ErrorModelProfile | None,
    calibration: QualityCalibration | None,
    error_rate_scale: float,
    outer_progress: Progress | None,
    inner_progress: Progress | None,
) -> None:
    """Run the read-generation pipeline.

    Extracted from main() so the Live display context can be set
    up before this function is called.
    """
    genomes, abundances = load_genomes(input_csv)
    logger.info("Loaded %d genomes", len(genomes))
    for gid, abd in abundances.items():
        total_bp = sum(len(r.seq) for r in genomes[gid])
        logger.debug(
            "  %s: abundance=%.4f, %d contigs, %d bp total",
            gid, abd, len(genomes[gid]), total_bp,
        )

    num_fragments = num_reads // 2 if paired_end else num_reads

    chunk_starts = list(range(0, num_fragments, chunk_size))
    chunk_counts = [
        min(chunk_size, num_fragments - start)
        for start in chunk_starts
    ]
    num_chunks = len(chunk_counts)
    logger.info(
        "Processing %d fragments in %d chunk(s) of up to %d",
        num_fragments, num_chunks, chunk_size,
    )

    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    bam_path = Path(f"{output_prefix}.bam")
    if paired_end:
        r1_path = Path(f"{output_prefix}_R1.fastq")
        r2_path = Path(f"{output_prefix}_R2.fastq")
    else:
        out_path = Path(f"{output_prefix}.fastq")

    header, ref_name_to_idx = build_bam_header(genomes)

    chunk_task = None
    if outer_progress is not None:
        chunk_task = outer_progress.add_task(
            "Chunks", total=num_chunks,
        )

    # Set up fragment generator based on mode (amplicon vs shearing)
    if amplicon:
        def make_fragments(n: int) -> list[Fragment]:
            return amplicon_fragments(
                genomes=genomes,
                abundances=abundances,
                num_fragments=n,
                rng=rng,
            )
    else:
        def make_fragments(n: int) -> list[Fragment]:
            return sample_fragments(
                genomes=genomes,
                abundances=abundances,
                num_fragments=n,
                fragment_mean=fragment_mean,
                fragment_variance=fragment_variance,
                gc_bias_strength=gc_bias_strength,
                rng=rng,
            )

    # Set up FASTQ writer based on SE/PE mode
    if paired_end:
        def write_fastqs(
            batch: ReadBatch, append: bool,
        ) -> None:
            assert batch.paired is not None
            r1_reads = [p[0] for p in batch.paired]
            r2_reads = [p[1] for p in batch.paired]
            write_fastq(r1_reads, r1_path, append=append)
            write_fastq(r2_reads, r2_path, append=append)
    else:
        def write_fastqs(
            batch: ReadBatch, append: bool,
        ) -> None:
            assert batch.single is not None
            write_fastq(
                batch.single, out_path, append=append,
            )

    with pysam.AlignmentFile(
        bam_path, "wb", header=header,
    ) as bam:
        for chunk_idx, (chunk_start, chunk_n) in enumerate(
            zip(chunk_starts, chunk_counts)
        ):
            # Clear completed inner tasks from previous chunk
            if inner_progress is not None:
                for tid in list(inner_progress.task_ids):
                    inner_progress.remove_task(tid)

            logger.info(
                "Chunk %d/%d: %d fragments (offset %d)",
                chunk_idx + 1, num_chunks, chunk_n,
                chunk_start,
            )

            fragments = make_fragments(chunk_n)
            logger.info(
                "Generated %d fragments", len(fragments),
            )

            read_batch = generate_reads(
                fragments=fragments,
                read_length_mean=read_length_mean,
                read_length_variance=read_length_variance,
                paired_end=paired_end,
                rng=rng,
                read_index_offset=chunk_start,
                long_read=long_read,
            )
            read_batch = apply_error_model(
                read_batch, profile, rng, calibration,
                error_rate_scale,
            )

            write_fastqs(read_batch, append=chunk_idx > 0)

            write_bam_chunk(
                bam, header, ref_name_to_idx,
                fragments, read_batch,
            )

            if outer_progress is not None and chunk_task is not None:
                outer_progress.advance(chunk_task)

    logger.info("Wrote ground-truth BAM to %s", bam_path)


if __name__ == "__main__":
    app()
