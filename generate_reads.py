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
from dataclasses import dataclass
from pathlib import Path

import click
import pysam
import torch
from Bio import SeqIO
from Bio.Seq import Seq
from pyro.distributions import Categorical as PyroCategorical
from pyro.distributions import LogNormal, NegativeBinomial, Poisson
from pyro.distributions.hmm import DiscreteHMM

logger = logging.getLogger(__name__)


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


def sample_quality_scores(
    profile: ErrorModelProfile,
    read_length: int,
) -> torch.Tensor:
    """Sample a sequence of quality scores from the HMM error model.

    Parameters
    ----------
    profile : ErrorModelProfile
        HMM parameters and error ratios.
    read_length : int
        Number of quality scores to generate.

    Returns
    -------
    torch.Tensor of shape (read_length,) with integer quality values.

    """
    hmm = DiscreteHMM(
        initial_logits=profile.initial_logits,
        transition_logits=profile.transition_logits,
        observation_dist=PyroCategorical(logits=profile.emission_logits),
        duration=read_length,
    )
    return hmm.sample()


_BASES = "ACGT"


def apply_errors_to_sequence(
    sequence: str,
    quality_scores: torch.Tensor,
    profile: ErrorModelProfile,
    rng: torch.Generator,
) -> tuple[str, str, list[tuple[int, int]]]:
    """Apply errors to a sequence based on quality scores.

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
    error_type_weights = torch.tensor(
        [profile.substitution_ratio, profile.insertion_ratio, profile.deletion_ratio],
        dtype=torch.float64,
    )

    modified_bases: list[str] = []
    qual_chars: list[str] = []
    cigar_ops: list[int] = []  # per-base ops before run-length encoding

    ref_len = len(sequence)
    q_scores = quality_scores[:ref_len]

    for pos in range(ref_len):
        q = q_scores[pos].item()
        p_error = 10.0 ** (-q / 10.0)

        is_error = torch.rand(1, generator=rng).item() < p_error

        if not is_error:
            # Correct base
            modified_bases.append(sequence[pos])
            qual_chars.append(chr(int(q) + 33))
            cigar_ops.append(0)  # M
        else:
            # Determine error type
            error_type = torch.multinomial(
                error_type_weights, 1, generator=rng
            ).item()

            if error_type == 0:
                # Substitution: replace with a different base
                original = sequence[pos].upper()
                alternatives = [b for b in _BASES if b != original]
                idx = torch.randint(0, len(alternatives), (1,), generator=rng).item()
                modified_bases.append(alternatives[idx])
                qual_chars.append(chr(int(q) + 33))
                cigar_ops.append(0)  # M (substitutions are alignment matches)
            elif error_type == 1:
                # Insertion: insert a random base, then emit the current base
                ins_idx = torch.randint(0, 4, (1,), generator=rng).item()
                modified_bases.append(_BASES[ins_idx])
                qual_chars.append(chr(int(q) + 33))
                cigar_ops.append(1)  # I
                # Also emit the current reference base
                modified_bases.append(sequence[pos])
                qual_chars.append(chr(int(q) + 33))
                cigar_ops.append(0)  # M
            else:
                # Deletion: skip this reference base
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

    modified_sequence = "".join(modified_bases)
    quality_string = "".join(qual_chars)

    return modified_sequence, quality_string, cigar_tuples


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

    with open(csv_path) as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            genome_id = row["genome_id"]
            fasta_path = Path(row["fasta_path"])
            abundance = float(row["abundance"])

            records = list(SeqIO.parse(fasta_path, "fasta"))
            if not records:
                logger.warning("No sequences found in %s", fasta_path)
                continue

            genomes[genome_id] = records
            raw_abundances[genome_id] = abundance
            logger.info(
                "Loaded %s: %d contigs from %s",
                genome_id,
                len(records),
                fasta_path,
            )

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
        logger.warning(
            "Fragment variance (%.1f) <= mean (%.1f); "
            "falling back to Poisson distribution",
            variance,
            mean,
        )
        return "poisson", {"rate": mean}

    # NB parameterisation: total_count = r = mu^2 / (sigma^2 - mu)
    # probs = p = mu / sigma^2
    r = mean**2 / (variance - mean)
    p = mean / variance
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
    dist_name, dist_params = _nb_params_from_mean_variance(
        fragment_mean, fragment_variance
    )
    if dist_name == "nb":
        frag_dist = NegativeBinomial(
            total_count=dist_params["total_count"],
            probs=dist_params["probs"],
        )
    else:
        frag_dist = Poisson(rate=dist_params["rate"])

    fragments: list[Fragment] = []

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
        max_attempts = n_frags * 20  # safety limit to avoid infinite loops
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
            frag_len = int(frag_dist.sample().clamp(min=1).item())
            if frag_len > contig_len:
                continue

            # Pick random start position
            max_start = contig_len - frag_len
            start = torch.randint(
                0, max_start + 1, (1,), generator=rng
            ).item()
            end = start + frag_len

            # Pick random strand
            strand = "+" if torch.rand(1, generator=rng).item() < 0.5 else "-"

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

        if accepted < n_frags:
            logger.warning(
                "Only generated %d/%d fragments for %s "
                "(GC bias may be too strong or contigs too short)",
                accepted,
                n_frags,
                genome_id,
            )

    return fragments


def generate_reads(
    fragments: list[Fragment],
    read_length_mean: float,
    read_length_variance: float,
    paired_end: bool,
    rng: torch.Generator,
) -> list[Read] | list[tuple[Read, Read]]:  # noqa: E501
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

    Returns
    -------
    list of Read (single-end) or list of (Read, Read) tuples (paired-end)

    """
    # rng is accepted for API consistency; torch uses global RNG state
    # which is seeded in main() via torch.manual_seed()
    _ = rng

    mu_ln, sigma_ln = _lognormal_params_from_mean_variance(
        read_length_mean, read_length_variance
    )
    read_len_dist = LogNormal(loc=mu_ln, scale=sigma_ln)

    se_reads: list[Read] = []
    pe_reads: list[tuple[Read, Read]] = []

    for i, frag in enumerate(fragments):
        frag_len = len(frag.sequence)
        sampled_len = int(read_len_dist.sample().clamp(min=1).item())
        read_len = min(sampled_len, frag_len)

        base_name = (
            f"{frag.genome_id}:{frag.contig_id}:"
            f"{frag.start}-{frag.end}:{frag.strand}"
        )

        if paired_end:
            r1_len = min(read_len, frag_len)
            r2_len = min(read_len, frag_len)

            r1_seq = frag.sequence[:r1_len]
            r2_seq = _reverse_complement(frag.sequence[-r2_len:])

            r1_qual = "I" * len(r1_seq)  # Q40 in Phred+33
            r2_qual = "I" * len(r2_seq)

            r1 = Read(
                name=f"{base_name}/1 read_{i}",
                sequence=r1_seq,
                quality=r1_qual,
            )
            r2 = Read(
                name=f"{base_name}/2 read_{i}",
                sequence=r2_seq,
                quality=r2_qual,
            )
            pe_reads.append((r1, r2))
        else:
            seq = frag.sequence[:read_len]
            qual = "I" * len(seq)  # Q40 in Phred+33
            se_reads.append(
                Read(
                    name=f"{base_name} read_{i}",
                    sequence=seq,
                    quality=qual,
                )
            )

    if paired_end:
        return pe_reads
    return se_reads


def apply_error_model(
    reads: list[Read] | list[tuple[Read, Read]],
    fragments: list[Fragment],
    profile: ErrorModelProfile | None,
    paired_end: bool,
    rng: torch.Generator,
) -> list[Read] | list[tuple[Read, Read]]:
    """Apply HMM-based sequencing error model to reads.

    If profile is None, returns reads unchanged (backwards compatible).
    Otherwise, samples quality scores from the HMM and applies errors
    (substitutions, insertions, deletions) based on those quality scores.

    Parameters
    ----------
    reads : list of Read (SE) or list of (Read, Read) tuples (PE)
    fragments : list of Fragment
        Source fragments for the reads.
    profile : ErrorModelProfile or None
        Error model parameters. None means no errors applied.
    paired_end : bool
        Whether reads are paired-end.
    rng : torch.Generator
        Random number generator.

    Returns
    -------
    list of Read (SE) or list of (Read, Read) tuples (PE)

    """
    if profile is None:
        return reads

    logger.info("Applying %s error model...", profile.name)

    def _apply_to_read(read: Read) -> Read:
        q_scores = sample_quality_scores(profile, len(read.sequence))
        new_seq, new_qual, cigar = apply_errors_to_sequence(
            read.sequence, q_scores, profile, rng
        )
        return Read(
            name=read.name,
            sequence=new_seq,
            quality=new_qual,
            cigar=cigar,
        )

    if paired_end:
        result_pe: list[tuple[Read, Read]] = []
        for r1, r2 in reads:  # type: ignore[misc]
            result_pe.append((_apply_to_read(r1), _apply_to_read(r2)))
        return result_pe
    else:
        return [_apply_to_read(r) for r in reads]  # type: ignore[union-attr]


def write_fastq(reads: list[Read], output_path: Path) -> None:
    """Write reads to a FASTQ file with Phred+33 encoding.

    Parameters
    ----------
    reads : list of Read
    output_path : Path
        Output FASTQ file path.

    """
    with open(output_path, "w") as fh:
        for read in reads:
            fh.write(f"@{read.name}\n")
            fh.write(f"{read.sequence}\n")
            fh.write("+\n")
            fh.write(f"{read.quality}\n")

    logger.info("Wrote %d reads to %s", len(reads), output_path)


def write_bam(
    fragments: list[Fragment],
    reads: list[Read] | list[tuple[Read, Read]],
    paired_end: bool,
    genomes: dict[str, list],
    output_path: Path,
) -> None:
    """Write ground-truth alignments to a BAM file.

    Each read is placed at the genomic position of the fragment it was
    derived from, with a simple all-match CIGAR string.

    Parameters
    ----------
    fragments : list of Fragment
        Source fragments (one per read or read-pair).
    reads : list of Read (SE) or list of (Read, Read) tuples (PE)
    paired_end : bool
        Whether reads are paired-end.
    genomes : dict mapping genome_id to list of Bio.SeqRecord
        Used to build the BAM header with contig lengths.
    output_path : Path
        Output BAM file path.

    """
    # Build reference name list and header @SQ entries
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

    with pysam.AlignmentFile(output_path, "wb", header=header) as bam:
        for i, frag in enumerate(fragments):
            ref_name = f"{frag.genome_id}:{frag.contig_id}"
            ref_id = ref_name_to_idx[ref_name]

            if paired_end:
                r1, r2 = reads[i]  # type: ignore[misc]
                # Strip /1, /2 suffixes — BAM uses FLAG bits
                qname = r1.name.split("/")[0]

                r1_is_reverse = frag.strand == "-"
                r2_is_reverse = not r1_is_reverse

                # R1
                a1 = pysam.AlignedSegment(header)
                a1.query_name = qname
                a1.query_sequence = r1.sequence
                a1.flag = 0
                a1.is_paired = True
                a1.is_proper_pair = True
                a1.is_read1 = True
                a1.is_reverse = r1_is_reverse
                a1.mate_is_reverse = r2_is_reverse
                a1.reference_id = ref_id
                a1.reference_start = frag.start
                a1.cigar = r1.cigar if r1.cigar is not None else [(0, len(r1.sequence))]
                a1.mapping_quality = 255
                a1.query_qualities = pysam.qualitystring_to_array(r1.quality)
                a1.next_reference_id = ref_id
                # R2 aligns at end of fragment minus its length
                r2_start = frag.end - len(r2.sequence)
                a1.next_reference_start = r2_start
                a1.template_length = frag.end - frag.start

                # R2
                a2 = pysam.AlignedSegment(header)
                a2.query_name = qname
                a2.query_sequence = r2.sequence
                a2.flag = 0
                a2.is_paired = True
                a2.is_proper_pair = True
                a2.is_read2 = True
                a2.is_reverse = r2_is_reverse
                a2.mate_is_reverse = r1_is_reverse
                a2.reference_id = ref_id
                a2.reference_start = r2_start
                a2.cigar = r2.cigar if r2.cigar is not None else [(0, len(r2.sequence))]
                a2.mapping_quality = 255
                a2.query_qualities = pysam.qualitystring_to_array(r2.quality)
                a2.next_reference_id = ref_id
                a2.next_reference_start = frag.start
                a2.template_length = -(frag.end - frag.start)

                bam.write(a1)
                bam.write(a2)
            else:
                read = reads[i]  # type: ignore[assignment]
                a = pysam.AlignedSegment(header)
                a.query_name = read.name
                a.query_sequence = read.sequence
                a.flag = 0
                a.is_reverse = frag.strand == "-"
                a.reference_id = ref_id
                a.reference_start = frag.start
                a.cigar = read.cigar if read.cigar is not None else [(0, len(read.sequence))]
                a.mapping_quality = 255
                a.query_qualities = pysam.qualitystring_to_array(read.quality)
                bam.write(a)

    logger.info("Wrote ground-truth BAM to %s", output_path)


@click.command()
@click.option(
    "--input-csv",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="CSV with columns: genome_id, fasta_path, abundance",
)
@click.option(
    "--num-reads",
    required=True,
    type=int,
    help="Total number of reads to generate",
)
@click.option(
    "--fragment-mean",
    default=300.0,
    type=float,
    help="Mean fragment length",
)
@click.option(
    "--fragment-variance",
    default=300.0,
    type=float,
    help="Variance of fragment length (Negative Binomial)",
)
@click.option(
    "--read-length-mean",
    default=150.0,
    type=float,
    help="Mean read length (LogNormal)",
)
@click.option(
    "--read-length-variance",
    default=10.0,
    type=float,
    help="Variance of read length (LogNormal)",
)
@click.option(
    "--gc-bias-strength",
    default=0.0,
    type=float,
    help="GC bias strength; 0 = no bias",
)
@click.option(
    "--paired-end/--single-end",
    default=False,
    help="Generate paired-end or single-end reads (default: single-end)",
)
@click.option(
    "--output-prefix",
    required=True,
    type=str,
    help="Output file prefix",
)
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--error-model",
    type=click.Choice(["none", "illumina", "pacbio", "nanopore"], case_sensitive=False),
    default="none",
    help="Sequencing error model profile",
)
def main(
    input_csv: Path,
    num_reads: int,
    fragment_mean: float,
    fragment_variance: float,
    read_length_mean: float,
    read_length_variance: float,
    gc_bias_strength: float,
    paired_end: bool,
    output_prefix: str,
    seed: int | None,
    error_model: str,
) -> None:
    """Generate simulated WGS reads from reference genomes."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

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
    profile = profile_map[error_model]() if error_model != "none" else None

    # Load genomes
    genomes, abundances = load_genomes(input_csv)
    logger.info("Loaded %d genomes", len(genomes))

    # For paired-end, each pair counts as 2 reads towards num_reads,
    # so we need num_reads // 2 fragments for PE, num_reads for SE
    num_fragments = num_reads // 2 if paired_end else num_reads

    # Sample fragments
    logger.info("Sampling %d fragments...", num_fragments)
    fragments = sample_fragments(
        genomes=genomes,
        abundances=abundances,
        num_fragments=num_fragments,
        fragment_mean=fragment_mean,
        fragment_variance=fragment_variance,
        gc_bias_strength=gc_bias_strength,
        rng=rng,
    )
    logger.info("Generated %d fragments", len(fragments))

    # Generate reads and write output
    logger.info("Generating reads (paired_end=%s)...", paired_end)
    if paired_end:
        pe_reads = generate_reads(
            fragments=fragments,
            read_length_mean=read_length_mean,
            read_length_variance=read_length_variance,
            paired_end=True,
            rng=rng,
        )
        pe_reads = apply_error_model(
            pe_reads, fragments, profile, paired_end=True, rng=rng
        )
        # pe_reads is list[tuple[Read, Read]]
        r1_path = Path(f"{output_prefix}_R1.fastq")
        r2_path = Path(f"{output_prefix}_R2.fastq")
        r1_reads = [pair[0] for pair in pe_reads]  # type: ignore[union-attr]
        r2_reads = [pair[1] for pair in pe_reads]  # type: ignore[union-attr]
        write_fastq(r1_reads, r1_path)
        write_fastq(r2_reads, r2_path)
        bam_path = Path(f"{output_prefix}.bam")
        write_bam(fragments, pe_reads, paired_end=True, genomes=genomes, output_path=bam_path)
    else:
        se_reads = generate_reads(
            fragments=fragments,
            read_length_mean=read_length_mean,
            read_length_variance=read_length_variance,
            paired_end=False,
            rng=rng,
        )
        se_reads = apply_error_model(
            se_reads, fragments, profile, paired_end=False, rng=rng
        )
        # se_reads is list[Read]
        out_path = Path(f"{output_prefix}.fastq")
        write_fastq(se_reads, out_path)  # type: ignore[arg-type]
        bam_path = Path(f"{output_prefix}.bam")
        write_bam(fragments, se_reads, paired_end=False, genomes=genomes, output_path=bam_path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
