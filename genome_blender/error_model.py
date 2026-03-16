"""HMM-based sequencing error model and quality calibration.

Build error model profiles, sample quality scores via a discrete
HMM, and apply position-wise errors (substitutions, insertions,
deletions) to read sequences.
"""

from __future__ import annotations

import logging

import torch
from pyro.distributions import Categorical as PyroCategorical
from pyro.distributions.hmm import DiscreteHMM

from genome_blender._progress import progress_task
from genome_blender._utils import _BASES
from genome_blender.models import (
    ErrorModelProfile,
    LogLinearCalibration,
    PhredCalibration,
    QualityCalibration,
    Read,
    ReadBatch,
    SigmoidCalibration,
)

logger = logging.getLogger(__name__)

# Optional Rust extension: parallel HMM quality sampling + error assembly
try:
    from genome_blender_core import (  # type: ignore[import-not-found]
        apply_errors_batch as _apply_errors_batch_rust,
        sample_and_apply_errors_batch as _sample_and_apply_errors_batch_rust,
    )
    _HAS_RUST = True
    logger.debug("genome_blender_core Rust extension loaded")
except ImportError:
    _HAS_RUST = False
    logger.debug(
        "genome_blender_core not found; "
        "using pure-Python error assembly"
    )

# Precomputed substitution alternatives for each base (opt 7)
_SUBSTITUTION_ALTS: dict[str, list[str]] = {
    b: [x for x in "ACGT" if x != b] for b in "ACGT"
}

# Cache for DiscreteHMM objects keyed by (profile_id, duration) (opt 5)
_hmm_cache: dict[tuple[int, int], DiscreteHMM] = {}

# Noise scales for per-run variability perturbation
_QCAL_NOISE_SCALES: dict[str, float] = {
    "intercept": 0.15,
    "slope": 0.01,
    "steepness": 0.05,
    "midpoint": 2.0,
}


# ------------------------------------------------------------------
# Quality calibration builder
# ------------------------------------------------------------------

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

    Args:
        model_name: One of ``"phred"``, ``"log-linear"``,
            ``"sigmoid"``.
        variability: Multiplier for per-run parameter noise.
            0 means no perturbation.
        rng: Random number generator for reproducible noise.
        intercept: Log-linear model intercept.
        slope: Log-linear model slope.
        floor: Minimum error probability.
        ceiling: Maximum error probability.
        steepness: Sigmoid model steepness.
        midpoint: Sigmoid model midpoint.

    Returns:
        Configured ``QualityCalibration`` instance.

    Raises:
        ValueError: If *model_name* is not recognised.
    """
    if variability > 0:
        noise = torch.randn(4, generator=rng)
        intercept += (
            variability * _QCAL_NOISE_SCALES["intercept"]
            * noise[0].item()
        )
        slope += (
            variability * _QCAL_NOISE_SCALES["slope"]
            * noise[1].item()
        )
        steepness += (
            variability * _QCAL_NOISE_SCALES["steepness"]
            * noise[2].item()
        )
        midpoint += (
            variability * _QCAL_NOISE_SCALES["midpoint"]
            * noise[3].item()
        )

    if model_name == "phred":
        logger.info("Quality calibration: phred (theoretical)")
        return PhredCalibration()

    if model_name == "log-linear":
        logger.info(
            "Quality calibration: log-linear "
            "(intercept=%.4f, slope=%.4f, "
            "floor=%.1e, ceiling=%.2f)",
            intercept, slope, floor, ceiling,
        )
        return LogLinearCalibration(
            intercept=intercept, slope=slope,
            floor=floor, ceiling=ceiling,
        )

    if model_name == "sigmoid":
        logger.info(
            "Quality calibration: sigmoid "
            "(steepness=%.4f, midpoint=%.2f, "
            "floor=%.1e, ceiling=%.2f)",
            steepness, midpoint, floor, ceiling,
        )
        return SigmoidCalibration(
            steepness=steepness, midpoint=midpoint,
            floor=floor, ceiling=ceiling,
        )

    raise ValueError(
        f"Unknown quality calibration model: {model_name!r}"
    )


# ------------------------------------------------------------------
# HMM emission / transition builders
# ------------------------------------------------------------------

def _build_emission_logits(
    num_states: int,
    quality_peaks: list[float],
    quality_spreads: list[float],
    num_quality_values: int = 94,
) -> torch.Tensor:
    """Build emission logits for HMM states.

    Each state emits quality scores centred around a peak value
    with a Gaussian-shaped distribution over the quality range.

    Returns:
        Tensor of shape ``(num_states, num_quality_values)``.
    """
    q_values = torch.arange(
        num_quality_values, dtype=torch.float32,
    )
    logits = torch.zeros(
        num_states, num_quality_values, dtype=torch.float32,
    )
    for s in range(num_states):
        logits[s] = (
            -0.5
            * ((q_values - quality_peaks[s])
               / quality_spreads[s]) ** 2
        )
    return logits


def _build_sticky_transitions(
    num_states: int,
    self_logit: float = 3.0,
    neighbour_logit: float = 0.5,
) -> torch.Tensor:
    """Build transition logits favouring self and nearby states.

    Returns:
        Tensor of shape ``(num_states, num_states)``.
    """
    logits = torch.full(
        (num_states, num_states), -2.0, dtype=torch.float32,
    )
    for s in range(num_states):
        logits[s, s] = self_logit
        if s > 0:
            logits[s, s - 1] = neighbour_logit
        if s < num_states - 1:
            logits[s, s + 1] = neighbour_logit
    return logits


# ------------------------------------------------------------------
# Default platform profiles
# ------------------------------------------------------------------

def default_illumina_profile() -> ErrorModelProfile:
    """Return a default Illumina-like error model profile.

    States represent quality zones: very-high (Q37), high (Q33),
    medium (Q25), low (Q15), very-low (Q5).  Errors are
    predominantly substitutions (~80%).
    """
    num_states = 5
    quality_peaks = [37.0, 33.0, 25.0, 15.0, 5.0]
    quality_spreads = [3.0, 3.0, 4.0, 4.0, 3.0]

    initial_logits = torch.tensor(
        [2.0, 1.0, -1.0, -3.0, -5.0], dtype=torch.float32,
    )
    transition_logits = _build_sticky_transitions(
        num_states, self_logit=3.5, neighbour_logit=0.5,
    )
    emission_logits = _build_emission_logits(
        num_states, quality_peaks, quality_spreads,
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

    States represent quality zones centred around Q10-Q20.  Errors
    are predominantly insertions and deletions (~85%).
    """
    num_states = 5
    quality_peaks = [20.0, 15.0, 10.0, 7.0, 3.0]
    quality_spreads = [3.0, 3.0, 3.0, 3.0, 2.0]

    initial_logits = torch.tensor(
        [1.0, 2.0, 1.0, -1.0, -3.0], dtype=torch.float32,
    )
    transition_logits = _build_sticky_transitions(
        num_states, self_logit=3.0, neighbour_logit=0.8,
    )
    emission_logits = _build_emission_logits(
        num_states, quality_peaks, quality_spreads,
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

    States represent quality zones centred around Q7-Q15.  Errors
    are predominantly insertions and deletions (~80%).
    """
    num_states = 5
    quality_peaks = [15.0, 12.0, 9.0, 6.0, 3.0]
    quality_spreads = [3.0, 3.0, 3.0, 3.0, 2.0]

    initial_logits = torch.tensor(
        [0.5, 1.5, 1.0, -0.5, -2.0], dtype=torch.float32,
    )
    transition_logits = _build_sticky_transitions(
        num_states, self_logit=2.5, neighbour_logit=1.0,
    )
    emission_logits = _build_emission_logits(
        num_states, quality_peaks, quality_spreads,
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


# ------------------------------------------------------------------
# Quality-score sampling
# ------------------------------------------------------------------

def batch_sample_quality_scores(
    profile: ErrorModelProfile,
    read_lengths: list[int],
) -> list[torch.Tensor]:
    """Sample quality scores for a batch of reads from the HMM.

    Construct a single ``DiscreteHMM`` at the maximum read length
    and sample all sequences in one call, then truncate each to
    its actual length.

    Args:
        profile: HMM parameters and error ratios.
        read_lengths: Length of each read.

    Returns:
        List of tensors, each of shape ``(read_length,)`` with
        integer quality values.
    """
    if not read_lengths:
        return []

    max_len = max(read_lengths)
    batch_size = len(read_lengths)

    cache_key = (id(profile), max_len)
    if cache_key not in _hmm_cache:
        _hmm_cache[cache_key] = DiscreteHMM(
            initial_logits=profile.initial_logits,
            transition_logits=profile.transition_logits,
            observation_dist=PyroCategorical(
                logits=profile.emission_logits,
            ),
            duration=max_len,
        )
    hmm = _hmm_cache[cache_key]

    all_scores = hmm.sample(sample_shape=(batch_size,))

    return [
        all_scores[i, :length]
        for i, length in enumerate(read_lengths)
    ]


# ------------------------------------------------------------------
# Error application
# ------------------------------------------------------------------

def apply_errors_to_sequence(
    sequence: str,
    quality_scores: torch.Tensor,
    profile: ErrorModelProfile,
    rng: torch.Generator,
    calibration: QualityCalibration | None = None,
    error_rate_scale: float = 1.0,
) -> tuple[str, str, list[tuple[int, int]]]:
    """Apply errors to a sequence based on quality scores.

    All random draws are made in vectorised batches up front.
    Only the sequential assembly of the output sequence and CIGAR
    string loops over reference positions.

    Returns:
        Tuple of (modified_sequence, quality_string,
        cigar_tuples).
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

    sub_choices = torch.randint(
        0, 3, (ref_len,), generator=rng,
    )
    ins_bases = torch.randint(
        0, 4, (ref_len,), generator=rng,
    )

    is_error_list = is_error.tolist()
    error_types_list = error_types.tolist()
    sub_choices_list = sub_choices.tolist()
    ins_bases_list = ins_bases.tolist()
    q_int = q_scores.to(torch.int64).tolist()

    # --- Sequential assembly ---
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
                alts = _SUBSTITUTION_ALTS.get(
                    sequence[pos].upper(),
                    _SUBSTITUTION_ALTS["A"],
                )
                modified_bases.append(
                    alts[sub_choices_list[pos]],
                )
                qual_chars.append(q_char)
                cigar_ops.append(0)  # M
            elif etype == 1:
                # Insertion then current base
                modified_bases.append(
                    _BASES[ins_bases_list[pos]],
                )
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

    return (
        "".join(modified_bases),
        "".join(qual_chars),
        cigar_tuples,
    )


def _get_calibration_params(
    calibration: QualityCalibration | None,
) -> tuple[str, list[float]]:
    """Extract calibration type string and parameter list for the Rust extension."""
    if calibration is None or isinstance(calibration, PhredCalibration):
        return ("phred", [])
    if isinstance(calibration, LogLinearCalibration):
        return (
            "log-linear",
            [
                calibration.intercept,
                calibration.slope,
                calibration.floor,
                calibration.ceiling,
            ],
        )
    if isinstance(calibration, SigmoidCalibration):
        return (
            "sigmoid",
            [
                calibration.steepness,
                calibration.midpoint,
                calibration.floor,
                calibration.ceiling,
            ],
        )
    return ("phred", [])


def _apply_errors_rust(
    flat_reads: list[Read],
    all_q_scores: list[torch.Tensor],
    profile: ErrorModelProfile,
    rng: torch.Generator,
    calibration: QualityCalibration | None,
    error_rate_scale: float,
) -> tuple[list[Read], int, int]:
    """Batch-compute all random decisions then call the Rust extension.

    Returns (modified_reads, n_errors_total, n_bases_total).
    """
    lengths = [len(r.sequence) for r in flat_reads]
    total_len = sum(lengths)

    # Concatenate quality scores across all reads (already truncated
    # to each read's length by batch_sample_quality_scores)
    all_q_cat = torch.cat(
        [q[:lengths[i]] for i, q in enumerate(all_q_scores)]
    )

    # Vectorised error probability for the whole batch at once
    if calibration is not None:
        p_error = calibration(all_q_cat)
    else:
        p_error = 10.0 ** (-all_q_cat.float() / 10.0)
    if error_rate_scale != 1.0:
        p_error = (p_error * error_rate_scale).clamp(max=1.0)

    is_error = torch.rand(total_len, generator=rng) < p_error

    error_type_weights = torch.tensor(
        [
            profile.substitution_ratio,
            profile.insertion_ratio,
            profile.deletion_ratio,
        ],
        dtype=torch.float32,
    )
    error_types = torch.multinomial(
        error_type_weights.expand(total_len, -1),
        1, generator=rng,
    ).squeeze(-1)

    sub_choices = torch.randint(0, 3, (total_len,), generator=rng)
    ins_bases = torch.randint(0, 4, (total_len,), generator=rng)
    q_ints = all_q_cat.to(torch.int64)

    # Convert to plain Python lists for PyO3 (one pass each)
    sequences = [r.sequence for r in flat_reads]
    results = _apply_errors_batch_rust(
        sequences,
        q_ints.tolist(),
        is_error.tolist(),
        error_types.tolist(),
        sub_choices.tolist(),
        ins_bases.tolist(),
        lengths,
    )

    modified: list[Read] = []
    n_errors_total = 0
    n_bases_total = 0
    for read, (new_seq, new_qual, cigar_raw) in zip(flat_reads, results):
        # Rust returns cigar as list of (op:int, length:int)
        cigar: list[tuple[int, int]] = [
            (int(op), int(ln)) for op, ln in cigar_raw
        ]
        n_bases_total += len(read.sequence)
        n_errors_total += sum(ln for op, ln in cigar if op != 0)
        modified.append(Read(
            name=read.name,
            sequence=new_seq,
            quality=new_qual,
            cigar=cigar,
        ))
    return modified, n_errors_total, n_bases_total


def _combined_rust(
    flat_reads: list[Read],
    profile: ErrorModelProfile,
    rng: torch.Generator,
    calibration: QualityCalibration | None,
    error_rate_scale: float,
) -> tuple[list[Read], int, int]:
    """Call the combined Rust HMM + error assembly path.

    Returns (modified_reads, n_errors_total, n_bases_total).
    """
    initial = profile.initial_logits.float().tolist()
    transition = profile.transition_logits.float().reshape(-1).tolist()
    emission = profile.emission_logits.float().reshape(-1).tolist()
    num_states = profile.num_states
    num_quality_values = int(profile.emission_logits.shape[1])

    cal_type, cal_params = _get_calibration_params(calibration)
    base_seed = int(
        torch.randint(0, 2**62, (1,), generator=rng).item()
    )
    sequences = [r.sequence for r in flat_reads]

    results = _sample_and_apply_errors_batch_rust(
        sequences,
        initial,
        transition,
        emission,
        num_states,
        num_quality_values,
        float(profile.substitution_ratio),
        float(profile.insertion_ratio),
        float(error_rate_scale),
        cal_type,
        cal_params,
        base_seed,
    )

    modified: list[Read] = []
    n_errors_total = 0
    n_bases_total = 0
    for read, (new_seq, new_qual, cigar_raw) in zip(flat_reads, results):
        cigar: list[tuple[int, int]] = [
            (int(op), int(ln)) for op, ln in cigar_raw
        ]
        n_bases_total += len(read.sequence)
        n_errors_total += sum(ln for op, ln in cigar if op != 0)
        modified.append(Read(
            name=read.name,
            sequence=new_seq,
            quality=new_qual,
            cigar=cigar,
        ))
    return modified, n_errors_total, n_bases_total


def apply_error_model(
    read_batch: ReadBatch,
    profile: ErrorModelProfile | None,
    rng: torch.Generator,
    calibration: QualityCalibration | None = None,
    error_rate_scale: float = 1.0,
) -> ReadBatch:
    """Apply HMM-based sequencing error model to reads.

    If *profile* is ``None``, return reads unchanged.

    Args:
        read_batch: Generated reads (single-end or paired-end).
        profile: Error model parameters, or ``None``.
        rng: Random number generator.
        calibration: Quality calibration model.
        error_rate_scale: Multiplier on error probabilities.

    Returns:
        A new ``ReadBatch`` with errors applied.
    """
    if profile is None:
        logger.debug(
            "No error model; skipping error application",
        )
        return read_batch

    logger.info("Applying %s error model...", profile.name)

    flat_reads: list[Read] = []
    if read_batch.is_paired:
        assert read_batch.paired is not None
        for r1, r2 in read_batch.paired:
            flat_reads.append(r1)
            flat_reads.append(r2)
    else:
        assert read_batch.single is not None
        flat_reads = list(read_batch.single)

    modified: list[Read] = []
    n_errors_total = 0
    n_bases_total = 0

    if _HAS_RUST:
        # Combined Rust path: HMM sampling + error assembly in one parallel call
        modified, n_errors_total, n_bases_total = _combined_rust(
            flat_reads, profile, rng, calibration, error_rate_scale,
        )
    else:
        read_lengths = [len(r.sequence) for r in flat_reads]
        logger.debug(
            "Sampling HMM quality scores for %d reads "
            "(max length %d)...",
            len(read_lengths),
            max(read_lengths) if read_lengths else 0,
        )
        all_q_scores = batch_sample_quality_scores(
            profile, read_lengths,
        )
        logger.debug("Quality score sampling complete")
        with progress_task(
            len(flat_reads), "Applying errors",
        ) as step:
            for read, q_scores in zip(flat_reads, all_q_scores):
                new_seq, new_qual, cigar = (
                    apply_errors_to_sequence(
                        read.sequence, q_scores, profile, rng,
                        calibration, error_rate_scale,
                    )
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
            "Error model summary: "
            "%d/%d bases affected (%.2f%%)",
            n_errors_total,
            n_bases_total,
            n_errors_total / n_bases_total * 100,
        )

    batch_key = "paired" if read_batch.is_paired else "single"
    if read_batch.is_paired:
        result = [
            (modified[i], modified[i + 1])
            for i in range(0, len(modified), 2)
        ]
    else:
        result = modified
    return ReadBatch(**{batch_key: result})
