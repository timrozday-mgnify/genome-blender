#!/usr/bin/env python3
"""Reconstruct basepair sequences from sampled minimizer paths and original reads.

For each minimizer path (produced by ``asf_sample.py``), this script:

  1. **Seed** — for each path minimizer, find reads that contain it (via the
     ``.read_minimizers`` files).
  2. **Chain** — align each candidate read to the path using a linear-chaining
     DP in minimizer space (analogous to minimap2 seed-and-chain).
  3. **Score** — rank alignments by anchor count, insert-size log-likelihood,
     and paired-end mate concordance.
  4. **Extract** — pull the inter-minimizer gap sequences from high-scoring reads.
  5. **Choose** — select the final gap sequence per path span using ``--mode``.
  6. **Stitch** — concatenate l-mer sequences and gap sequences into a single
     basepair sequence per path.

Reads spanning each path are found by scanning the ``{prefix}.*.read_minimizers``
files (produced by rust-mdbg ``--dump-read-minimizers``).  Only reads that share
at least one minimizer with a sampled path are kept in memory, making this
approach practical for large datasets.

Usage::

    python scripts/reconstruct_sequences.py paths.jsonl \\
        --prefix rust_mdbg_out \\
        --reads reads_R1.fq.gz --reads reads_R2.fq.gz \\
        --insert-size-json insert_size_estimate.json \\
        --mode best \\
        --output reconstructed.fa
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated, Final, Literal

import pysam
import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# Import helpers from sibling scripts.
sys.path.insert(0, str(Path(__file__).parent))
from read_minimizers import iter_records as _iter_read_minimizer_records  # noqa: E402
from read_minimizers import load_minimizer_table  # noqa: E402

app = typer.Typer(add_completion=False)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROGRESS_COLUMNS: Final = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)

_PAIR_SUFFIX_PATTERNS: Final = (
    re.compile(r"/R?([12])$"),
    re.compile(r"_R([12])$"),
    re.compile(r"\.R?([12])$"),
    re.compile(r"[ \t]([12]):.*$"),
)

_DEFAULT_GAP_FILL: Final = "N"
_DEFAULT_MIN_ANCHORS: Final = 3
_DEFAULT_MAX_READS_PER_MIN: Final = 200
_DEFAULT_MIN_COV_FRACTION: Final = 0.5
_COMP_TABLE: Final = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


# ---------------------------------------------------------------------------
# Enums and data structures
# ---------------------------------------------------------------------------


class Mode(str, Enum):
    """Sequence selection strategy for each path span."""

    best = "best"
    random = "random"
    common = "common"
    consensus = "consensus"


@dataclass
class PathResult:
    """A sampled path through the minimizer-space de Bruijn graph.

    Attributes:
        minimizer_ids: Ordered minimizer hash IDs along the path.
        distances: Minimizer-count distances between consecutive minimizers.
        support: Read-support counts for each extension step.
    """

    minimizer_ids: list[int]
    distances: list[int]
    support: list[int]


@dataclass
class InsertSizeDistribution:
    """Log-normal insert-size distribution in basepair space.

    Attributes:
        mu_log: Log-scale mean (ln units).
        sigma_log: Log-scale standard deviation.
    """

    mu_log: float
    sigma_log: float

    def log_prob(self, insert_bp: float) -> float:
        """Return log P(insert_bp) under this log-normal distribution.

        Args:
            insert_bp: Insert size in basepairs (must be > 0).

        Returns:
            Log probability; ``-inf`` if ``insert_bp <= 0``.
        """
        if insert_bp <= 0.0:
            return -math.inf
        log_x = math.log(insert_bp)
        return (
            -((log_x - self.mu_log) ** 2) / (2.0 * self.sigma_log ** 2)
            - math.log(insert_bp * self.sigma_log * math.sqrt(2.0 * math.pi))
        )


@dataclass
class ScoringWeights:
    """Weights for the composite alignment score.

    Attributes:
        coverage: Weight for anchor-count component.
        insert: Weight for insert-size log-likelihood component.
        mate: Bonus added when the paired mate also aligns concordantly.
    """

    coverage: float = 1.0
    insert: float = 0.5
    mate: float = 2.0


@dataclass
class ReadAlignment:
    """A read aligned to a minimizer path.

    Attributes:
        read_name: Original read name from the FASTQ.
        chain: List of (path_pos, read_min_idx) anchor pairs, sorted by path_pos.
        strand: ``"+"`` if the read aligns in the forward path direction.
        anchor_count: Number of chained anchors.
        insert_size_ll: Log-likelihood of the insert size (0.0 if not computed).
        mate_concordant: True if the paired mate also aligns to the same path.
        score: Composite alignment score (set after scoring).
    """

    read_name: str
    chain: list[tuple[int, int]]
    strand: Literal["+", "-"]
    anchor_count: int
    insert_size_ll: float = 0.0
    mate_concordant: bool = False
    score: float = 0.0


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_COMP_TABLE)[::-1]


def _canonical_pair_name(name: str) -> tuple[str, int | None]:
    """Strip R1/R2 suffixes and return the canonical name and pair number.

    Args:
        name: Read name, possibly with a pairing suffix.

    Returns:
        ``(canonical_name, pair_num)`` where ``pair_num`` is 1 or 2, or
        ``(name, None)`` when no recognisable suffix is found.
    """
    for pat in _PAIR_SUFFIX_PATTERNS:
        m = pat.search(name)
        if m:
            return name[: m.start()], int(m.group(1))
    return name, None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _collect_relevant_reads(
    prefix: Path,
    path_min_set: set[int],
    progress: Progress | None = None,
) -> tuple[dict[str, tuple[list[int], list[int]]], dict[int, list[str]]]:
    """Scan ``.read_minimizers`` files and retain reads overlapping the paths.

    Args:
        prefix: rust-mdbg output prefix.
        path_min_set: Set of minimizer hashes that appear in any sampled path.
        progress: Optional Rich :class:`Progress` for display.

    Returns:
        A tuple ``(relevant_reads, min_to_reads)`` where ``relevant_reads``
        maps ``read_name → (minimizer_ids, bp_positions)`` for every read that
        shares at least one minimizer with a path, and ``min_to_reads`` maps
        each path minimizer hash to the list of matching read names.
    """
    relevant_reads: dict[str, tuple[list[int], list[int]]] = {}
    min_to_reads: dict[int, list[str]] = defaultdict(list)

    task = None
    if progress is not None:
        task = progress.add_task("Scanning .read_minimizers …", total=None)

    n_scanned = 0
    n_kept = 0
    for record in _iter_read_minimizer_records(prefix):
        n_scanned += 1
        path_hits = [mid for mid in record.minimizer_ids if mid in path_min_set]
        if not path_hits:
            continue
        n_kept += 1
        relevant_reads[record.read_id] = (record.minimizer_ids, record.positions)
        for mid in path_hits:
            min_to_reads[mid].append(record.read_id)
        if progress is not None and n_scanned % 100_000 == 0:
            progress.update(
                task,
                description=f"Scanning .read_minimizers … {n_kept:,} / {n_scanned:,}",
            )

    if progress is not None and task is not None:
        progress.update(
            task,
            description=f"Scanned {n_scanned:,} reads, kept {n_kept:,} relevant",
            completed=n_kept,
            total=n_kept,
        )
    log.info("Scanned %d reads; kept %d relevant to paths", n_scanned, n_kept)
    return relevant_reads, min_to_reads


def _load_fastq(
    fastq_paths: Iterable[Path],
    relevant_names: set[str],
    progress: Progress | None = None,
) -> dict[str, tuple[str, str | None]]:
    """Stream FASTQ/FASTA files and return sequences for relevant reads.

    Args:
        fastq_paths: Paths to FASTQ or FASTA input files (R1 and R2).
        relevant_names: Set of read names to retain.
        progress: Optional Rich :class:`Progress` for display.

    Returns:
        Mapping of ``read_name → (sequence, quality_or_None)``.
    """
    seqs: dict[str, tuple[str, str | None]] = {}
    for fq in fastq_paths:
        task = None
        if progress is not None:
            task = progress.add_task(f"Loading {fq.name} …", total=None)
        n = 0
        with pysam.FastxFile(str(fq)) as fh:
            for read in fh:
                if read.name in relevant_names:
                    seqs[read.name] = (read.sequence, read.quality)
                    n += 1
        if progress is not None and task is not None:
            progress.update(
                task,
                description=f"Loaded {n:,} reads from {fq.name}",
                completed=n,
                total=n,
            )
        log.info("Loaded %d sequences from %s", n, fq)
    return seqs


def _build_pair_index(
    read_names: Iterable[str],
) -> tuple[dict[str, str], dict[str, tuple[str | None, str | None]]]:
    """Index paired-end read names for mate lookup.

    Args:
        read_names: All relevant read names.

    Returns:
        A tuple ``(read_to_canonical, canonical_to_pair)`` where
        ``read_to_canonical`` maps each read name to its canonical (suffix-free)
        name, and ``canonical_to_pair`` maps canonical names to
        ``(r1_name_or_None, r2_name_or_None)``.
    """
    read_to_canonical: dict[str, str] = {}
    canonical_to_pair: dict[str, list[str | None]] = {}
    for name in read_names:
        canonical, pair_num = _canonical_pair_name(name)
        read_to_canonical[name] = canonical
        if canonical not in canonical_to_pair:
            canonical_to_pair[canonical] = [None, None]
        if pair_num == 1:
            canonical_to_pair[canonical][0] = name
        elif pair_num == 2:
            canonical_to_pair[canonical][1] = name
    return read_to_canonical, {k: (v[0], v[1]) for k, v in canonical_to_pair.items()}


def _load_insert_size(
    insert_size_json: Path | None,
    insert_size_mean: float | None,
    insert_size_std: float | None,
) -> InsertSizeDistribution | None:
    """Construct an insert-size distribution from CLI options.

    Parses a JSON file from ``asf_sample.py`` or uses explicit mean/std values.

    Args:
        insert_size_json: Path to JSON with ``median_bp`` and ``sigma_bp`` keys.
        insert_size_mean: Mean insert size in bp (if no JSON given).
        insert_size_std: Standard deviation in bp (if no JSON given).

    Returns:
        An :class:`InsertSizeDistribution`, or ``None`` if no parameters given.

    Raises:
        typer.BadParameter: If the JSON keys are missing or values are invalid.
    """
    if insert_size_json is not None:
        data = json.loads(insert_size_json.read_text())
        # asf_sample.py writes either mu_bp / sigma_bp or median_bp with sigma_bp.
        if "mu_bp" in data and "sigma_bp" in data:
            return InsertSizeDistribution(
                mu_log=float(data["mu_bp"]),
                sigma_log=float(data["sigma_bp"]),
            )
        if "median_bp" in data and "sigma_bp" in data:
            median = float(data["median_bp"])
            if median <= 0.0:
                raise typer.BadParameter(
                    f"median_bp must be positive, got {median}", param_hint="--insert-size-json"
                )
            return InsertSizeDistribution(
                mu_log=math.log(median),
                sigma_log=float(data["sigma_bp"]),
            )
        raise typer.BadParameter(
            "JSON must contain 'mu_bp'/'sigma_bp' or 'median_bp'/'sigma_bp'",
            param_hint="--insert-size-json",
        )

    if insert_size_mean is not None and insert_size_std is not None:
        if insert_size_mean <= 0.0:
            raise typer.BadParameter("--insert-size-mean must be positive")
        if insert_size_std <= 0.0:
            raise typer.BadParameter("--insert-size-std must be positive")
        # Convert mean/std of normal to log-normal parameters.
        var = insert_size_std ** 2
        mu2 = insert_size_mean ** 2
        sigma_log = math.sqrt(math.log(1.0 + var / mu2))
        mu_log = math.log(insert_size_mean) - sigma_log ** 2 / 2.0
        return InsertSizeDistribution(mu_log=mu_log, sigma_log=sigma_log)

    return None


# ---------------------------------------------------------------------------
# Seed-and-chain alignment
# ---------------------------------------------------------------------------


def _chain_dp(
    path_min_ids: list[int],
    query_min_ids: list[int],
    gap_tolerance: int,
) -> list[tuple[int, int]]:
    """Run linear-chaining DP on (path_pos, query_pos) anchors.

    Anchors are generated where ``path_min_ids[path_pos] == query_min_ids[query_pos]``.
    The DP finds the longest chain of anchors that is strictly monotone in both
    coordinates and satisfies ``|path_diff − query_diff| ≤ gap_tolerance``.

    Args:
        path_min_ids: Ordered minimizer hashes of the target path.
        query_min_ids: Ordered minimizer hashes of the query read (one strand).
        gap_tolerance: Maximum allowed deviation between path and query offsets.

    Returns:
        Sorted list of ``(path_pos, query_pos)`` anchor pairs in the best chain.
    """
    path_hash_to_pos: dict[int, list[int]] = defaultdict(list)
    for i, h in enumerate(path_min_ids):
        path_hash_to_pos[h].append(i)

    anchors: list[tuple[int, int]] = []
    for qpos, h in enumerate(query_min_ids):
        for ppos in path_hash_to_pos.get(int(h), []):
            anchors.append((ppos, qpos))

    if not anchors:
        return []

    anchors.sort()
    n = len(anchors)
    dp = [1] * n
    prev = [-1] * n

    for i in range(1, n):
        pi, qi = anchors[i]
        best_score = 1
        best_j = -1
        for j in range(i - 1, -1, -1):
            pj, qj = anchors[j]
            if pj >= pi or qj >= qi:
                continue
            if abs((pi - pj) - (qi - qj)) <= gap_tolerance:
                cand = dp[j] + 1
                if cand > best_score:
                    best_score = cand
                    best_j = j
        dp[i] = best_score
        prev[i] = best_j

    best_end = max(range(n), key=lambda x: dp[x])
    chain: list[tuple[int, int]] = []
    idx = best_end
    while idx >= 0:
        chain.append(anchors[idx])
        idx = prev[idx]
    chain.reverse()
    return chain


def align_read_to_path(
    path_min_ids: list[int],
    read_min_ids: list[int],
    gap_tolerance: int = 0,
) -> tuple[list[tuple[int, int]], Literal["+", "-"]]:
    """Align a read to a path in minimizer space, trying both strands.

    The chain coordinates are always reported in terms of the read's original
    (un-reversed) minimizer indices, even for reverse-strand alignments.

    Args:
        path_min_ids: Ordered minimizer hashes of the path.
        read_min_ids: Ordered minimizer hashes of the read (forward strand).
        gap_tolerance: Maximum offset difference allowed for chaining.

    Returns:
        A tuple ``(chain, strand)`` where ``chain`` is a list of
        ``(path_pos, read_min_idx)`` pairs sorted by path_pos, and ``strand``
        is ``"+"`` or ``"-"``.
    """
    n = len(read_min_ids)
    fwd_chain = _chain_dp(path_min_ids, read_min_ids, gap_tolerance)
    rev_chain_rev = _chain_dp(path_min_ids, list(reversed(read_min_ids)), gap_tolerance)
    # Convert reversed-array indices back to original read indices.
    rev_chain = [(p, n - 1 - r) for p, r in rev_chain_rev]

    if len(fwd_chain) >= len(rev_chain_rev):
        return fwd_chain, "+"
    return rev_chain, "-"


def _score_alignment(
    aln: ReadAlignment,
    weights: ScoringWeights,
) -> float:
    """Compute a composite score for a read alignment.

    Args:
        aln: The alignment to score.
        weights: Relative weights for each score component.

    Returns:
        Composite floating-point score (higher is better).
    """
    score = weights.coverage * aln.anchor_count
    score += weights.insert * aln.insert_size_ll
    score += weights.mate * float(aln.mate_concordant)
    return score


def _compute_insert_size_bp(
    r1_chain: list[tuple[int, int]],
    r2_chain: list[tuple[int, int]],
    r1_positions: list[int],
    r2_positions: list[int],
    lmer_len: int,
) -> float | None:
    """Estimate insert size in basepairs for a concordant PE pair on the same path.

    Uses the bp positions of the outermost minimizer anchors from each mate.

    Args:
        r1_chain: Alignment chain for R1.
        r2_chain: Alignment chain for R2.
        r1_positions: bp positions of R1 minimizers in the raw read.
        r2_positions: bp positions of R2 minimizers in the raw read.
        lmer_len: Length of each l-mer.

    Returns:
        Estimated insert size in bp, or ``None`` if it cannot be determined.
    """
    if not r1_chain or not r2_chain:
        return None

    def _read_span_bp(chain: list[tuple[int, int]], positions: list[int]) -> int:
        min_rmi = min(c[1] for c in chain)
        max_rmi = max(c[1] for c in chain)
        return positions[max_rmi] + lmer_len - positions[min_rmi]

    r1_span = _read_span_bp(r1_chain, r1_positions)
    r2_span = _read_span_bp(r2_chain, r2_positions)

    r1_path_start = r1_chain[0][0]
    r1_path_end = r1_chain[-1][0]
    r2_path_start = r2_chain[0][0]
    r2_path_end = r2_chain[-1][0]

    # Path distance between the outermost anchors (in minimizer units).
    outer_start = min(r1_path_start, r2_path_start)
    outer_end = max(r1_path_end, r2_path_end)
    path_span_mers = outer_end - outer_start

    if path_span_mers <= 0:
        return None

    # Estimate bp per minimizer from the reads themselves.
    bp_per_mer_r1 = r1_span / max(len(r1_chain) - 1, 1)
    bp_per_mer_r2 = r2_span / max(len(r2_chain) - 1, 1)
    bp_per_mer = (bp_per_mer_r1 + bp_per_mer_r2) / 2.0

    return path_span_mers * bp_per_mer


# ---------------------------------------------------------------------------
# Basepair gap extraction
# ---------------------------------------------------------------------------


def _gap_seq(
    read_seq: str,
    read_min_positions: list[int],
    chain: list[tuple[int, int]],
    strand: Literal["+", "-"],
    path_gap_start: int,
    path_gap_end: int,
    lmer_len: int,
) -> str | None:
    """Extract the inter-minimizer gap sequence between two consecutive path positions.

    The returned sequence is the bp between the *end* of the l-mer at
    ``path_gap_start`` and the *start* of the l-mer at ``path_gap_end``,
    in path orientation.

    Args:
        read_seq: Full raw (pre-HPC) read sequence.
        read_min_positions: bp position of each minimizer in the read (0-based).
        chain: Sorted list of ``(path_pos, read_min_idx)`` anchor pairs.
        strand: Alignment orientation.
        path_gap_start: Path minimizer index at the gap start (inclusive).
        path_gap_end: Path minimizer index at the gap end (exclusive).
        lmer_len: Length of each l-mer.

    Returns:
        Gap sequence string (may be empty for adjacent minimizers), or ``None``
        if the alignment does not cover both ``path_gap_start`` and ``path_gap_end``.
    """
    chain_dict = {p: r for p, r in chain}
    if path_gap_start not in chain_dict or path_gap_end not in chain_dict:
        return None

    rmi_start = chain_dict[path_gap_start]
    rmi_end = chain_dict[path_gap_end]

    if strand == "+":
        bp_after = read_min_positions[rmi_start] + lmer_len
        bp_before = read_min_positions[rmi_end]
        if bp_after > bp_before:
            return ""
        return read_seq[bp_after:bp_before]

    # "-" strand: rmi_start > rmi_end (path advances → read position decreases).
    # In the original (forward) read the gap_end minimizer is to the left of gap_start.
    bp_after = read_min_positions[rmi_end] + lmer_len
    bp_before = read_min_positions[rmi_start]
    if bp_after > bp_before:
        return ""
    return _reverse_complement(read_seq[bp_after:bp_before])


def _lmer_from_alignment(
    read_seq: str,
    read_min_positions: list[int],
    chain: list[tuple[int, int]],
    strand: Literal["+", "-"],
    path_pos: int,
    lmer_len: int,
) -> str | None:
    """Extract the l-mer sequence at a given path position from a read alignment.

    Args:
        read_seq: Full raw read sequence.
        read_min_positions: bp positions of minimizers in the read.
        chain: Alignment chain.
        strand: Alignment strand.
        path_pos: Path minimizer position to extract.
        lmer_len: Length of the l-mer.

    Returns:
        l-mer string in path orientation, or ``None`` if not covered.
    """
    chain_dict = {p: r for p, r in chain}
    if path_pos not in chain_dict:
        return None
    rmi = chain_dict[path_pos]
    bp = read_min_positions[rmi]
    raw_lmer = read_seq[bp: bp + lmer_len]
    if strand == "-":
        return _reverse_complement(raw_lmer)
    return raw_lmer


# ---------------------------------------------------------------------------
# Sequence selection
# ---------------------------------------------------------------------------


def _choose_gap_seq(
    candidates: list[tuple[str, float]],
    mode: Mode,
    rng: object,
    weight_coverage: bool,
) -> str | None:
    """Choose the gap sequence from a list of (sequence, score) candidates.

    Args:
        candidates: Non-empty list of ``(gap_sequence, alignment_score)`` tuples.
        mode: Selection strategy.
        rng: Random number generator with a ``choices`` method (e.g. ``random.Random``).
        weight_coverage: Whether to weight random selection by alignment score.

    Returns:
        The chosen gap sequence, or ``None`` if ``candidates`` is empty.
    """
    if not candidates:
        return None

    if mode is Mode.best:
        return max(candidates, key=lambda x: x[1])[0]

    if mode is Mode.random:
        seqs = [c[0] for c in candidates]
        if weight_coverage and len(seqs) > 1:
            scores = [max(c[1], 0.0) for c in candidates]
            total = sum(scores)
            weights = [s / total for s in scores] if total > 0.0 else None
        else:
            weights = None
        return rng.choices(seqs, weights=weights, k=1)[0]  # type: ignore[union-attr]

    if mode is Mode.common:
        count: Counter[str] = Counter()
        best_score: dict[str, float] = {}
        for seq, score in candidates:
            count[seq] += 1
            if seq not in best_score or score > best_score[seq]:
                best_score[seq] = score
        max_count = max(count.values())
        most_common = [s for s, c in count.items() if c == max_count]
        return max(most_common, key=lambda s: best_score[s])

    # consensus — requires abpoa or spoa on PATH.
    seqs = [c[0] for c in candidates]
    return _call_consensus_aligner(seqs)


def _call_consensus_aligner(seqs: list[str]) -> str:
    """Compute a multi-sequence consensus using abpoa or spoa.

    Searches PATH for ``abpoa`` first, then ``spoa``.  Raises
    :class:`RuntimeError` if neither is found.

    Args:
        seqs: Input sequences (all should be similar in length and orientation).

    Returns:
        Consensus sequence string.

    Raises:
        RuntimeError: If neither ``abpoa`` nor ``spoa`` is on PATH.
        subprocess.CalledProcessError: If the aligner exits with non-zero status.
    """
    if not seqs:
        return ""
    if len(seqs) == 1:
        return seqs[0]

    aligner = shutil.which("abpoa") or shutil.which("spoa")
    if aligner is None:
        raise RuntimeError(
            "Neither 'abpoa' nor 'spoa' found on PATH.  "
            "Install one of them to use --mode consensus."
        )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".fa", delete=False
    ) as tmp:
        for i, seq in enumerate(seqs):
            tmp.write(f">seq{i}\n{seq}\n")
        tmp_path = tmp.name

    try:
        aligner_name = Path(aligner).name
        if aligner_name == "abpoa":
            cmd = [aligner, "-m", "0", tmp_path]
        else:
            cmd = [aligner, tmp_path]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    finally:
        import os as _os
        _os.unlink(tmp_path)

    # Parse all FASTA sequences from the output; return the last one.
    # abpoa -m 0 emits only the consensus; spoa emits the MSA followed by the consensus.
    all_seqs: list[str] = []
    current: list[str] = []
    for line in result.stdout.splitlines():
        if line.startswith(">"):
            if current:
                all_seqs.append("".join(current))
            current = []
        else:
            current.append(line.strip())
    if current:
        all_seqs.append("".join(current))
    return all_seqs[-1] if all_seqs else ""


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_path(
    path: PathResult,
    relevant_reads: dict[str, tuple[list[int], list[int]]],
    min_to_reads: dict[int, list[str]],
    read_seqs: dict[str, tuple[str, str | None]],
    read_to_canonical: dict[str, str],
    canonical_to_pair: dict[str, tuple[str | None, str | None]],
    minimizer_table: dict[int, str],
    lmer_len: int,
    min_anchors: int,
    max_reads_per_min: int,
    mode: Mode,
    weights: ScoringWeights,
    insert_dist: InsertSizeDistribution | None,
    gap_fill_char: str,
    min_cov_fraction: float,
    weight_coverage: bool,
    rng: object,
) -> tuple[str | None, int, int]:
    """Reconstruct the basepair sequence for one minimizer path.

    Args:
        path: The minimizer path to reconstruct.
        relevant_reads: ``{read_name: (min_ids, positions)}`` for reads overlapping paths.
        min_to_reads: ``{min_hash: [read_name]}`` for path minimizers.
        read_seqs: ``{read_name: (seq, qual)}`` loaded from FASTQ.
        read_to_canonical: ``{read_name: canonical_name}`` for pair lookup.
        canonical_to_pair: ``{canonical_name: (r1_name, r2_name)}``.
        minimizer_table: ``{hash: lmer_string}`` from rust-mdbg.
        lmer_len: Length of each l-mer.
        min_anchors: Minimum chain length to keep an alignment.
        max_reads_per_min: Maximum candidate reads looked up per minimizer.
        mode: Sequence selection strategy.
        weights: Alignment scoring weights.
        insert_dist: Insert-size distribution, or ``None`` if not used.
        gap_fill_char: Character used for uncovered gaps.
        min_cov_fraction: Minimum fraction of path minimizers with read coverage.
        weight_coverage: Weight random gap selection by alignment score.
        rng: Random number generator.

    Returns:
        A tuple ``(sequence_or_None, n_minimizers, n_covered_minimizers)``.
        ``sequence_or_None`` is ``None`` when coverage is below the threshold.
    """
    n_path = len(path.minimizer_ids)
    if n_path == 0:
        return None, 0, 0

    # --- Collect candidate reads (up to max_reads_per_min per minimizer) -------
    candidate_names: set[str] = set()
    for mid in path.minimizer_ids:
        reads_for_min = min_to_reads.get(mid, [])
        candidate_names.update(reads_for_min[:max_reads_per_min])

    # --- Align each candidate read to the path --------------------------------
    alignments: list[ReadAlignment] = []
    for read_name in candidate_names:
        if read_name not in relevant_reads or read_name not in read_seqs:
            continue
        min_ids, _ = relevant_reads[read_name]
        chain, strand = align_read_to_path(path.minimizer_ids, min_ids)
        if len(chain) < min_anchors:
            continue
        alignments.append(
            ReadAlignment(
                read_name=read_name,
                chain=chain,
                strand=strand,
                anchor_count=len(chain),
            )
        )

    # --- Compute insert-size LL and mate concordance --------------------------
    # Build a lookup: path positions covered → alignment index, for mate checking.
    aln_by_name: dict[str, ReadAlignment] = {a.read_name: a for a in alignments}
    for aln in alignments:
        canonical = read_to_canonical.get(aln.read_name)
        if canonical is None:
            continue
        r1_name, r2_name = canonical_to_pair.get(canonical, (None, None))
        mate_name = r2_name if aln.read_name == r1_name else r1_name
        if mate_name is None or mate_name not in aln_by_name:
            continue
        mate_aln = aln_by_name[mate_name]
        aln.mate_concordant = True
        mate_aln.mate_concordant = True

        if insert_dist is not None and aln.read_name == r1_name:
            _, r1_pos = relevant_reads[aln.read_name]
            _, r2_pos = relevant_reads[mate_name]
            insert_bp = _compute_insert_size_bp(
                aln.chain, mate_aln.chain,
                r1_pos, r2_pos,
                lmer_len,
            )
            if insert_bp is not None:
                ll = insert_dist.log_prob(insert_bp)
                aln.insert_size_ll = ll
                mate_aln.insert_size_ll = ll

    for aln in alignments:
        aln.score = _score_alignment(aln, weights)

    # --- Determine dominant strand for minimizer_table fallback ---------------
    n_fwd = sum(1 for a in alignments if a.strand == "+")
    dominant_strand: Literal["+", "-"] = "+" if n_fwd >= len(alignments) - n_fwd else "-"

    # --- Build per-position l-mer and per-gap sequences -----------------------
    # For each position, pick the best covering alignment.
    best_aln_at: dict[int, ReadAlignment] = {}
    for aln in sorted(alignments, key=lambda a: a.score):
        for path_pos, _ in aln.chain:
            best_aln_at[path_pos] = aln  # higher score overwrites lower

    n_covered = len(best_aln_at)
    if n_path > 1 and n_covered / n_path < min_cov_fraction:
        return None, n_path, n_covered

    # Collect l-mers.
    lmers: list[str | None] = []
    for i, mid in enumerate(path.minimizer_ids):
        if i in best_aln_at:
            aln = best_aln_at[i]
            read_seq_str, _ = read_seqs[aln.read_name]
            _, pos = relevant_reads[aln.read_name]
            lmer = _lmer_from_alignment(read_seq_str, pos, aln.chain, aln.strand, i, lmer_len)
        else:
            lmer = minimizer_table.get(mid)
            if lmer is not None and dominant_strand == "-":
                lmer = _reverse_complement(lmer)
        lmers.append(lmer)

    # Collect gap sequences (per mode).
    gaps: list[str | None] = []
    for i in range(n_path - 1):
        gap_start, gap_end = i, i + 1
        # Gather all alignments that cover both positions.
        candidates: list[tuple[str, float]] = []
        for aln in alignments:
            read_seq_str, _ = read_seqs[aln.read_name]
            _, pos = relevant_reads[aln.read_name]
            seq = _gap_seq(read_seq_str, pos, aln.chain, aln.strand, gap_start, gap_end, lmer_len)
            if seq is not None:
                candidates.append((seq, aln.score))
        if candidates:
            chosen = _choose_gap_seq(candidates, mode, rng, weight_coverage=weight_coverage)
        else:
            # No read covers this gap; fill with Ns.
            est_gap_len = max(0, path.distances[i] - 1) if i < len(path.distances) else 0
            chosen = gap_fill_char * est_gap_len
        gaps.append(chosen)

    # --- Stitch ---------------------------------------------------------------
    parts: list[str] = []
    for i in range(n_path):
        lmer_i = lmers[i] or (gap_fill_char * lmer_len)
        parts.append(lmer_i)
        if i < n_path - 1:
            parts.append(gaps[i] or "")

    return "".join(parts), n_path, n_covered


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    paths_file: Annotated[Path, typer.Argument(
        help="JSONL file of sampled minimizer paths (from asf_sample.py).",
        exists=True, readable=True,
    )],
    prefix: Annotated[Path, typer.Option(
        "--prefix",
        help="rust-mdbg output prefix (used to locate .read_minimizers files "
             "and minimizer_table).",
    )],
    reads: Annotated[list[Path], typer.Option(
        "--reads",
        help="FASTQ/FASTA file with original reads.  Repeat twice for paired-end "
             "(R1 then R2).",
        exists=True, readable=True,
    )],
    output: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Output FASTA file.",
    )] = Path("reconstructed.fa"),
    mode: Annotated[Mode, typer.Option(
        "--mode",
        help="Sequence selection strategy per gap span.",
    )] = Mode.best,
    insert_size_json: Annotated[Path | None, typer.Option(
        "--insert-size-json",
        help="JSON file from asf_sample.py with insert-size estimate "
             "(keys: 'mu_bp'/'sigma_bp' or 'median_bp'/'sigma_bp').",
    )] = None,
    insert_size_mean: Annotated[float | None, typer.Option(
        "--insert-size-mean",
        help="Mean insert size in bp (alternative to --insert-size-json).",
    )] = None,
    insert_size_std: Annotated[float | None, typer.Option(
        "--insert-size-std",
        help="Standard deviation of insert size in bp "
             "(required when --insert-size-mean is given).",
    )] = None,
    min_anchors: Annotated[int, typer.Option(
        "--min-anchors",
        help="Minimum chained minimizer hits required to keep a read alignment.",
        min=1,
    )] = _DEFAULT_MIN_ANCHORS,
    max_reads_per_min: Annotated[int, typer.Option(
        "--max-reads-per-minimizer",
        help="Maximum candidate reads examined per path minimizer.",
        min=1,
    )] = _DEFAULT_MAX_READS_PER_MIN,
    min_cov_fraction: Annotated[float, typer.Option(
        "--min-coverage-fraction",
        help="Omit paths where fewer than this fraction of minimizer "
             "positions have read coverage.",
        min=0.0, max=1.0,
    )] = _DEFAULT_MIN_COV_FRACTION,
    weight_coverage: Annotated[bool, typer.Option(
        "--weight-coverage/--no-weight-coverage",
        help="Weight random selection by anchor-count score component.",
    )] = True,
    weight_insert: Annotated[bool, typer.Option(
        "--weight-insert/--no-weight-insert",
        help="Include insert-size log-likelihood in alignment scoring.",
    )] = True,
    weight_mate: Annotated[bool, typer.Option(
        "--weight-mate/--no-weight-mate",
        help="Add mate-concordance bonus to alignment scoring.",
    )] = True,
    gap_fill_char: Annotated[str, typer.Option(
        "--gap-fill-char",
        help="Character used to fill uncovered path gaps.",
        min_length=1, max_length=1,
    )] = _DEFAULT_GAP_FILL,
    seed: Annotated[int, typer.Option(
        "--seed",
        help="Random seed for reproducibility.",
    )] = 42,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable debug logging.",
    )] = False,
) -> None:
    """Reconstruct basepair sequences from sampled minimizer paths and reads."""
    import random

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    rng = random.Random(seed)

    # --- Validate insert-size options ----------------------------------------
    if insert_size_mean is not None and insert_size_std is None:
        raise typer.BadParameter(
            "--insert-size-std is required when --insert-size-mean is given"
        )
    insert_dist = _load_insert_size(insert_size_json, insert_size_mean, insert_size_std)

    # Adjust weights based on flags.
    scoring_weights = ScoringWeights(
        coverage=1.0,
        insert=0.5 if (weight_insert and insert_dist is not None) else 0.0,
        mate=2.0 if weight_mate else 0.0,
    )

    with Progress(*_PROGRESS_COLUMNS) as progress:
        # --- Load minimizer table ------------------------------------------------
        table_path = Path(f"{prefix}.minimizer_table")
        if not table_path.exists():
            typer.echo(
                f"Error: minimizer_table not found: {table_path}",
                err=True,
            )
            raise typer.Exit(1)
        load_task = progress.add_task("Loading minimizer_table …", total=None)
        minimizer_table = load_minimizer_table(table_path)
        # Detect l-mer length from the first entry.
        lmer_len = len(next(iter(minimizer_table.values()))) if minimizer_table else 17
        progress.update(
            load_task,
            description=f"Loaded {len(minimizer_table):,} minimizers (l={lmer_len})",
            completed=1, total=1,
        )
        log.info("Loaded minimizer_table: %d entries, lmer_len=%d", len(minimizer_table), lmer_len)

        # --- Load paths ----------------------------------------------------------
        paths_task = progress.add_task("Loading paths …", total=None)
        paths: list[PathResult] = []
        with open(paths_file) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                paths.append(PathResult(
                    minimizer_ids=obj["minimizer_ids"],
                    distances=obj.get("distances", []),
                    support=obj.get("support", []),
                ))
        progress.update(
            paths_task,
            description=f"Loaded {len(paths):,} paths",
            completed=len(paths), total=len(paths),
        )
        log.info("Loaded %d paths from %s", len(paths), paths_file)

        # --- Collect path minimizer hashes ---------------------------------------
        path_min_set: set[int] = set()
        for p in paths:
            path_min_set.update(p.minimizer_ids)
        log.info("%d unique minimizer hashes across all paths", len(path_min_set))

        # --- Scan .read_minimizers files -----------------------------------------
        relevant_reads, min_to_reads = _collect_relevant_reads(
            prefix, path_min_set, progress=progress,
        )

        # --- Build pair name index -----------------------------------------------
        pair_task = progress.add_task("Building pair index …", total=None)
        read_to_canonical, canonical_to_pair = _build_pair_index(relevant_reads.keys())
        n_pairs = sum(
            1 for v in canonical_to_pair.values() if v[0] is not None and v[1] is not None
        )
        progress.update(
            pair_task,
            description=f"Indexed {len(relevant_reads):,} reads, {n_pairs:,} complete pairs",
            completed=1, total=1,
        )

        # --- Load FASTQ sequences ------------------------------------------------
        read_seqs = _load_fastq(reads, set(relevant_reads.keys()), progress=progress)
        n_missing = len(relevant_reads) - len(read_seqs)
        if n_missing > 0:
            log.warning(
                "%d reads in .read_minimizers have no matching FASTQ entry", n_missing
            )

        # --- Reconstruct paths ---------------------------------------------------
        recon_task = progress.add_task("Reconstructing paths …", total=len(paths))
        n_written = 0
        n_skipped = 0

        with open(output, "w") as out_fh:
            for path_idx, path in enumerate(paths):
                seq, n_mers, n_covered = _reconstruct_path(
                    path=path,
                    relevant_reads=relevant_reads,
                    min_to_reads=min_to_reads,
                    read_seqs=read_seqs,
                    read_to_canonical=read_to_canonical,
                    canonical_to_pair=canonical_to_pair,
                    minimizer_table=minimizer_table,
                    lmer_len=lmer_len,
                    min_anchors=min_anchors,
                    max_reads_per_min=max_reads_per_min,
                    mode=mode,
                    weights=scoring_weights,
                    insert_dist=insert_dist,
                    gap_fill_char=gap_fill_char,
                    min_cov_fraction=min_cov_fraction,
                    weight_coverage=weight_coverage,
                    rng=rng,
                )
                progress.advance(recon_task)

                if seq is None:
                    n_skipped += 1
                    log.debug(
                        "Path %d skipped: %d/%d minimizers covered",
                        path_idx, n_covered, n_mers,
                    )
                    continue

                header = (
                    f">path_{path_idx} "
                    f"n_minimizers={n_mers} "
                    f"covered_minimizers={n_covered} "
                    f"seq_len={len(seq)}"
                )
                out_fh.write(f"{header}\n{seq}\n")
                n_written += 1

        progress.update(
            recon_task,
            description=(
                f"Reconstructed {n_written:,} paths "
                f"({n_skipped:,} skipped for low coverage)"
            ),
        )

    typer.echo(
        f"Wrote {n_written} sequences to {output} "
        f"({n_skipped} paths skipped for coverage < {min_cov_fraction:.0%})",
        err=True,
    )


if __name__ == "__main__":
    app()
