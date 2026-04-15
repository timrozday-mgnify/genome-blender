#!/usr/bin/env python3
"""Reconstruct basepair sequences from sampled minimizer paths and original reads.

For each minimizer path (produced by ``asf_sample.py``), this script:

  1. **Seed** — for each path minimizer, query the minimizer-index LMDB to find
     candidate reads.
  2. **Chain** — align each candidate read to the path using a linear-chaining
     DP in minimizer space.
  3. **Score** — rank alignments by anchor count, insert-size log-likelihood
     (in minimizer space, Poisson-corrected), and paired-end mate concordance.
  4. **Extract** — pull the basepair span between consecutive path positions
     from high-scoring reads using linear interpolation (no stored positions
     needed).
  5. **Choose** — select the final span per path gap using ``--mode``.
  6. **Stitch** — concatenate per-gap spans into a single basepair sequence per
     path.

Candidate reads are found via the ``{prefix}.minimizer_index*.lmdb`` shards.
Read minimizer arrays are fetched from ``{prefix}.index*.lmdb`` shards.
Paired-end reads are matched by arithmetic: rust-mdbg assigns IDs with stride 2;
R1 read at FASTQ position i has ``read_id = 2i − 1`` (odd) and its R2 mate has
``read_id = 2i`` (even).

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
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
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

# Import LMDB helpers from sibling script asf_sample.py.
sys.path.insert(0, str(Path(__file__).parent))
from asf_sample import (  # noqa: E402
    _LRUCache,
    _get_read_cached,
    _open_minimizer_lmdb,
    _open_reads_lmdb,
    _read_ids_for_minimizer_multi,
    _reads_shard_ranges,
)

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

_DEFAULT_GAP_FILL: Final = "N"
_DEFAULT_MIN_ANCHORS: Final = 3
_DEFAULT_TERMINAL_MIN_ANCHORS: Final = 1
_DEFAULT_MAX_READS_PER_MIN: Final = 200
_DEFAULT_MIN_COV_FRACTION: Final = 0.5
_DEFAULT_DENSITY: Final = 0.01
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
    """Log-normal insert-size distribution in minimizer space.

    Attributes:
        mu_log: Log-scale mean (ln units, minimizer-count space).
        sigma_log: Log-scale standard deviation (minimizer-count space).
    """

    mu_log: float
    sigma_log: float


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
        read_id: Integer read ID (1-based, stride-2 paired assignment from rust-mdbg).
        chain: List of ``(path_pos, read_min_idx)`` anchor pairs, sorted by path_pos.
        strand: ``"+"`` if the read aligns in the forward path direction.
        anchor_count: Number of chained anchors.
        insert_size_ll: Log-likelihood of the insert size (0.0 if not computed).
        mate_concordant: True if the paired mate also aligns to the same path.
        score: Composite alignment score (set after scoring).
    """

    read_id: int
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


def _mate_id(read_id: int) -> int:
    """Return the paired-end mate ID for a stride-2 sequential read ID.

    rust-mdbg assigns IDs with stride 2 in paired two-file mode:
    R1 read at 1-based FASTQ position i has ``read_id = 2i − 1`` (odd).
    R2 read at 1-based FASTQ position i has ``read_id = 2i`` (even).

    Args:
        read_id: 1-based integer read ID.

    Returns:
        ID of the paired mate.
    """
    return read_id + 1 if read_id % 2 == 1 else read_id - 1


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _collect_candidate_read_ids(
    mi_txns_dbs: list[tuple],
    path_hashes: list[int],
    max_per_min: int,
    read_id_width: int,
    cache: _LRUCache,
    progress: Progress | None = None,
) -> dict[int, list[int]]:
    """Query the minimizer-index LMDB for all path minimizer hashes.

    Populates *cache* so that per-path reconstruction can reuse results
    without re-querying LMDB.

    Args:
        mi_txns_dbs: Sharded minimizer-index ``(txn, db)`` pairs.
        path_hashes: Unique minimizer hashes to query.
        max_per_min: Maximum read IDs to return per minimizer.
        read_id_width: Byte width of read-ID values (4 or 8).
        cache: LRU cache to populate with hash → read-ID-list results.
        progress: Optional Rich progress display.

    Returns:
        Mapping of ``minimizer_hash → [read_id, ...]``.
    """
    task = None
    if progress is not None:
        task = progress.add_task(
            "Querying minimizer index …", total=len(path_hashes),
        )
    result: dict[int, list[int]] = {}
    for h in path_hashes:
        ids = _read_ids_for_minimizer_multi(mi_txns_dbs, h, max_per_min, read_id_width)
        cache.put(h, ids)
        result[h] = ids
        if progress is not None and task is not None:
            progress.advance(task)
    if progress is not None and task is not None:
        n_with_reads = sum(1 for v in result.values() if v)
        progress.update(
            task,
            description=f"Queried {len(path_hashes):,} minimizers, "
                        f"{n_with_reads:,} with reads",
        )
    return result


def _load_read_sequences(
    read_paths: list[Path],
    candidate_ids: set[int],
    progress: Progress | None = None,
) -> dict[int, str]:
    """Stream FASTQ files and return sequences keyed by integer read ID.

    rust-mdbg assigns IDs with stride 2 in paired two-file mode:
    the first file (R1): read at 1-based position i → ``read_id = 2i − 1``.
    the second file (R2): read at 1-based position i → ``read_id = 2i``.

    Only reads whose IDs (or their mates) appear in *candidate_ids* are loaded.

    Args:
        read_paths: FASTQ/FASTA files; first is R1, second (if given) is R2.
        candidate_ids: Integer read IDs required for reconstruction.
        progress: Optional Rich progress display.

    Returns:
        Mapping of ``read_id → sequence_string`` for the retained reads.
    """
    needed_ids = candidate_ids | {_mate_id(r) for r in candidate_ids}
    seqs: dict[int, str] = {}
    for file_idx, fq in enumerate(read_paths):
        task = None
        if progress is not None:
            task = progress.add_task(f"Loading {fq.name} …", total=None)
        n = 0
        with pysam.FastxFile(str(fq)) as fh:
            for pos, read in enumerate(fh, start=1):
                # First file (idx 0): odd IDs = 2*pos - 1.
                # Second file (idx 1): even IDs = 2*pos.
                rid = 2 * pos - 1 + file_idx
                if rid in needed_ids and read.sequence:
                    seqs[rid] = read.sequence
                    n += 1
                if progress is not None and task is not None:
                    progress.advance(task)
        if progress is not None and task is not None:
            progress.update(
                task,
                description=f"Loaded {n:,} reads from {fq.name}",
                completed=n,
                total=n,
            )
        log.info("Loaded %d sequences from %s", n, fq)
    return seqs


# ---------------------------------------------------------------------------
# Insert-size model (minimizer space)
# ---------------------------------------------------------------------------


def _load_insert_size(
    insert_size_json: Path | None,
    insert_size_mean: float | None,
    insert_size_std: float | None,
) -> InsertSizeDistribution | None:
    """Construct an insert-size distribution in minimizer space from CLI options.

    When a JSON file is given, the top-level ``mu_log``/``sigma_log`` keys are
    used directly.  These are the minimizer-space log-normal parameters produced
    by ``asf_sample.py``.  The nested ``bp_space`` dict (if present) is ignored
    because the bp-space sigma is typically near zero and unusable.

    Args:
        insert_size_json: Path to an ``asf_sample.py`` insert-size JSON file.
        insert_size_mean: Mean insert size in minimizer units (alternative to JSON).
        insert_size_std: Standard deviation in minimizer units (with mean).

    Returns:
        An :class:`InsertSizeDistribution`, or ``None`` if no parameters given.

    Raises:
        typer.BadParameter: If required JSON keys are missing or values are invalid.
    """
    if insert_size_json is not None:
        data = json.loads(insert_size_json.read_text())
        mu_log = data.get("mu_log")
        sigma_log = data.get("sigma_log")
        if mu_log is None or sigma_log is None:
            raise typer.BadParameter(
                "JSON must contain top-level 'mu_log' and 'sigma_log' keys "
                "(minimizer-space log-normal parameters from asf_sample.py)",
                param_hint="--insert-size-json",
            )
        return InsertSizeDistribution(mu_log=float(mu_log), sigma_log=float(sigma_log))

    if insert_size_mean is not None and insert_size_std is not None:
        if insert_size_mean <= 0.0:
            raise typer.BadParameter("--insert-size-mean must be positive")
        if insert_size_std <= 0.0:
            raise typer.BadParameter("--insert-size-std must be positive")
        var = insert_size_std ** 2
        mu2 = insert_size_mean ** 2
        sigma_log = math.sqrt(math.log(1.0 + var / mu2))
        mu_log = math.log(insert_size_mean) - sigma_log ** 2 / 2.0
        return InsertSizeDistribution(mu_log=mu_log, sigma_log=sigma_log)

    return None


def _insert_log_prob_mers(n_mers: int, dist: InsertSizeDistribution) -> float:
    """Return log P(n_mers) under a Poisson-corrected log-normal distribution.

    The Poisson uncertainty in minimizer counting broadens the effective sigma:

    .. math::
        \\sigma_{\\text{eff}}^2 = \\sigma_{\\log}^2 + 1 / n_{\\text{mers}}

    Without this correction, a small ``sigma_log`` produces a near-delta-function
    likelihood that rejects most reads.

    Args:
        n_mers: Observed minimizer-space insert size (outermost-anchor span).
        dist: Log-normal parameters for the minimizer-space insert distribution.

    Returns:
        Log probability; ``-inf`` if *n_mers* ≤ 0.
    """
    x = float(n_mers)
    if x <= 0.0:
        return -math.inf
    sigma_eff = math.sqrt(dist.sigma_log ** 2 + 1.0 / n_mers)
    z = (math.log(x) - dist.mu_log) / sigma_eff
    return -0.5 * z * z - math.log(sigma_eff * x) - 0.5 * math.log(2.0 * math.pi)


def _compute_insert_size_mers(
    r1_chain: list[tuple[int, int]],
    r2_chain: list[tuple[int, int]],
) -> int | None:
    """Return the minimizer-space insert size as the outermost-anchor span.

    Args:
        r1_chain: Alignment chain for the R1 read.
        r2_chain: Alignment chain for the R2 read.

    Returns:
        Number of path minimizers spanning from the outermost R1 anchor to the
        outermost R2 anchor, or ``None`` if either chain is empty.
    """
    if not r1_chain or not r2_chain:
        return None
    all_path_pos = [p for p, _ in r1_chain] + [p for p, _ in r2_chain]
    span = max(all_path_pos) - min(all_path_pos)
    return span if span > 0 else None


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

    Chain coordinates are always reported in terms of the read's original
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


# ---------------------------------------------------------------------------
# Basepair span extraction
# ---------------------------------------------------------------------------


def _extract_read_span(
    read_seq: str,
    read_min_count: int,
    chain: list[tuple[int, int]],
    strand: Literal["+", "-"],
    *,
    extend_to_read_start: bool = False,
    extend_to_read_end: bool = False,
) -> str:
    """Extract the basepair span covered by chain anchors via linear interpolation.

    Given a read of length L with m total minimizers, minimizer at index k is
    estimated to start at bp position ``k * L / m``.  The extracted span runs
    from the first to the last chained minimizer (inclusive), covering the gap
    and both flanking l-mer regions.

    When *extend_to_read_start* is ``True``, the left boundary is clamped to 0
    (the physical start of the read) regardless of the first anchor position.
    When *extend_to_read_end* is ``True``, the right boundary is clamped to L
    (the physical end of the read).  These flags are used for partial terminal
    reads that only anchor at one side of the first or last path gap.

    Args:
        read_seq: Full read sequence (forward strand).
        read_min_count: Total number of minimizers in the read (from LMDB).
        chain: ``(path_pos, read_min_idx)`` anchor pairs; must be non-empty.
        strand: ``"+"`` for forward; ``"-"`` for reverse (span is reverse-complemented).
        extend_to_read_start: If ``True``, force the left bp boundary to 0.
        extend_to_read_end: If ``True``, force the right bp boundary to ``len(read_seq)``.

    Returns:
        Extracted sequence in path orientation; empty string if span is zero length.
    """
    first_rmi = min(c[1] for c in chain)
    last_rmi = max(c[1] for c in chain)
    L = len(read_seq)
    m = max(read_min_count, 1)
    bp_start = 0 if extend_to_read_start else int(first_rmi * L / m)
    bp_end = L if extend_to_read_end else min(int((last_rmi + 1) * L / m), L)
    if bp_start >= bp_end:
        return ""
    span = read_seq[bp_start:bp_end]
    return _reverse_complement(span) if strand == "-" else span


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
        The chosen gap sequence, or ``None`` if *candidates* is empty.
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

    Searches PATH for ``abpoa`` first, then ``spoa``.

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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as tmp:
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

    # Parse all FASTA sequences from stdout; return the last (consensus).
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
# Terminal-gap alignment helpers
# ---------------------------------------------------------------------------


def _align_terminal_reads(
    terminal_positions: set[int],
    candidate_ids: set[int],
    path_min_ids: list[int],
    read_seqs: dict[int, str],
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    read_id_width: int,
    read_cache: _LRUCache,
    min_anchors: int,
    terminal_min_anchors: int,
) -> list[ReadAlignment]:
    """Return weak alignments whose chains touch a terminal path position.

    Reads that already produced a chain of length ≥ *min_anchors* are in the
    main alignment list and are *not* returned here.  This function returns
    only reads with chain length in ``[terminal_min_anchors, min_anchors)``
    that have at least one anchor in *terminal_positions*, so they can be used
    to fill the first or last path gap when no full-spanning read is available.

    Args:
        terminal_positions: Path positions to check for anchor presence
            (typically ``{0}`` for the first gap or ``{n-1}`` for the last).
        candidate_ids: Read IDs to consider.
        path_min_ids: Ordered minimizer hashes of the path.
        read_seqs: ``{read_id: sequence}`` loaded from FASTQ.
        reads_txns_dbs: Sharded reads-index ``(txn, db)`` pairs.
        shard_ranges: Per-shard ``(lo, hi)`` key ranges.
        read_id_width: Byte width of read-ID keys (4 or 8).
        read_cache: LRU cache for read_id → minimizer-hash array.
        min_anchors: Main anchor threshold; reads meeting this are excluded.
        terminal_min_anchors: Minimum chain length to accept here.

    Returns:
        List of :class:`ReadAlignment` objects for qualifying reads.
    """
    results: list[ReadAlignment] = []
    for read_id in candidate_ids:
        if read_id not in read_seqs:
            continue
        min_ids_arr = _get_read_cached(
            reads_txns_dbs, shard_ranges, read_id, read_cache, read_id_width,
        )
        if min_ids_arr is None or len(min_ids_arr) == 0:
            continue
        min_ids: list[int] = [int(x) for x in min_ids_arr]
        chain, strand = align_read_to_path(path_min_ids, min_ids)
        n_anchors = len(chain)
        if n_anchors < terminal_min_anchors or n_anchors >= min_anchors:
            continue
        chain_positions = {p for p, _ in chain}
        if not chain_positions & terminal_positions:
            continue
        results.append(ReadAlignment(
            read_id=read_id,
            chain=chain,
            strand=strand,
            anchor_count=n_anchors,
        ))
    return results


# ---------------------------------------------------------------------------
# Path reconstruction
# ---------------------------------------------------------------------------


def _reconstruct_path(
    path: PathResult,
    mi_txns_dbs: list[tuple],
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    read_id_width: int,
    read_seqs: dict[int, str],
    read_cache: _LRUCache,
    minimizer_cache: _LRUCache,
    density: float,
    min_anchors: int,
    terminal_min_anchors: int,
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
        mi_txns_dbs: Sharded minimizer-index ``(txn, db)`` pairs.
        reads_txns_dbs: Sharded reads-index ``(txn, db)`` pairs.
        shard_ranges: Per-shard ``(lo, hi)`` key ranges for the reads index.
        read_id_width: Byte width of read-ID keys (4 or 8).
        read_seqs: ``{read_id: sequence}`` loaded from FASTQ.
        read_cache: LRU cache for read_id → minimizer-hash array (from LMDB).
        minimizer_cache: LRU cache for minimizer_hash → read-ID list.
        density: Minimizer density in minimizers per basepair (used for gap-fill size).
        min_anchors: Minimum chain length to keep a read alignment globally.
        terminal_min_anchors: Minimum chain length for reads used only at terminal
            path gaps (first and last).  Ignored when ≥ *min_anchors*.
        max_reads_per_min: Maximum candidate reads examined per path minimizer.
        mode: Sequence selection strategy for each gap span.
        weights: Alignment scoring weights.
        insert_dist: Minimizer-space insert-size distribution, or ``None``.
        gap_fill_char: Character for uncovered path spans.
        min_cov_fraction: Minimum fraction of path positions needing read coverage.
        weight_coverage: Weight random span selection by alignment score.
        rng: Random number generator.

    Returns:
        A tuple ``(sequence_or_None, n_minimizers, n_covered_positions)``.
        ``sequence_or_None`` is ``None`` when coverage is below *min_cov_fraction*.
    """
    n_path = len(path.minimizer_ids)
    if n_path == 0:
        return None, 0, 0

    # --- Collect candidate reads (cached minimizer LMDB lookups) ----------------
    candidate_ids: set[int] = set()
    for mid in path.minimizer_ids:
        cached = minimizer_cache.get(mid)
        if cached is not None:
            ids: list[int] = cached  # type: ignore[assignment]
        else:
            ids = _read_ids_for_minimizer_multi(mi_txns_dbs, mid, max_reads_per_min, read_id_width)
            minimizer_cache.put(mid, ids)
        candidate_ids.update(ids[:max_reads_per_min])

    # --- Align each candidate read to the path ----------------------------------
    alignments: list[ReadAlignment] = []
    for read_id in candidate_ids:
        if read_id not in read_seqs:
            continue
        min_ids_arr = _get_read_cached(
            reads_txns_dbs, shard_ranges, read_id, read_cache, read_id_width,
        )
        if min_ids_arr is None or len(min_ids_arr) == 0:
            continue
        min_ids: list[int] = [int(x) for x in min_ids_arr]
        chain, strand = align_read_to_path(path.minimizer_ids, min_ids)
        if len(chain) < min_anchors:
            continue
        alignments.append(ReadAlignment(
            read_id=read_id,
            chain=chain,
            strand=strand,
            anchor_count=len(chain),
        ))

    # --- Collect weak alignments for terminal gaps ------------------------------
    # Reads with chains shorter than min_anchors are normally discarded, but for
    # the first and last path gap they can still provide basepair sequence if
    # they anchor at the terminal path position.
    if n_path >= 2 and terminal_min_anchors < min_anchors:
        terminal_alns = _align_terminal_reads(
            terminal_positions={0, n_path - 1},
            candidate_ids=candidate_ids,
            path_min_ids=path.minimizer_ids,
            read_seqs=read_seqs,
            reads_txns_dbs=reads_txns_dbs,
            shard_ranges=shard_ranges,
            read_id_width=read_id_width,
            read_cache=read_cache,
            min_anchors=min_anchors,
            terminal_min_anchors=terminal_min_anchors,
        )
        alignments.extend(terminal_alns)

    # --- Compute insert-size LL and mate concordance ----------------------------
    aln_by_id: dict[int, ReadAlignment] = {a.read_id: a for a in alignments}
    for aln in alignments:
        mate = aln_by_id.get(_mate_id(aln.read_id))
        if mate is None:
            continue
        aln.mate_concordant = True
        mate.mate_concordant = True

        # Compute insert-size LL only once per pair (when processing R1).
        if insert_dist is not None and aln.read_id % 2 == 1:
            n_mers = _compute_insert_size_mers(aln.chain, mate.chain)
            if n_mers is not None:
                ll = _insert_log_prob_mers(n_mers, insert_dist)
                aln.insert_size_ll = ll
                mate.insert_size_ll = ll

    for aln in alignments:
        aln.score = _score_alignment(aln, weights)

    # --- Determine coverage: best alignment per path position -------------------
    best_aln_at: dict[int, ReadAlignment] = {}
    for aln in sorted(alignments, key=lambda a: a.score):
        for path_pos, _ in aln.chain:
            best_aln_at[path_pos] = aln  # higher score overwrites lower

    n_covered = len(best_aln_at)
    if n_path > 1 and n_covered / n_path < min_cov_fraction:
        return None, n_path, n_covered

    # --- Extract per-gap spans and stitch --------------------------------------
    # For each consecutive pair (i, i+1), find alignments with anchors at both
    # positions and extract the bp span using linear interpolation.
    # For the first and last gap, partial reads anchoring at only ONE side are
    # also accepted; the missing boundary is the read's physical start or end.
    # Adjacent spans share ~1/density bp at their junction (acceptable overlap).
    parts: list[str] = []
    for i in range(n_path - 1):
        is_first_gap = i == 0
        is_last_gap = i == n_path - 2
        is_terminal = is_first_gap or is_last_gap

        candidates: list[tuple[str, float]] = []
        for aln in alignments:
            chain_dict = {p: r for p, r in aln.chain}
            has_i = i in chain_dict
            has_i1 = (i + 1) in chain_dict

            if not has_i and not has_i1:
                continue
            if not is_terminal and (not has_i or not has_i1):
                continue

            read_seq = read_seqs.get(aln.read_id)
            if read_seq is None:
                continue
            min_ids_arr = _get_read_cached(
                reads_txns_dbs, shard_ranges, aln.read_id, read_cache, read_id_width,
            )
            if min_ids_arr is None:
                continue

            # Determine sub-chain and whether to extend to the read boundary.
            if has_i and has_i1:
                sub_chain = [(i, chain_dict[i]), (i + 1, chain_dict[i + 1])]
                extend_start = False
                extend_end = False
            elif is_first_gap and not has_i:
                # Read starts inside the first gap; extend left to read start.
                sub_chain = [(i + 1, chain_dict[i + 1])]
                extend_start = True
                extend_end = False
            elif is_first_gap:
                # has_i but not has_i1: read ends before path pos 1.
                sub_chain = [(i, chain_dict[i])]
                extend_start = False
                extend_end = True
            elif is_last_gap and not has_i1:
                # Read ends inside the last gap; extend right to read end.
                sub_chain = [(i, chain_dict[i])]
                extend_start = False
                extend_end = True
            else:
                # is_last_gap and has_i1 but not has_i: read starts after n-2.
                sub_chain = [(i + 1, chain_dict[i + 1])]
                extend_start = True
                extend_end = False

            span = _extract_read_span(
                read_seq, len(min_ids_arr), sub_chain, aln.strand,
                extend_to_read_start=extend_start,
                extend_to_read_end=extend_end,
            )
            if span:
                candidates.append((span, aln.score))

        if candidates:
            chosen = _choose_gap_seq(candidates, mode, rng, weight_coverage=weight_coverage)
            parts.append(chosen or "")
        else:
            # No read covers this gap; fill with estimated basepair count.
            dist_mers = path.distances[i] if i < len(path.distances) else 1
            est_bp = max(1, int(dist_mers / density))
            parts.append(gap_fill_char * est_bp)

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
        help="rust-mdbg output prefix (used to locate LMDB index files).",
    )],
    reads: Annotated[list[Path], typer.Option(
        "--reads",
        help="FASTQ/FASTA file with original reads.  Repeat twice for "
             "paired-end (R1 then R2).",
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
        help="JSON file from asf_sample.py with minimizer-space insert-size "
             "estimate (top-level keys: 'mu_log' and 'sigma_log').",
    )] = None,
    insert_size_mean: Annotated[float | None, typer.Option(
        "--insert-size-mean",
        help="Mean insert size in minimizer units (alternative to --insert-size-json).",
    )] = None,
    insert_size_std: Annotated[float | None, typer.Option(
        "--insert-size-std",
        help="Standard deviation of insert size in minimizer units "
             "(required when --insert-size-mean is given).",
    )] = None,
    min_anchors: Annotated[int, typer.Option(
        "--min-anchors",
        help="Minimum chained minimizer hits required to keep a read alignment.",
        min=1,
    )] = _DEFAULT_MIN_ANCHORS,
    terminal_min_anchors: Annotated[int, typer.Option(
        "--terminal-min-anchors",
        help="Minimum chain length for partial reads used only at the first and last "
             "path gap.  Can be lower than --min-anchors to recover reads that only "
             "partially overlap a path end.",
        min=1,
    )] = _DEFAULT_TERMINAL_MIN_ANCHORS,
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
        help="Character used to fill uncovered path gaps (must be exactly one character).",
    )] = _DEFAULT_GAP_FILL,
    density: Annotated[float, typer.Option(
        "--density",
        help="Minimizer density in minimizers per basepair, used for gap-fill "
             "length estimation when no read covers a span.",
        min=1e-6,
    )] = _DEFAULT_DENSITY,
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

    # --- Validate gap_fill_char -------------------------------------------------
    if len(gap_fill_char) != 1:
        raise typer.BadParameter(
            "--gap-fill-char must be exactly one character",
            param_hint="--gap-fill-char",
        )

    # --- Validate insert-size options ------------------------------------------
    if insert_size_mean is not None and insert_size_std is None:
        raise typer.BadParameter(
            "--insert-size-std is required when --insert-size-mean is given"
        )
    insert_dist = _load_insert_size(insert_size_json, insert_size_mean, insert_size_std)

    scoring_weights = ScoringWeights(
        coverage=1.0,
        insert=0.5 if (weight_insert and insert_dist is not None) else 0.0,
        mate=2.0 if weight_mate else 0.0,
    )

    with Progress(*_PROGRESS_COLUMNS) as progress:
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

        # --- Collect unique path minimizer hashes --------------------------------
        path_min_set: set[int] = set()
        for p in paths:
            path_min_set.update(p.minimizer_ids)
        log.info("%d unique minimizer hashes across all paths", len(path_min_set))

        # --- Open LMDB indexes ---------------------------------------------------
        lmdb_task = progress.add_task("Opening LMDB indexes …", total=None)
        reads_shards, read_id_width = _open_reads_lmdb(prefix)
        shard_ranges = _reads_shard_ranges(reads_shards, read_id_width)
        reads_txns_dbs = [(env.begin(), db) for env, db, _ in reads_shards]

        mi_shards, _ = _open_minimizer_lmdb(prefix)
        mi_txns_dbs = [(env.begin(), db) for env, db, _ in mi_shards]

        progress.update(
            lmdb_task,
            description=(
                f"Opened {len(reads_shards)} reads shard(s), "
                f"{len(mi_shards)} minimizer shard(s) "
                f"(read_id_width={read_id_width})"
            ),
            completed=1, total=1,
        )
        log.info(
            "LMDB: %d reads shards, %d minimizer shards, read_id_width=%d",
            len(reads_shards), len(mi_shards), read_id_width,
        )

        # --- Query minimizer index for all path minimizers -----------------------
        minimizer_cache = _LRUCache(maxsize=len(path_min_set) + 10_000)
        min_to_read_ids = _collect_candidate_read_ids(
            mi_txns_dbs,
            list(path_min_set),
            max_reads_per_min,
            read_id_width,
            minimizer_cache,
            progress=progress,
        )
        candidate_ids: set[int] = set()
        for ids in min_to_read_ids.values():
            candidate_ids.update(ids)
        log.info(
            "Collected %d candidate read IDs from %d path minimizers",
            len(candidate_ids), len(path_min_set),
        )

        # --- Load FASTQ sequences for candidate reads ----------------------------
        read_seqs = _load_read_sequences(reads, candidate_ids, progress=progress)
        n_missing = len(candidate_ids) - sum(
            1 for r in candidate_ids if r in read_seqs
        )
        if n_missing > 0:
            log.warning(
                "%d candidate read IDs have no matching FASTQ entry "
                "(FASTQ may not cover all reads in the index)",
                n_missing,
            )

        # --- Reconstruct paths ---------------------------------------------------
        read_cache = _LRUCache(maxsize=20_000)
        recon_task = progress.add_task("Reconstructing paths …", total=len(paths))
        n_written = 0
        n_skipped = 0

        with open(output, "w") as out_fh:
            for path_idx, path in enumerate(paths):
                seq, n_mers, n_cov = _reconstruct_path(
                    path=path,
                    mi_txns_dbs=mi_txns_dbs,
                    reads_txns_dbs=reads_txns_dbs,
                    shard_ranges=shard_ranges,
                    read_id_width=read_id_width,
                    read_seqs=read_seqs,
                    read_cache=read_cache,
                    minimizer_cache=minimizer_cache,
                    density=density,
                    min_anchors=min_anchors,
                    terminal_min_anchors=terminal_min_anchors,
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
                        path_idx, n_cov, n_mers,
                    )
                    continue

                header = (
                    f">path_{path_idx} "
                    f"n_minimizers={n_mers} "
                    f"covered_minimizers={n_cov} "
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

        # --- Close LMDB transactions ---------------------------------------------
        for txn, _ in reads_txns_dbs:
            txn.abort()
        for txn, _ in mi_txns_dbs:
            txn.abort()

    typer.echo(
        f"Wrote {n_written} sequences to {output} "
        f"({n_skipped} paths skipped for coverage < {min_cov_fraction:.0%})",
        err=True,
    )


if __name__ == "__main__":
    app()
