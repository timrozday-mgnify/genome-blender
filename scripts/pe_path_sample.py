#!/usr/bin/env python3
"""Sample unambiguous de Bruijn paths that span paired-end read inserts.

For each sampled R1 read the script extends in both directions through the
minimizer-space de Bruijn graph until one direction reaches the paired mate R2
or both directions hit a branching point.  Extension stops at branching (more
than one equally-supported successor), so every output path is an unambiguous
linear subpath.

The output path covers the minimizer span from R1 to R2 (inclusive) in the
direction that successfully connected; the other direction's extension is
discarded.

Usage::

    python scripts/pe_path_sample.py rust_mdbg_out \\
        --k 7 --n-paths 200 --output pe_paths.jsonl

The output JSONL is compatible with ``reconstruct_sequences.py``.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# Import helpers from sibling script.
sys.path.insert(0, str(Path(__file__).parent))
from asf_sample import (  # noqa: E402
    _LRUCache,
    _find_extensions,
    _get_read_cached,
    _open_minimizer_lmdb,
    _open_reads_lmdb,
    _reads_shard_ranges,
)

app = typer.Typer(add_completion=False)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)

_DEFAULT_K: int = 7
_DEFAULT_N_PATHS: int = 200
_DEFAULT_MIN_SUPPORT: int = 2
_DEFAULT_MAX_READS_PER_KMER: int = 200
_DEFAULT_MAX_PATH_MERS: int = 1000
_DEFAULT_SAMPLE_FACTOR: int = 10
_DEFAULT_SEED: int = 42

_READ_CACHE_SIZE: int = 100_000
_MINIMIZER_CACHE_SIZE: int = 50_000
_INTERSECTION_CACHE_SIZE: int = 10_000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PePathResult:
    """A path spanning from a sampled R1 read to its paired mate R2.

    Attributes:
        read_id: The R1 read ID that seeded the path (odd, 1-based).
        minimizer_ids: Ordered minimizer hash IDs along the path.
        distances: Minimizer-count distances between consecutive minimizers.
        support: Read-support counts for each extension step.
    """

    read_id: int
    minimizer_ids: list[int]
    distances: list[int]
    support: list[int]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _mate_id(read_id: int) -> int:
    """Return the paired-end mate ID using rust-mdbg stride-2 assignment.

    R1 reads have odd IDs (``2i - 1``), R2 reads have even IDs (``2i``).

    Args:
        read_id: 1-based integer read ID.

    Returns:
        ID of the paired mate.
    """
    return read_id + 1 if read_id % 2 == 1 else read_id - 1


# ---------------------------------------------------------------------------
# Extension helpers
# ---------------------------------------------------------------------------


def _extend_fwd_to_mate(
    r1_mers: list[int],
    fwd_ext_mers: list[int],
    fwd_ext_dist: list[int],
    fwd_ext_sup: list[int],
    r2_set: set[int],
    k: int,
    min_support: int,
    max_path_mers: int,
    max_reads_per_kmer: int,
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    mi_txns_dbs: list[tuple],
    read_cache: _LRUCache,
    minimizer_cache: _LRUCache,
    intersection_cache: _LRUCache,
    read_id_width: int,
) -> bool:
    """Extend the path forward from R1 until R2 is reached or a branch occurs.

    Appends to *fwd_ext_mers*, *fwd_ext_dist*, and *fwd_ext_sup* in-place.

    Args:
        r1_mers: Minimizer IDs of the R1 read (fixed seed).
        fwd_ext_mers: Accumulator for forward extension minimizers.
        fwd_ext_dist: Accumulator for forward extension distances.
        fwd_ext_sup: Accumulator for forward extension support counts.
        r2_set: Set of R2 minimizer hashes for connection detection.
        k: De Bruijn graph order.
        min_support: Minimum read support to accept an extension.
        max_path_mers: Combined path length limit (r1 + fwd_ext).
        max_reads_per_kmer: Maximum reads examined per lookup.
        reads_txns_dbs: Sharded reads-index ``(txn, db)`` pairs.
        shard_ranges: Per-shard key ranges.
        mi_txns_dbs: Sharded minimizer-index ``(txn, db)`` pairs.
        read_cache: LRU cache for decoded reads.
        minimizer_cache: LRU cache for minimizer→reads lookups.
        intersection_cache: LRU cache for intersection suffix results.
        read_id_width: Byte width of read-ID keys (4 or 8).

    Returns:
        ``True`` if R2 was reached; ``False`` if a branch or dead end stopped
        the extension before reaching R2.
    """
    while len(r1_mers) + len(fwd_ext_mers) < max_path_mers:
        full_path = r1_mers + fwd_ext_mers
        tip_list = full_path[-k:]
        tip = np.array(tip_list, dtype=np.uint64)
        tip_rev = tip[::-1]

        counts, dists = _find_extensions(
            tip, tip_rev,
            reads_txns_dbs, shard_ranges, mi_txns_dbs,
            k, True, max_reads_per_kmer,
            read_cache, minimizer_cache, intersection_cache, None,
            None, read_id_width,
        )
        if not counts:
            break

        top2 = counts.most_common(2)
        best_mid, best_count = top2[0]
        if best_count < min_support:
            break
        if len(top2) > 1 and top2[1][1] >= best_count:
            break  # ambiguous branch

        d_list = dists.get(best_mid, [1])
        median_dist = int(np.median(d_list)) if d_list else 1

        fwd_ext_mers.append(best_mid)
        fwd_ext_dist.append(median_dist)
        fwd_ext_sup.append(best_count)

        if best_mid in r2_set:
            return True

    return False


def _extend_fwd_greedy(
    r1_mers: list[int],
    fwd_ext_mers: list[int],
    fwd_ext_dist: list[int],
    fwd_ext_sup: list[int],
    r2_set: set[int] | None,
    max_path_mers: int,
    k: int,
    min_support: int,
    max_reads_per_kmer: int,
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    mi_txns_dbs: list[tuple],
    read_cache: _LRUCache,
    minimizer_cache: _LRUCache,
    intersection_cache: _LRUCache,
    read_id_width: int,
) -> None:
    """Extend the path forward greedily until a branching point or dead end.

    Two-phase extension: while the R2 bridge has not yet been found the
    combined path length is capped at *max_path_mers*; once the bridge is
    found (or if *r2_set* is ``None``) the cap is lifted and extension
    continues without limit.

    Appends to *fwd_ext_mers*, *fwd_ext_dist*, and *fwd_ext_sup* in-place.

    Args:
        r1_mers: Minimizer IDs of the R1 read (fixed seed).
        fwd_ext_mers: Accumulator for forward extension minimizers.
        fwd_ext_dist: Accumulator for forward extension distances.
        fwd_ext_sup: Accumulator for forward extension support counts.
        r2_set: Set of R2 minimizer hashes used as bridge target, or
            ``None`` to skip bridging and extend without any length cap.
        max_path_mers: Combined path length cap applied only while seeking
            the R2 bridge.  Ignored once the bridge is established.
        k: De Bruijn graph order.
        min_support: Minimum read support to accept an extension.
        max_reads_per_kmer: Maximum reads examined per lookup.
        reads_txns_dbs: Sharded reads-index ``(txn, db)`` pairs.
        shard_ranges: Per-shard key ranges.
        mi_txns_dbs: Sharded minimizer-index ``(txn, db)`` pairs.
        read_cache: LRU cache for decoded reads.
        minimizer_cache: LRU cache for minimizer→reads lookups.
        intersection_cache: LRU cache for intersection suffix results.
        read_id_width: Byte width of read-ID keys (4 or 8).
    """
    bridged = r2_set is None
    while True:
        if not bridged and len(r1_mers) + len(fwd_ext_mers) >= max_path_mers:
            break
        full_path = r1_mers + fwd_ext_mers
        tip_list = full_path[-k:]
        tip = np.array(tip_list, dtype=np.uint64)
        tip_rev = tip[::-1]

        counts, dists = _find_extensions(
            tip, tip_rev,
            reads_txns_dbs, shard_ranges, mi_txns_dbs,
            k, True, max_reads_per_kmer,
            read_cache, minimizer_cache, intersection_cache, None,
            None, read_id_width,
        )
        if not counts:
            break

        top2 = counts.most_common(2)
        best_mid, best_count = top2[0]
        if best_count < min_support:
            break
        if len(top2) > 1 and top2[1][1] >= best_count:
            break  # ambiguous branch

        d_list = dists.get(best_mid, [1])
        median_dist = int(np.median(d_list)) if d_list else 1

        fwd_ext_mers.append(best_mid)
        fwd_ext_dist.append(median_dist)
        fwd_ext_sup.append(best_count)

        if not bridged and best_mid in r2_set:  # type: ignore[operator]
            bridged = True  # lift the length cap; keep extending


def _extend_bwd_greedy(
    r1_mers: list[int],
    bwd_ext_mers: list[int],
    bwd_ext_dist: list[int],
    bwd_ext_sup: list[int],
    k: int,
    min_support: int,
    max_reads_per_kmer: int,
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    mi_txns_dbs: list[tuple],
    read_cache: _LRUCache,
    minimizer_cache: _LRUCache,
    intersection_cache: _LRUCache,
    read_id_width: int,
) -> None:
    """Extend the path backward greedily until a branching point or dead end.

    No length cap is applied; extension continues until a dead end,
    insufficient support, or an ambiguous branch is reached.

    Prepends to *bwd_ext_mers*, *bwd_ext_dist*, and *bwd_ext_sup* in-place.

    Args:
        r1_mers: Minimizer IDs of the R1 read (fixed seed).
        bwd_ext_mers: Accumulator for backward extension minimizers (path order).
        bwd_ext_dist: Accumulator for backward extension distances.
        bwd_ext_sup: Accumulator for backward extension support counts.
        k: De Bruijn graph order.
        min_support: Minimum read support to accept an extension.
        max_reads_per_kmer: Maximum reads examined per lookup.
        reads_txns_dbs: Sharded reads-index ``(txn, db)`` pairs.
        shard_ranges: Per-shard key ranges.
        mi_txns_dbs: Sharded minimizer-index ``(txn, db)`` pairs.
        read_cache: LRU cache for decoded reads.
        minimizer_cache: LRU cache for minimizer→reads lookups.
        intersection_cache: LRU cache for intersection suffix results.
        read_id_width: Byte width of read-ID keys (4 or 8).
    """
    while True:
        full_path = bwd_ext_mers + r1_mers
        tip_list = full_path[:k][::-1]
        tip = np.array(tip_list, dtype=np.uint64)
        tip_rev = tip[::-1]

        counts, dists = _find_extensions(
            tip, tip_rev,
            reads_txns_dbs, shard_ranges, mi_txns_dbs,
            k, False, max_reads_per_kmer,
            read_cache, minimizer_cache, intersection_cache, None,
            None, read_id_width,
        )
        if not counts:
            break

        top2 = counts.most_common(2)
        best_mid, best_count = top2[0]
        if best_count < min_support:
            break
        if len(top2) > 1 and top2[1][1] >= best_count:
            break  # ambiguous branch

        d_list = dists.get(best_mid, [1])
        median_dist = int(np.median(d_list)) if d_list else 1

        bwd_ext_mers.insert(0, best_mid)
        bwd_ext_dist.insert(0, median_dist)
        bwd_ext_sup.insert(0, best_count)


def _extend_bwd_to_mate(
    r1_mers: list[int],
    bwd_ext_mers: list[int],
    bwd_ext_dist: list[int],
    bwd_ext_sup: list[int],
    r2_set: set[int],
    k: int,
    min_support: int,
    max_path_mers: int,
    max_reads_per_kmer: int,
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    mi_txns_dbs: list[tuple],
    read_cache: _LRUCache,
    minimizer_cache: _LRUCache,
    intersection_cache: _LRUCache,
    read_id_width: int,
) -> bool:
    """Extend the path backward from R1 until R2 is reached or a branch occurs.

    Prepends to *bwd_ext_mers*, *bwd_ext_dist*, and *bwd_ext_sup* in-place.
    Index 0 of each list is the outermost (leftmost) minimizer in path order.

    Args:
        r1_mers: Minimizer IDs of the R1 read (fixed seed).
        bwd_ext_mers: Accumulator for backward extension minimizers (path order).
        bwd_ext_dist: Accumulator for backward extension distances.
        bwd_ext_sup: Accumulator for backward extension support counts.
        r2_set: Set of R2 minimizer hashes for connection detection.
        k: De Bruijn graph order.
        min_support: Minimum read support to accept an extension.
        max_path_mers: Combined path length limit (bwd_ext + r1).
        max_reads_per_kmer: Maximum reads examined per lookup.
        reads_txns_dbs: Sharded reads-index ``(txn, db)`` pairs.
        shard_ranges: Per-shard key ranges.
        mi_txns_dbs: Sharded minimizer-index ``(txn, db)`` pairs.
        read_cache: LRU cache for decoded reads.
        minimizer_cache: LRU cache for minimizer→reads lookups.
        intersection_cache: LRU cache for intersection suffix results.
        read_id_width: Byte width of read-ID keys (4 or 8).

    Returns:
        ``True`` if R2 was reached; ``False`` if a branch or dead end stopped
        the extension before reaching R2.
    """
    while len(bwd_ext_mers) + len(r1_mers) < max_path_mers:
        full_path = bwd_ext_mers + r1_mers
        # Backward tip: first k minimizers, reversed for _find_extensions.
        tip_list = full_path[:k][::-1]
        tip = np.array(tip_list, dtype=np.uint64)
        tip_rev = tip[::-1]

        counts, dists = _find_extensions(
            tip, tip_rev,
            reads_txns_dbs, shard_ranges, mi_txns_dbs,
            k, False, max_reads_per_kmer,
            read_cache, minimizer_cache, intersection_cache, None,
            None, read_id_width,
        )
        if not counts:
            break

        top2 = counts.most_common(2)
        best_mid, best_count = top2[0]
        if best_count < min_support:
            break
        if len(top2) > 1 and top2[1][1] >= best_count:
            break  # ambiguous branch

        d_list = dists.get(best_mid, [1])
        median_dist = int(np.median(d_list)) if d_list else 1

        bwd_ext_mers.insert(0, best_mid)
        bwd_ext_dist.insert(0, median_dist)
        bwd_ext_sup.insert(0, best_count)

        if best_mid in r2_set:
            return True

    return False


# ---------------------------------------------------------------------------
# Core path sampling
# ---------------------------------------------------------------------------


def _sample_pe_path(
    r1_id: int,
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    mi_txns_dbs: list[tuple],
    k: int,
    min_support: int,
    max_path_mers: int,
    max_reads_per_kmer: int,
    read_cache: _LRUCache,
    minimizer_cache: _LRUCache,
    intersection_cache: _LRUCache,
    read_id_width: int,
) -> PePathResult | None:
    """Sample one path connecting R1 to its paired mate R2.

    Loads the full minimizer arrays of R1 and R2, then extends the path in
    both directions independently.  The direction that first encounters a
    minimizer from R2 wins; its extension plus the R1 core form the output
    path.  The other direction's extension is discarded.

    Returns ``None`` when R1 or R2 cannot be loaded, R1 has fewer than k
    minimizers, or neither direction reaches R2 before hitting a branching
    point or ``max_path_mers``.

    Args:
        r1_id: 1-based R1 read ID (must be odd).
        reads_txns_dbs: Sharded reads-index ``(txn, db)`` pairs.
        shard_ranges: Per-shard key ranges.
        mi_txns_dbs: Sharded minimizer-index ``(txn, db)`` pairs.
        k: De Bruijn graph order.
        min_support: Minimum read support per extension step.
        max_path_mers: Maximum combined path length before abandoning.
        max_reads_per_kmer: Maximum reads examined per minimizer lookup.
        read_cache: LRU cache for decoded reads.
        minimizer_cache: LRU cache for minimizer→reads lookups.
        intersection_cache: LRU cache for intersection suffix results.
        read_id_width: Byte width of read-ID keys (4 or 8).

    Returns:
        A :class:`PePathResult` if R2 was reached, otherwise ``None``.
    """
    r2_id = _mate_id(r1_id)

    r1_arr = _get_read_cached(reads_txns_dbs, shard_ranges, r1_id, read_cache, read_id_width)
    r2_arr = _get_read_cached(reads_txns_dbs, shard_ranges, r2_id, read_cache, read_id_width)
    if r1_arr is None or len(r1_arr) < k:
        return None
    if r2_arr is None or len(r2_arr) == 0:
        return None

    r1_mers: list[int] = [int(x) for x in r1_arr]
    r2_set: set[int] = {int(x) for x in r2_arr}

    # Separate accumulators for each direction.
    fwd_ext_mers: list[int] = []
    fwd_ext_dist: list[int] = []
    fwd_ext_sup: list[int] = []

    bwd_ext_mers: list[int] = []
    bwd_ext_dist: list[int] = []
    bwd_ext_sup: list[int] = []

    fwd_connected = _extend_fwd_to_mate(
        r1_mers, fwd_ext_mers, fwd_ext_dist, fwd_ext_sup,
        r2_set, k, min_support, max_path_mers, max_reads_per_kmer,
        reads_txns_dbs, shard_ranges, mi_txns_dbs,
        read_cache, minimizer_cache, intersection_cache, read_id_width,
    )

    bwd_connected = _extend_bwd_to_mate(
        r1_mers, bwd_ext_mers, bwd_ext_dist, bwd_ext_sup,
        r2_set, k, min_support, max_path_mers, max_reads_per_kmer,
        reads_txns_dbs, shard_ranges, mi_txns_dbs,
        read_cache, minimizer_cache, intersection_cache, read_id_width,
    )

    r1_base_dist = [1] * (len(r1_mers) - 1)
    r1_base_sup = [1] * (len(r1_mers) - 1)

    if fwd_connected:
        return PePathResult(
            read_id=r1_id,
            minimizer_ids=r1_mers + fwd_ext_mers,
            distances=r1_base_dist + fwd_ext_dist,
            support=r1_base_sup + fwd_ext_sup,
        )

    if bwd_connected:
        return PePathResult(
            read_id=r1_id,
            minimizer_ids=bwd_ext_mers + r1_mers,
            distances=bwd_ext_dist + r1_base_dist,
            support=bwd_ext_sup + r1_base_sup,
        )

    return None


def _sample_greedy_path(
    r1_id: int,
    reads_txns_dbs: list[tuple],
    shard_ranges: list[tuple],
    mi_txns_dbs: list[tuple],
    k: int,
    min_support: int,
    max_path_mers: int,
    max_reads_per_kmer: int,
    read_cache: _LRUCache,
    minimizer_cache: _LRUCache,
    intersection_cache: _LRUCache,
    read_id_width: int,
) -> PePathResult | None:
    """Sample the longest unambiguous path reachable from an R1 seed.

    Extends forward using R2 as an optional bridge target (capped at
    *max_path_mers* until the bridge is established, then unlimited) and
    backward without any length cap.  Both directions extend until they hit a
    dead end, insufficient support, or an ambiguous branch.

    Unlike :func:`_sample_pe_path` this function always returns a
    :class:`PePathResult` as long as R1 can be loaded from the index; it does
    not require R2 to be reached.

    Args:
        r1_id: 1-based R1 read ID (must be odd).
        reads_txns_dbs: Sharded reads-index ``(txn, db)`` pairs.
        shard_ranges: Per-shard key ranges.
        mi_txns_dbs: Sharded minimizer-index ``(txn, db)`` pairs.
        k: De Bruijn graph order.
        min_support: Minimum read support per extension step.
        max_path_mers: Cap applied during the forward bridge-seeking phase
            only.  Ignored once R2 is found or if R2 is unavailable.
        max_reads_per_kmer: Maximum reads examined per minimizer lookup.
        read_cache: LRU cache for decoded reads.
        minimizer_cache: LRU cache for minimizer→reads lookups.
        intersection_cache: LRU cache for intersection suffix results.
        read_id_width: Byte width of read-ID keys (4 or 8).

    Returns:
        A :class:`PePathResult` with the full greedy path, or ``None`` if
        R1 cannot be loaded or has fewer than *k* minimizers.
    """
    r1_arr = _get_read_cached(reads_txns_dbs, shard_ranges, r1_id, read_cache, read_id_width)
    if r1_arr is None or len(r1_arr) < k:
        return None

    r2_id = _mate_id(r1_id)
    r2_arr = _get_read_cached(reads_txns_dbs, shard_ranges, r2_id, read_cache, read_id_width)
    r2_set: set[int] | None = (
        {int(x) for x in r2_arr} if r2_arr is not None and len(r2_arr) > 0 else None
    )

    r1_mers: list[int] = [int(x) for x in r1_arr]

    fwd_ext_mers: list[int] = []
    fwd_ext_dist: list[int] = []
    fwd_ext_sup: list[int] = []

    bwd_ext_mers: list[int] = []
    bwd_ext_dist: list[int] = []
    bwd_ext_sup: list[int] = []

    _extend_fwd_greedy(
        r1_mers, fwd_ext_mers, fwd_ext_dist, fwd_ext_sup,
        r2_set, max_path_mers,
        k, min_support, max_reads_per_kmer,
        reads_txns_dbs, shard_ranges, mi_txns_dbs,
        read_cache, minimizer_cache, intersection_cache, read_id_width,
    )

    _extend_bwd_greedy(
        r1_mers, bwd_ext_mers, bwd_ext_dist, bwd_ext_sup,
        k, min_support, max_reads_per_kmer,
        reads_txns_dbs, shard_ranges, mi_txns_dbs,
        read_cache, minimizer_cache, intersection_cache, read_id_width,
    )

    r1_base_dist = [1] * (len(r1_mers) - 1)
    r1_base_sup = [1] * (len(r1_mers) - 1)

    return PePathResult(
        read_id=r1_id,
        minimizer_ids=bwd_ext_mers + r1_mers + fwd_ext_mers,
        distances=bwd_ext_dist + r1_base_dist + fwd_ext_dist,
        support=bwd_ext_sup + r1_base_sup + fwd_ext_sup,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    prefix: Annotated[Path, typer.Argument(
        help="rust-mdbg output prefix (used to locate LMDB index files).",
    )],
    output: Annotated[Path, typer.Option(
        "--output", "-o",
        help="Output JSONL file (one path per line).",
    )] = Path("pe_paths.jsonl"),
    k: Annotated[int, typer.Option(
        "--k",
        help="De Bruijn graph order (consecutive minimizers per node).",
        min=2,
    )] = _DEFAULT_K,
    n_paths: Annotated[int, typer.Option(
        "--n-paths",
        help="Target number of output paths.",
        min=1,
    )] = _DEFAULT_N_PATHS,
    min_support: Annotated[int, typer.Option(
        "--min-support",
        help="Minimum read support count to accept an extension step.",
        min=1,
    )] = _DEFAULT_MIN_SUPPORT,
    max_reads_per_kmer: Annotated[int, typer.Option(
        "--max-reads-per-kmer",
        help="Maximum candidate reads examined per minimizer lookup.",
        min=1,
    )] = _DEFAULT_MAX_READS_PER_KMER,
    max_path_mers: Annotated[int, typer.Option(
        "--max-path-mers",
        help="In PE mode: abandon a path if the total minimizer count exceeds "
             "this without connecting to the paired mate.  In greedy mode: cap "
             "applied only while seeking the R2 bridge; lifted once the bridge "
             "is established (no upper limit after bridging).",
        min=1,
    )] = _DEFAULT_MAX_PATH_MERS,
    mode: Annotated[Literal["pe", "greedy"], typer.Option(
        "--mode",
        help="Extension strategy.  'pe': extend until R2 is reached (or "
             "branch/dead-end), discard paths that never connect.  'greedy': "
             "extend as far as possible in both directions until an unambiguous "
             "branch or dead end, always emitting a result.",
    )] = "pe",
    min_path_mers: Annotated[int, typer.Option(
        "--min-path-mers",
        help="Discard output paths with fewer minimizers than this threshold.",
        min=0,
    )] = 0,
    sample_factor: Annotated[int, typer.Option(
        "--sample-factor",
        help="R1 IDs to draw per desired output path.  Oversampling "
             "accounts for reads that lack mates in the index or fail to "
             "connect within max_path_mers.",
        min=1,
    )] = _DEFAULT_SAMPLE_FACTOR,
    seed: Annotated[int, typer.Option(
        "--seed",
        help="Random seed for reproducibility.",
    )] = _DEFAULT_SEED,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable debug logging.",
    )] = False,
) -> None:
    """Sample unambiguous de Bruijn paths from paired-end reads.

    Two extension modes are available:

    \b
    pe      Extend forward and backward from the R1 seed until one direction
            reaches R2.  Paths that never connect to R2 are discarded.
    greedy  Extend in both directions as far as possible without ambiguity.
            Forward extension uses R2 as a bridge target (capped by
            --max-path-mers until bridged, then unlimited); backward
            extension has no length cap.  A result is always emitted.
    """
    import random

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    rng = random.Random(seed)

    with Progress(*_PROGRESS_COLUMNS) as progress:

        # --- Open LMDB indexes ---------------------------------------------------
        open_task = progress.add_task("Opening LMDB indexes …", total=None)
        reads_shards, read_id_width = _open_reads_lmdb(prefix)
        shard_ranges = _reads_shard_ranges(reads_shards, read_id_width)
        reads_txns_dbs = [(env.begin(), db) for env, db, _ in reads_shards]

        mi_shards, _ = _open_minimizer_lmdb(prefix)
        mi_txns_dbs = [(env.begin(), db) for env, db, _ in mi_shards]

        progress.update(
            open_task,
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

        # --- Determine pool of R1 IDs to sample ---------------------------------
        valid_ranges = [(lo, hi) for lo, hi in shard_ranges if lo is not None and hi is not None]
        if not valid_ranges:
            typer.echo("ERROR: could not determine read ID range from LMDB.", err=True)
            raise typer.Exit(1)
        max_read_id = max(hi for _, hi in valid_ranges)
        # R1 IDs are odd (1, 3, 5, ...).
        all_r1_ids = list(range(1, max_read_id + 1, 2))
        pool_size = min(n_paths * sample_factor, len(all_r1_ids))
        sample_pool = rng.sample(all_r1_ids, pool_size)
        log.info(
            "Max read ID: %d  Total R1 IDs: %d  Sample pool: %d",
            max_read_id, len(all_r1_ids), pool_size,
        )

        # --- Initialise caches ---------------------------------------------------
        read_cache = _LRUCache(maxsize=_READ_CACHE_SIZE)
        minimizer_cache = _LRUCache(maxsize=_MINIMIZER_CACHE_SIZE)
        intersection_cache = _LRUCache(maxsize=_INTERSECTION_CACHE_SIZE)

        # --- Sample paths --------------------------------------------------------
        task_desc = "Sampling greedy paths …" if mode == "greedy" else "Sampling PE paths …"
        sample_task = progress.add_task(task_desc, total=min(n_paths, pool_size))
        n_written = 0
        n_tried = 0
        n_no_mate = 0
        n_no_connect = 0
        n_no_r1 = 0
        n_too_short = 0

        with open(output, "w") as out_fh:
            for r1_id in sample_pool:
                if n_written >= n_paths:
                    break
                n_tried += 1

                if mode == "greedy":
                    result = _sample_greedy_path(
                        r1_id=r1_id,
                        reads_txns_dbs=reads_txns_dbs,
                        shard_ranges=shard_ranges,
                        mi_txns_dbs=mi_txns_dbs,
                        k=k,
                        min_support=min_support,
                        max_path_mers=max_path_mers,
                        max_reads_per_kmer=max_reads_per_kmer,
                        read_cache=read_cache,
                        minimizer_cache=minimizer_cache,
                        intersection_cache=intersection_cache,
                        read_id_width=read_id_width,
                    )
                    if result is None:
                        n_no_r1 += 1
                        log.debug("R1 %d: not in index or too short", r1_id)
                        continue
                else:
                    result = _sample_pe_path(
                        r1_id=r1_id,
                        reads_txns_dbs=reads_txns_dbs,
                        shard_ranges=shard_ranges,
                        mi_txns_dbs=mi_txns_dbs,
                        k=k,
                        min_support=min_support,
                        max_path_mers=max_path_mers,
                        max_reads_per_kmer=max_reads_per_kmer,
                        read_cache=read_cache,
                        minimizer_cache=minimizer_cache,
                        intersection_cache=intersection_cache,
                        read_id_width=read_id_width,
                    )
                    if result is None:
                        # Distinguish "no mate" from "no connection" via debug logging.
                        r2_id = _mate_id(r1_id)
                        r2_arr = _get_read_cached(
                            reads_txns_dbs, shard_ranges, r2_id, read_cache, read_id_width,
                        )
                        if r2_arr is None:
                            n_no_mate += 1
                            log.debug("R1 %d: mate not in index", r1_id)
                        else:
                            n_no_connect += 1
                            log.debug("R1 %d: failed to connect to mate", r1_id)
                        continue

                if min_path_mers > 0 and len(result.minimizer_ids) < min_path_mers:
                    n_too_short += 1
                    log.debug(
                        "R1 %d: path too short (%d < %d mers)",
                        r1_id, len(result.minimizer_ids), min_path_mers,
                    )
                    continue

                obj = {
                    "read_id": result.read_id,
                    "minimizer_ids": result.minimizer_ids,
                    "distances": result.distances,
                    "support": result.support,
                }
                out_fh.write(json.dumps(obj) + "\n")
                n_written += 1
                progress.advance(sample_task)
                log.debug(
                    "Path %d: read_id=%d  n_mers=%d",
                    n_written, r1_id, len(result.minimizer_ids),
                )

        if mode == "greedy":
            progress.update(
                sample_task,
                description=(
                    f"Sampled {n_written:,}/{n_tried:,} tried "
                    f"({n_no_r1:,} no-R1, {n_too_short:,} too-short)"
                ),
            )
        else:
            progress.update(
                sample_task,
                description=(
                    f"Sampled {n_written:,}/{n_tried:,} tried "
                    f"({n_no_mate:,} no-mate, {n_no_connect:,} no-connect, "
                    f"{n_too_short:,} too-short)"
                ),
            )

        # --- Close LMDB transactions ---------------------------------------------
        for txn, _ in reads_txns_dbs:
            txn.abort()
        for txn, _ in mi_txns_dbs:
            txn.abort()

    if mode == "greedy":
        typer.echo(
            f"Wrote {n_written} paths to {output} "
            f"({n_tried} reads tried, {n_no_r1} missing R1, "
            f"{n_too_short} too short)",
            err=True,
        )
    else:
        typer.echo(
            f"Wrote {n_written} paths to {output} "
            f"({n_tried} reads tried, {n_no_mate} missing mates, "
            f"{n_no_connect} failed to connect, {n_too_short} too short)",
            err=True,
        )


if __name__ == "__main__":
    app()
