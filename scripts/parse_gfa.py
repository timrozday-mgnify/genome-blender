#!/usr/bin/env python3
"""Parse a rust-mdbg GFA output into a rustworkx graph and report properties.

Usage::

    python scripts/parse_gfa.py rust_mdbg_out.gfa
    python scripts/parse_gfa.py rust_mdbg_out.gfa --samples 5000
    python scripts/parse_gfa.py rust_mdbg_out.gfa --no-sample
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import struct
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

import numpy as np

try:
    import lmdb as _lmdb  # type: ignore[import-untyped]
except ImportError:
    _lmdb = None  # type: ignore[assignment]

import rustworkx as rx
import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

try:
    import pysam as _pysam
except ImportError:
    _pysam = None  # type: ignore[assignment]

try:
    import lz4.frame as _lz4  # type: ignore[import-untyped]
except ImportError:
    _lz4 = None  # type: ignore[assignment]

try:
    import torch as _torch
    import pyro as _pyro
    import pyro.distributions as _pyro_dist
    from pyro.infer import MCMC, NUTS, Predictive, SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoNormal
    from pyro.optim import Adam as _PyroAdam
except ImportError:
    _torch = None  # type: ignore[assignment]
    _pyro = None  # type: ignore[assignment]
    _pyro_dist = None  # type: ignore[assignment]
    MCMC = NUTS = Predictive = SVI = Trace_ELBO = None  # type: ignore[assignment,misc]
    AutoNormal = None  # type: ignore[assignment]
    _PyroAdam = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# GFA1 record types (spec: gfa-spec.github.io/GFA-spec/GFA1.html)
_COMMENT = "#"
_HEADER = "H"
_SEGMENT = "S"
_LINK = "L"
_CONTAINMENT = "C"
_PATH = "P"

_PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("[cyan]{task.elapsed:.1f}s"),
)

app = typer.Typer()


class WeightMode(str, Enum):
    """Random walk neighbour-selection weighting."""

    kmer = "kmer"
    overlap = "overlap"
    unweighted = "unweighted"


@dataclass
class Segment:
    """A GFA segment (node).

    Attributes:
        name: Segment name from the GFA S-record.
        sequence: Sequence string, or ``*`` if not stored.
        length: Sequence length (from ``LN`` tag, or
            ``len(sequence)`` if the sequence is present).
        kmer_count: Optional k-mer count from the ``KC`` tag.
    """

    name: str
    sequence: str
    length: int
    kmer_count: int | None = None


@dataclass
class Link:
    """A GFA link (edge).

    Attributes:
        from_orient: Orientation of the source segment (``"+"`` or ``"-"``).
        to_orient: Orientation of the target segment (``"+"`` or ``"-"``).
        overlap: CIGAR overlap string (e.g. ``"100M"``).
        from_idx: Graph node index of the source segment.
        to_idx: Graph node index of the target segment.
    """

    from_orient: str
    to_orient: str
    overlap: str
    from_idx: int = 0
    to_idx: int = 0


def _parse_tags(fields: list[str]) -> dict[str, str]:
    """Parse optional GFA tag fields into a dict.

    Tags have the format ``XX:T:VALUE`` where XX is the tag
    name, T is the type character, and VALUE is the value.
    """
    tags: dict[str, str] = {}
    for field in fields:
        parts = field.split(":", 2)
        if len(parts) == 3:
            tags[parts[0]] = parts[2]
    return tags


def parse_gfa(path: Path) -> rx.PyGraph:
    """Parse a GFA file into an undirected rustworkx graph.

    Segments become nodes (data: ``Segment``).  Links become
    edges (data: ``Link``).  Duplicate links between the same
    pair of segments are allowed (multigraph).

    Args:
        path: Path to the GFA file.

    Returns:
        A ``PyGraph`` with ``Segment`` node data and ``Link``
        edge data.
    """
    graph: rx.PyGraph = rx.PyGraph()
    name_to_idx: dict[str, int] = {}

    lines = path.read_text().splitlines()

    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Reading GFA", total=len(lines),
        )
        for line_no, line in enumerate(lines, start=1):
            progress.advance(task)
            if not line:
                continue

            fields = line.split("\t")
            record_type = fields[0]

            if record_type in (_HEADER, _COMMENT):
                continue

            if record_type == _SEGMENT:
                _add_segment(
                    graph, name_to_idx, fields, line_no,
                )
            elif record_type == _LINK:
                _add_link(
                    graph, name_to_idx, fields, line_no,
                )
            else:
                logger.debug(
                    "Skipping unknown record type %r "
                    "at line %d",
                    record_type, line_no,
                )

    logger.info(
        "Parsed %s: %d segments, %d links",
        path, graph.num_nodes(), graph.num_edges(),
    )
    return graph


def _add_segment(
    graph: rx.PyGraph,
    name_to_idx: dict[str, int],
    fields: list[str],
    line_no: int,
) -> None:
    """Parse an S-record and add a node to the graph."""
    if len(fields) < 3:
        logger.warning(
            "Malformed S-record at line %d: %r",
            line_no, fields,
        )
        return

    name = fields[1]
    sequence = fields[2]
    tags = _parse_tags(fields[3:])

    if sequence != "*":
        length = len(sequence)
    elif "LN" in tags:
        length = int(tags["LN"])
    else:
        length = 0
        logger.warning(
            "Segment %s has no sequence and no LN tag "
            "(line %d)",
            name, line_no,
        )

    kmer_count = (
        int(tags["KC"]) if "KC" in tags else None
    )

    segment = Segment(
        name=name,
        sequence=sequence,
        length=length,
        kmer_count=kmer_count,
    )
    idx = graph.add_node(segment)
    name_to_idx[name] = idx


def _add_link(
    graph: rx.PyGraph,
    name_to_idx: dict[str, int],
    fields: list[str],
    line_no: int,
) -> None:
    """Parse an L-record and add an edge to the graph."""
    if len(fields) < 6:
        logger.warning(
            "Malformed L-record at line %d: %r",
            line_no, fields,
        )
        return

    from_name = fields[1]
    from_orient = fields[2]
    to_name = fields[3]
    to_orient = fields[4]
    overlap = fields[5]

    from_idx = name_to_idx.get(from_name)
    to_idx = name_to_idx.get(to_name)

    if from_idx is None or to_idx is None:
        logger.warning(
            "Link references unknown segment(s) "
            "at line %d: %s -> %s",
            line_no, from_name, to_name,
        )
        return

    link = Link(
        from_orient=from_orient,
        to_orient=to_orient,
        overlap=overlap,
        from_idx=from_idx,
        to_idx=to_idx,
    )
    graph.add_edge(from_idx, to_idx, link)


# ------------------------------------------------------------------
# Combination minimizer sketch
# ------------------------------------------------------------------

_READS_LMDB_MAP_SIZE: int = 32 * 1024 ** 3
_READS_LMDB_MAX_DBS: int = 6

# Maximum value of a uint64 — used as the fixed upper bound for combo hash
# thinning so that threshold = density × UINT64_MAX is identical across all
# sketch types (PE, intra, path).  Splitmix64 outputs are uniform over [0,
# UINT64_MAX], so training a per-dataset max_hash is unnecessary.
_UINT64_MAX: int = (1 << 64) - 1

# splitmix64 mixing constants as numpy uint64 scalars
_CM1 = np.uint64(0x9E3779B97F4A7C15)
_CM2 = np.uint64(0xBF58476D1CE4E5B9)
_CM3 = np.uint64(0x94D049BB133111EB)
_CR30 = np.uint64(30)
_CR27 = np.uint64(27)
_CR31 = np.uint64(31)


def _combo_hash_v(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised splitmix64 pair hash over uint64 numpy arrays."""
    h: np.ndarray = a ^ (b * _CM1)
    h = (h ^ (h >> _CR30)) * _CM2
    h = (h ^ (h >> _CR27)) * _CM3
    return h ^ (h >> _CR31)


def _chain_hash(rows: np.ndarray) -> np.ndarray:
    """Vectorised sequential splitmix64 reduction along columns of a 2-D matrix.

    Each row is an ordered sequence of uint64 values.  The first column
    seeds the hash; each subsequent column is mixed in with the same
    splitmix64-inspired round used by :func:`_combo_hash_v`.  For a
    single-column matrix the raw seed value is returned unchanged.

    Args:
        rows: Shape ``[N, K]`` uint64 array; each row is a k-mer sequence.

    Returns:
        Shape ``[N]`` uint64 hash array.
    """
    h: np.ndarray = rows[:, 0].copy()
    for col in range(1, rows.shape[1]):
        h ^= rows[:, col] * _CM1
        h = (h ^ (h >> _CR30)) * _CM2
        h = (h ^ (h >> _CR27)) * _CM3
        h ^= h >> _CR31
    return h


def _compute_kmer_hashes(mids: np.ndarray, k: int) -> np.ndarray:
    """Compute canonical minimizer-space k-mer hashes for a sequence.

    Slides a window of *k* consecutive minimizer IDs over *mids* and
    hashes each window both forward (left-to-right) and in reverse
    (right-to-left) via :func:`_chain_hash`.  The canonical hash is
    ``min(forward, reverse)``, making it invariant to sequence orientation.

    For ``k == 1`` the result is the raw minimizer ID array (forward ==
    reverse for single elements, no mixing applied).

    Args:
        mids: 1-D uint64 array of minimizer IDs.
        k: Window size — number of consecutive minimizers per k-mer.

    Returns:
        1-D uint64 array of length ``max(0, len(mids) - k + 1)``.
    """
    n = len(mids)
    if n < k:
        return np.empty(0, dtype=np.uint64)
    if k == 1:
        return mids.copy()
    n_kmers = n - k + 1
    offsets = np.arange(k, dtype=np.intp)
    starts = np.arange(n_kmers, dtype=np.intp)
    windows: np.ndarray = mids[starts[:, None] + offsets]  # [n_kmers, k]
    h_fwd = _chain_hash(windows)
    h_rev = _chain_hash(windows[:, ::-1])
    return np.minimum(h_fwd, h_rev)


def _canonical_combo_hash_v(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Commutative combination hash: ``hash(a, b) == hash(b, a)`` element-wise.

    Achieved by sorting each pair so that the smaller value is always
    passed as the first argument to :func:`_combo_hash_v`.

    Args:
        a: First uint64 array.
        b: Second uint64 array (same shape as *a*).

    Returns:
        uint64 array of combination hashes, same shape as inputs.
    """
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return _combo_hash_v(lo, hi)


class CombSketchIndex:
    """Combination minimizer sketch: maps combo_hash → occurrence count.

    Backed by an in-memory :class:`dict` (default) or an on-disk LMDB
    key-value store.  Call :meth:`close` when done to flush any remaining
    buffered counts to the LMDB.
    """

    _FLUSH_BUFFER: int = 200_000

    def __init__(self, lmdb_path: Path | None = None) -> None:
        """Initialise index.

        Args:
            lmdb_path: If given, persist counts to an LMDB directory at
                this path (in-memory write buffer is flushed periodically).
        """
        self._buf: dict[int, int] = {}
        self._env = None
        self._lmdb_db = None
        if lmdb_path is not None:
            if _lmdb is None:
                raise RuntimeError(
                    "lmdb package required for on-disk mode; "
                    "install with: pip install lmdb"
                )
            lmdb_path.mkdir(parents=True, exist_ok=True)
            self._env = _lmdb.open(
                str(lmdb_path),
                map_size=4 * 1024 ** 3,
                max_dbs=1,
            )
            self._lmdb_db = self._env.open_db(b"combo")

    @property
    def counts(self) -> dict[int, int]:
        """In-memory count dict (available in memory mode only)."""
        if self._env is not None:
            raise RuntimeError(
                "counts not directly accessible in on-disk LMDB mode"
            )
        return self._buf

    def increment(self, h: int) -> None:
        """Increment the occurrence count of combination hash *h* by 1."""
        self._buf[h] = self._buf.get(h, 0) + 1
        if self._env is not None and len(self._buf) >= self._FLUSH_BUFFER:
            self._flush()

    def _bulk_add(self, hashes: np.ndarray) -> None:
        """Add an array of (already thinned) combination hashes to the index."""
        if len(hashes) == 0:
            return
        ukeys, ucounts = np.unique(hashes, return_counts=True)
        for h, c in zip(ukeys.tolist(), ucounts.tolist()):
            self._buf[h] = self._buf.get(h, 0) + c
        if self._env is not None and len(self._buf) >= self._FLUSH_BUFFER:
            self._flush()

    def _flush(self) -> None:
        if not self._buf or self._env is None:
            return
        with self._env.begin(write=True) as txn:
            for h, delta in self._buf.items():
                key = struct.pack("<Q", h)
                raw = txn.get(key, db=self._lmdb_db)
                old = struct.unpack("<I", raw)[0] if raw else 0
                txn.put(
                    key,
                    struct.pack("<I", min(old + delta, 0xFFFFFFFF)),
                    db=self._lmdb_db,
                )
        self._buf.clear()

    def close(self) -> None:
        """Flush remaining counts and close LMDB (no-op in memory mode)."""
        self._flush()
        if self._env is not None:
            self._env.close()
            self._env = None

    def __len__(self) -> int:
        if self._env is None:
            return len(self._buf)
        with self._env.begin() as txn:
            return txn.stat(db=self._lmdb_db)["entries"] + len(self._buf)


def _iter_reads_lmdb(
    lmdb_path: Path,
    limit: int | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(read_id_0based, minimizer_ids)`` from a rust-mdbg reads LMDB.

    Iterates the ``reads`` sub-database in ascending *numeric* key order.
    Keys are 1-based ASCII-decimal read indices (e.g. ``b'1'``, ``b'42'``);
    values are packed little-endian u64 minimizer IDs.

    The LMDB default cursor iterates keys in lexicographic order, which does
    not preserve numeric order for ASCII-encoded integers.  This function
    collects all keys, sorts them numerically, and yields entries in that
    order so that read pairs (odd/even 1-based IDs) are adjacent.

    Args:
        lmdb_path: LMDB environment directory.
        limit: Stop after yielding this many entries (``None`` = no limit).

    Raises:
        RuntimeError: If the ``lmdb`` package is not installed.
    """
    if _lmdb is None:
        raise RuntimeError(
            "lmdb package required; install with: pip install lmdb"
        )
    env = _lmdb.open(
        str(lmdb_path), readonly=True, lock=False,
        max_dbs=_READS_LMDB_MAX_DBS, map_size=_READS_LMDB_MAP_SIZE,
    )
    reads_db = env.open_db(b"reads")
    try:
        with env.begin() as txn:
            # Collect all keys and sort numerically.
            # ASCII-decimal keys are typically short (~1–7 bytes for ≤10M reads).
            all_keys: list[bytes] = []
            cur = txn.cursor(db=reads_db)
            for key in cur.iternext(keys=True, values=False):
                try:
                    int(key)  # validate parseable
                    all_keys.append(key)
                except (ValueError, UnicodeDecodeError):
                    continue
            all_keys.sort(key=lambda k: int(k))

            count = 0
            for key in all_keys:
                if limit is not None and count >= limit:
                    break
                val = txn.get(key, db=reads_db)
                if val is None:
                    continue
                read_id = int(key) - 1  # 1-based → 0-based
                n = len(val) // 8
                mids = (
                    np.frombuffer(val, dtype="<u8").copy()
                    if n else np.empty(0, dtype=np.uint64)
                )
                yield read_id, mids
                count += 1
    finally:
        env.close()


def build_pe_combo_sketch(
    lmdb_path: Path,
    density: float,
    k: int = 7,
    out_lmdb: Path | None = None,
) -> CombSketchIndex:
    """Build a cross-pair (R1 × R2) combination minimizer sketch.

    For each interleaved read pair, first converts each read's minimizer ID
    sequence into canonical k-mer hashes via :func:`_compute_kmer_hashes`
    (``k`` consecutive minimizer IDs per k-mer, orientation-invariant).
    Then computes the Cartesian product of R1 and R2 k-mer hash arrays and
    hashes each ``(kmer_R1, kmer_R2)`` pair via
    :func:`_canonical_combo_hash_v` (commutative — same value regardless of
    pair order).  Retains hashes ≤ ``_UINT64_MAX × density``.

    The thinning threshold is fixed at ``int(density × _UINT64_MAX)`` — no
    training is needed because splitmix64 outputs are uniform over uint64.

    Args:
        lmdb_path: rust-mdbg reads LMDB directory (``{prefix}.index.lmdb``).
        density: Fraction of combination hashes to retain (0 < density ≤ 1).
        k: Number of consecutive minimizer IDs per k-mer (default 7, matching
            the rust-mdbg ``-k`` parameter so each k-mer maps to one graph node).
        out_lmdb: If given, persist the sketch to this LMDB directory.

    Returns:
        Populated :class:`CombSketchIndex`.
    """
    threshold = np.uint64(int(_UINT64_MAX * density))
    logger.info(
        "PE combo thinning: threshold=%d density=%.4f", int(threshold), density,
    )

    index = CombSketchIndex(lmdb_path=out_lmdb)
    r1_kmers: np.ndarray | None = None
    pairs_done = 0
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task("PE combo sketch", total=None)
        for read_id, mids in _iter_reads_lmdb(lmdb_path):
            if read_id % 2 == 0:
                r1_kmers = _compute_kmer_hashes(mids, k)
            elif r1_kmers is not None:
                r2_kmers = _compute_kmer_hashes(mids, k)
                if len(r1_kmers) > 0 and len(r2_kmers) > 0:
                    hashes = _canonical_combo_hash_v(
                        np.repeat(r1_kmers, len(r2_kmers)),
                        np.tile(r2_kmers, len(r1_kmers)),
                    )
                    index._bulk_add(hashes[hashes <= threshold])
                r1_kmers = None
                pairs_done += 1
                if pairs_done % 50_000 == 0:
                    progress.update(task, completed=pairs_done)
        progress.update(task, completed=pairs_done)

    return index


def build_intra_combo_sketch(
    lmdb_path: Path,
    density: float,
    k: int = 7,
    out_lmdb: Path | None = None,
) -> CombSketchIndex:
    """Build a within-read combination minimizer sketch.

    For each read, converts its minimizer ID sequence into canonical k-mer
    hashes via :func:`_compute_kmer_hashes`, then computes all C(n, 2) pairs
    of k-mer hashes via :func:`_canonical_combo_hash_v` (commutative), and
    retains hashes ≤ ``_UINT64_MAX × density``.

    The thinning threshold is fixed at ``int(density × _UINT64_MAX)`` — no
    training is needed because splitmix64 outputs are uniform over uint64.

    Args:
        lmdb_path: rust-mdbg reads LMDB directory (``{prefix}.index.lmdb``).
        density: Fraction of combination hashes to retain.
        k: Number of consecutive minimizer IDs per k-mer.
        out_lmdb: If given, persist the sketch to this LMDB directory.

    Returns:
        Populated :class:`CombSketchIndex`.
    """
    threshold = np.uint64(int(_UINT64_MAX * density))
    logger.info(
        "Intra combo thinning: threshold=%d density=%.4f", int(threshold), density,
    )

    index = CombSketchIndex(lmdb_path=out_lmdb)
    reads_done = 0
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task("Intra combo sketch", total=None)
        for _, mids in _iter_reads_lmdb(lmdb_path):
            kmers = _compute_kmer_hashes(mids, k)
            n_k = len(kmers)
            if n_k >= 2:
                i_idx, j_idx = np.triu_indices(n_k, k=1)
                hashes = _canonical_combo_hash_v(kmers[i_idx], kmers[j_idx])
                index._bulk_add(hashes[hashes <= threshold])
            reads_done += 1
            if reads_done % 100_000 == 0:
                progress.update(task, completed=reads_done)
        progress.update(task, completed=reads_done)

    return index


_RC_TABLE = str.maketrans("ACGTacgt", "TGCAtgca")


def _reverse_complement(seq: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return seq.translate(_RC_TABLE)[::-1]


def _compute_path_minimizer_positions(
    path: _OrientedPath,
    path_min_seq: np.ndarray,
    graph: rx.PyGraph,
    min_table: dict[int, str],
    l: int,
) -> np.ndarray | None:
    """Compute exact base-pair positions for every minimizer in a path.

    Reconstructs the full base-pair sequence of the path by concatenating
    GFA segment sequences (reverse-complementing reversed segments) with
    overlap correction from each link's CIGAR string.  Then scans the
    reconstructed sequence for each l-mer in *path_min_seq* (looked up via
    *min_table*) and records its start position.

    Minimizers whose l-mer is absent from *min_table* or whose sequence is
    unavailable in the GFA cause the function to return ``None``.  The caller
    should fall back to the approximate distance method in that case.

    Args:
        path: Ordered list of ``(node_idx, is_forward)`` pairs.
        path_min_seq: 1-D uint64 array of minimizer IDs in traversal order.
        graph: Parsed GFA graph (segment sequences must be present).
        min_table: Minimizer hash → l-mer string mapping.
        l: l-mer length (minimizer size).

    Returns:
        Int64 numpy array of bp positions (one per minimizer), or ``None``
        if exact positions cannot be determined.
    """
    # --- Build lmer → hash lookup (forward and RC both map to the same hash) -
    lmer_to_hash: dict[str, int] = {}
    for h_raw in path_min_seq.tolist():
        h = int(h_raw)
        lmer = min_table.get(h)
        if lmer is None:
            logger.debug(
                "Minimizer hash %d not in minimizer table; "
                "falling back to approximate distances", h,
            )
            return None
        lmer_to_hash[lmer] = h
        lmer_to_hash[_reverse_complement(lmer)] = h

    # --- Reconstruct path sequence with overlap-corrected concatenation ------
    seq_parts: list[str] = []
    for seg_i, (node_idx, is_forward) in enumerate(path):
        seg = graph[node_idx]
        if not seg.sequence or seg.sequence == "*":
            logger.debug(
                "Segment %s has no sequence; falling back to approximate distances",
                seg.name,
            )
            return None
        seq = seg.sequence if is_forward else _reverse_complement(seg.sequence)
        if seg_i == 0:
            seq_parts.append(seq)
        else:
            prev_idx = path[seg_i - 1][0]
            overlap_bp = 0
            edges = graph.get_all_edge_data(prev_idx, node_idx)
            if edges:
                overlap_bp = int(_overlap_length(edges[0].overlap))
            seq_parts.append(seq[overlap_bp:])

    full_seq = "".join(seq_parts)

    # --- Scan for l-mer positions -------------------------------------------
    pos_by_hash: dict[int, list[int]] = {}
    for p in range(len(full_seq) - l + 1):
        lmer = full_seq[p: p + l]
        h = lmer_to_hash.get(lmer)
        if h is not None:
            pos_by_hash.setdefault(h, []).append(p)

    # --- Match each minimizer in traversal order to a position ---------------
    positions = np.zeros(len(path_min_seq), dtype=np.int64)
    last_pos = -1
    for i, h_raw in enumerate(path_min_seq.tolist()):
        h = int(h_raw)
        candidates = [p for p in pos_by_hash.get(h, []) if p > last_pos]
        if not candidates:
            logger.debug(
                "Could not locate minimizer %d (index %d) in reconstructed "
                "path sequence; falling back to approximate distances", h, i,
            )
            return None
        positions[i] = candidates[0]
        last_pos = candidates[0]

    return positions


def build_path_combo_sketch(
    path_min_seq: np.ndarray,
    bin_distances: list[float],
    density: float,
    bp_scale: float,
    k: int = 7,
    use_exact_distances: bool = False,
    path: _OrientedPath | None = None,
    graph: rx.PyGraph | None = None,
    min_table: dict[int, str] | None = None,
    l: int = 12,
) -> list[CombSketchIndex]:
    """Build a distance-binned combination minimizer sketch for a path.

    For every pair of minimizers ``(i, j)`` in the path (``i < j``) whose
    base-pair distance falls within the range
    ``[bin_distances[0], bin_distances[-1]]``, a combination hash is computed
    via :func:`_combo_hash_v` and — if it survives thinning — added to the
    sketch for the appropriate distance bin.

    **Distance bins** are defined by the sorted *bin_distances* breakpoints.
    *n* breakpoints produce *n − 1* bins:

    * Bin 0: ``[bin_distances[0], bin_distances[1])``
    * Bin 1: ``[bin_distances[1], bin_distances[2])``
    * …
    * Bin n−2: ``[bin_distances[n−2], bin_distances[n−1]]``

    Minimizer pairs outside ``[bin_distances[0], bin_distances[-1]]`` are
    discarded.

    **Distance methods:**

    * *Approximate* (default, ``use_exact_distances=False``): the bp distance
      between minimizers at path indices *i* and *j* is ``(j − i) × bp_scale``,
      where *bp_scale* is the average number of base pairs per minimizer
      position derived from GFA assembly statistics.

    * *Exact* (``use_exact_distances=True``): the path's full base-pair
      sequence is reconstructed by concatenating GFA segment sequences
      (reverse-complementing reversed segments) with overlap correction from
      each link's CIGAR string.  The reconstructed sequence is then scanned
      for each l-mer from *min_table* to recover exact bp positions.  Falls
      back silently to the approximate method if *path*, *graph*, or
      *min_table* are not supplied or if any minimizer cannot be located.

    Args:
        path_min_seq: 1-D uint64 array of minimizer IDs for the path in
            traversal order.
        bin_distances: Sorted base-pair breakpoints defining bins and limits.
            Must have at least 2 values.
        density: Thinning density — fraction of combo hashes to retain.
        bp_scale: Average base pairs per minimizer (used for the approximate
            method and as fallback).
        k: Number of consecutive minimizer IDs per k-mer.  Each position *i*
            in the resulting k-mer array corresponds to minimizer position *i*
            in *path_min_seq*, so distance calculations are unchanged.
        use_exact_distances: Use exact bp positions from sequence
            reconstruction rather than the average-scale approximation.
        path: Ordered ``(node_idx, is_forward)`` path — required for exact
            distances.
        graph: Parsed GFA graph — required for exact distances.
        min_table: Minimizer hash → l-mer string mapping — required for
            exact distances.
        l: l-mer length; used to scan the reconstructed sequence.

    Returns:
        List of :class:`CombSketchIndex` instances, one per bin
        (length = ``len(bin_distances) − 1``).
    """
    if len(bin_distances) < 2:
        raise ValueError("bin_distances must have at least 2 values")

    n_bins = len(bin_distances) - 1
    bin_arr = np.array(bin_distances, dtype=np.float64)

    # --- Resolve bp positions ------------------------------------------------
    exact_positions: np.ndarray | None = None
    if use_exact_distances:
        if path is not None and graph is not None and min_table is not None:
            exact_positions = _compute_path_minimizer_positions(
                path, path_min_seq, graph, min_table, l,
            )
            if exact_positions is None:
                logger.warning(
                    "Exact distance computation failed; "
                    "falling back to approximate bp_scale method.",
                )
        else:
            logger.warning(
                "--exact-path-distances requested but path/graph/min_table "
                "not supplied; falling back to approximate method.",
            )

    # --- Convert distance limits to index-space window (approximate path) ----
    # Used when exact_positions is None.
    min_gap = max(1, int(bin_distances[0] / bp_scale))
    max_gap = int(bin_distances[-1] / bp_scale) + 1

    threshold = np.uint64(int(_UINT64_MAX * density))
    logger.info(
        "Path combo thinning: threshold=%d density=%.4f", int(threshold), density,
    )

    # Convert minimizer sequence to canonical k-mer hashes.
    # k-mer index i corresponds to minimizer index i (start of window), so
    # all distance calculations and bp_scale usage remain unchanged.
    path_kmer_seq = _compute_kmer_hashes(path_min_seq, k)

    sketches: list[CombSketchIndex] = [CombSketchIndex() for _ in range(n_bins)]
    n = len(path_kmer_seq)

    if exact_positions is not None:
        # Exact method: use reconstructed bp positions.
        # Trim exact_positions to n k-mer positions (each k-mer starts at the
        # corresponding minimizer index, so positions[:n] is correct).
        exact_pos_k = exact_positions[:n]
        # Same single-occurrence deduplication as the approximate method.
        hash_to_bin_ex: dict[int, int] = {}
        for i in range(n):
            pos_i = exact_pos_k[i]
            lo_bp = pos_i + bin_distances[0]
            hi_bp = pos_i + bin_distances[-1]
            j_lo = int(np.searchsorted(exact_pos_k, lo_bp, side="left"))
            j_hi = int(np.searchsorted(exact_pos_k, hi_bp, side="right")) - 1
            j_lo = max(j_lo, i + 1)
            if j_lo > j_hi:
                continue

            j_range = np.arange(j_lo, j_hi + 1, dtype=np.intp)
            a_arr = np.full(len(j_range), path_kmer_seq[i], dtype=np.uint64)
            hashes = _canonical_combo_hash_v(a_arr, path_kmer_seq[j_range])

            keep = hashes <= threshold
            if not np.any(keep):
                continue

            kept_hashes = hashes[keep]
            kept_dists = (exact_pos_k[j_range[keep]] - pos_i).astype(np.float64)
            bin_indices = np.searchsorted(bin_arr, kept_dists, side="right") - 1
            for h_val, b_idx in zip(kept_hashes.tolist(), bin_indices.tolist()):
                if 0 <= b_idx < n_bins:
                    if h_val in hash_to_bin_ex:
                        if hash_to_bin_ex[h_val] != b_idx:
                            hash_to_bin_ex[h_val] = -1
                    else:
                        hash_to_bin_ex[h_val] = b_idx

        for h_val, b_idx in hash_to_bin_ex.items():
            if b_idx >= 0:
                sketches[b_idx].increment(h_val)
    else:
        # Approximate method: scale minimizer-index gaps by bp_scale.
        # Use single-occurrence deduplication: each combo hash is assigned to
        # exactly one bin (the shortest distance at which it first appears).
        # Hashes that appear in more than one bin (due to repeated k-mer hashes
        # in tandem-repeat or cyclic path sequences) are discarded entirely.
        # -1 in hash_to_bin signals a multi-bin conflict.
        hash_to_bin: dict[int, int] = {}
        for i in range(n):
            j_lo = i + min_gap
            j_hi = min(i + max_gap, n - 1)
            if j_lo > n - 1:
                break

            j_range = np.arange(j_lo, j_hi + 1, dtype=np.intp)
            a_arr = np.full(len(j_range), path_kmer_seq[i], dtype=np.uint64)
            hashes = _canonical_combo_hash_v(a_arr, path_kmer_seq[j_range])

            keep = hashes <= threshold
            if not np.any(keep):
                continue

            kept_hashes = hashes[keep]
            kept_dists = (j_range[keep] - i).astype(np.float64) * bp_scale
            bin_indices = np.searchsorted(bin_arr, kept_dists, side="right") - 1
            for h_val, b_idx in zip(kept_hashes.tolist(), bin_indices.tolist()):
                if 0 <= b_idx < n_bins:
                    if h_val in hash_to_bin:
                        if hash_to_bin[h_val] != b_idx:
                            hash_to_bin[h_val] = -1  # mark multi-bin conflict
                    else:
                        hash_to_bin[h_val] = b_idx

        for h_val, b_idx in hash_to_bin.items():
            if b_idx >= 0:
                sketches[b_idx].increment(h_val)

    return sketches


# ------------------------------------------------------------------
# Path sampling
# ------------------------------------------------------------------

_CIGAR_RE = re.compile(r"(\d+)[MIDNSHP=X]")

# Oriented path: list of (node_idx, is_forward) pairs where
# is_forward=True means minimizers are used in stored order,
# is_forward=False means they are reversed.
_OrientedPath = list[tuple[int, bool]]


def _opp(orient: str) -> str:
    """Return the opposite orientation character."""
    return "-" if orient == "+" else "+"


def _next_orient(
    cur_idx: int,
    cur_orient: str,
    link: Link,
) -> str | None:
    """Return the orientation of the neighbour reachable via *link*.

    A GFA L-record encodes a bidirected edge with two valid traversals:

    * Forward: ``from_idx(from_orient) → to_idx(to_orient)``
    * Backward: ``to_idx(opp(to_orient)) → from_idx(opp(from_orient))``

    Returns ``None`` when *link* is not usable from ``(cur_idx, cur_orient)``.

    Args:
        cur_idx: Current node index.
        cur_orient: Current traversal orientation (``"+"`` or ``"-"``).
        link: Edge data from the GFA L-record.

    Returns:
        The orientation string for the next node, or ``None`` if the edge
        cannot be traversed from the given state.
    """
    if cur_idx == link.from_idx and cur_orient == link.from_orient:
        return link.to_orient
    if cur_idx == link.to_idx and cur_orient == _opp(link.to_orient):
        return _opp(link.from_orient)
    return None


def _overlap_length(cigar: str) -> float:
    """Return the total length of a CIGAR overlap string.

    Sums all numeric lengths regardless of operation type.
    Returns 1.0 for ``*`` (unspecified) so it acts as a
    neutral weight.
    """
    if cigar == "*":
        return 1.0
    lengths = _CIGAR_RE.findall(cigar)
    return float(sum(int(n) for n in lengths)) if lengths else 1.0


def leaf_nodes(graph: rx.PyGraph) -> list[int]:
    """Return node indices whose degree is exactly 1.

    Leaf nodes have only one link, so random walks starting
    from them always travel inward and produce paths that
    span from one graph boundary to another.

    Args:
        graph: The assembled sequence graph.

    Returns:
        List of node indices with degree == 1.
    """
    return [
        idx for idx in graph.node_indices()
        if graph.degree(idx) == 1
    ]


def large_component_nodes(
    graph: rx.PyGraph,
    min_proportion: float,
) -> tuple[set[int], list[set[int]]]:
    """Return node indices in the largest components covering *min_proportion*.

    Components are sorted by total base-pair span (sum of segment lengths)
    and added largest-first until the cumulative node count reaches or
    exceeds ``min_proportion * graph.num_nodes()``.  When *min_proportion*
    is 0 all components are returned.

    Args:
        graph: The assembled sequence graph.
        min_proportion: Fraction of total nodes that selected components
            must collectively cover (0.0–1.0).  Pass 0.0 to select all.

    Returns:
        Tuple of ``(selected_nodes, selected_components)`` where
        *selected_nodes* is the union of all selected component node sets
        and *selected_components* is the list of chosen components ordered
        largest-first by total base-pair span.
    """
    components = sorted(
        rx.connected_components(graph),
        key=lambda c: sum(graph[idx].length for idx in c),
        reverse=True,
    )
    if min_proportion <= 0.0:
        return set().union(*components) if components else set(), components
    threshold = math.ceil(min_proportion * graph.num_nodes())
    selected: set[int] = set()
    chosen: list[set[int]] = []
    for comp in components:
        selected |= comp
        chosen.append(comp)
        if len(selected) >= threshold:
            break
    return selected, chosen


def _pick_by_kmer(
    graph: rx.PyGraph,
    cands: list[int],
    _edges: dict[int, Link],
) -> int:
    """Choose next node weighted by k-mer count."""
    weights = [float(graph[n].kmer_count or 1) for n in cands]
    return random.choices(cands, weights=weights)[0]


def _pick_by_overlap(
    _graph: rx.PyGraph,
    cands: list[int],
    edges: dict[int, Link],
) -> int:
    """Choose next node weighted by overlap length."""
    weights = [_overlap_length(edges[n].overlap) for n in cands]
    return random.choices(cands, weights=weights)[0]


def _pick_uniformly(
    _graph: rx.PyGraph,
    cands: list[int],
    _edges: dict[int, Link],
) -> int:
    """Choose next node uniformly at random."""
    return random.choice(cands)


_PICK_FN = {
    "kmer": _pick_by_kmer,
    "overlap": _pick_by_overlap,
    "unweighted": _pick_uniformly,
}


def _random_simple_path(
    graph: rx.PyGraph,
    start_nodes: list[int],
    weight_mode: str,
) -> _OrientedPath:
    """Sample a random simple path by greedy random walk.

    Start from a randomly chosen leaf node, then repeatedly
    extend to an unvisited neighbour, chosen according to
    *weight_mode*.  Each step tracks the traversal orientation
    of the next node by consulting the GFA edge orientations
    stored in each :class:`Link`.  This is necessary because
    rust-mdbg stores k-mer nodes in their canonical (normalised)
    form, so some nodes must be reversed when concatenating
    minimizer sequences.

    The start orientation is derived from the first incident edge:
    leaf nodes have exactly one edge, so the valid traversal
    direction is unambiguous.

    Args:
        graph: The graph to walk.
        start_nodes: Pool of nodes to pick a start from.
        weight_mode: One of ``"kmer"``, ``"overlap"``,
            ``"unweighted"``.

    Returns:
        Ordered list of ``(node_idx, is_forward)`` pairs where
        ``is_forward=True`` means the node's stored minimizer
        sequence is used as-is and ``False`` means it is reversed.
    """
    pick = _PICK_FN.get(weight_mode, _pick_uniformly)
    start = random.choice(start_nodes)

    # Determine start orientation from the first incident edge.
    adj_start = graph.adj(start)
    if adj_start:
        first_link: Link = next(iter(adj_start.values()))
        start_orient = (
            first_link.from_orient
            if first_link.from_idx == start
            else _opp(first_link.to_orient)
        )
    else:
        start_orient = "+"

    path: _OrientedPath = [(start, start_orient == "+")]
    visited: set[int] = {start}

    while True:
        cur_idx, cur_fwd = path[-1]
        cur_orient = "+" if cur_fwd else "-"

        # Build candidate neighbours respecting bidirected edge semantics.
        # Prefer orientation-valid neighbours; fall back to any unvisited
        # neighbour if none are reachable (handles non-standard topologies).
        valid: dict[int, tuple[Link, str]] = {}
        fallback: dict[int, Link] = {}
        for nbr_idx, link in graph.adj(cur_idx).items():
            if nbr_idx in visited:
                continue
            nbr_orient = _next_orient(cur_idx, cur_orient, link)
            if nbr_orient is not None:
                valid[nbr_idx] = (link, nbr_orient)
            else:
                fallback[nbr_idx] = link

        if valid:
            chosen = pick(
                graph,
                list(valid.keys()),
                {k: v[0] for k, v in valid.items()},
            )
            _, chosen_orient = valid[chosen]
        elif fallback:
            chosen = pick(graph, list(fallback.keys()), fallback)
            chosen_orient = "+"
        else:
            break

        path.append((chosen, chosen_orient == "+"))
        visited.add(chosen)

    return path


def _iter_sampled_paths(
    graph: rx.PyGraph,
    n_samples: int,
    weight_mode: str = "kmer",
    eligible_nodes: set[int] | None = None,
) -> Iterator[tuple[_OrientedPath, int]]:
    """Yield ``(path, bp_length)`` for *n_samples* random simple paths.

    Walks start from leaf nodes (degree == 1).  If the graph has no
    leaves, all nodes are used as start candidates instead.

    When *eligible_nodes* is provided, only nodes in that set are used
    as start candidates.  Because components are disjoint, walks stay
    within the eligible components automatically.

    Args:
        graph: The assembled sequence graph.
        n_samples: Number of paths to sample.
        weight_mode: Neighbour selection weighting —
            ``"kmer"``, ``"overlap"``, or ``"unweighted"``.
        eligible_nodes: Optional set of node indices to restrict sampling
            to.  When ``None``, all nodes are eligible.

    Yields:
        ``(path, bp_length)`` where *path* is an oriented path
        (list of ``(node_idx, is_forward)`` tuples) and *bp_length*
        is the sum of their segment lengths.
    """
    leaves = leaf_nodes(graph)
    if eligible_nodes is not None:
        leaves = [n for n in leaves if n in eligible_nodes]
    if leaves:
        start_nodes = leaves
    else:
        fallback = (
            list(eligible_nodes)
            if eligible_nodes is not None
            else list(graph.node_indices())
        )
        logger.warning(
            "No eligible leaf nodes (all degrees > 1); "
            "using %d eligible nodes as start candidates.",
            len(fallback),
        )
        start_nodes = fallback

    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task("Sampling paths", total=n_samples)
        for _ in range(n_samples):
            path = _random_simple_path(graph, start_nodes, weight_mode)
            bp = sum(graph[idx].length for idx, _ in path)
            yield path, bp
            progress.advance(task)


def sample_path_lengths(
    graph: rx.PyGraph,
    n_samples: int,
    weight_mode: str = "kmer",
    eligible_nodes: set[int] | None = None,
) -> list[int]:
    """Sample random simple paths and return their bp lengths.

    Args:
        graph: The assembled sequence graph.
        n_samples: Number of paths to sample.
        weight_mode: Neighbour selection weighting —
            ``"kmer"``, ``"overlap"``, or ``"unweighted"``.
        eligible_nodes: Optional set of node indices to restrict sampling
            to.  When ``None``, all nodes are eligible.

    Returns:
        List of path lengths in base pairs.
    """
    return [
        bp for _, bp in _iter_sampled_paths(
            graph, n_samples, weight_mode, eligible_nodes,
        )
    ]


def sample_paths(
    graph: rx.PyGraph,
    n_samples: int,
    weight_mode: str = "kmer",
    eligible_nodes: set[int] | None = None,
) -> list[tuple[_OrientedPath, int]]:
    """Sample random simple paths and return them with their bp lengths.

    Like :func:`sample_path_lengths` but also returns the path node
    sequences needed for minimizer matching.

    Args:
        graph: The assembled sequence graph.
        n_samples: Number of paths to sample.
        weight_mode: Neighbour selection weighting —
            ``"kmer"``, ``"overlap"``, or ``"unweighted"``.
        eligible_nodes: Optional set of node indices to restrict sampling
            to.  When ``None``, all nodes are eligible.

    Returns:
        List of ``(path, bp_length)`` pairs.
    """
    return list(
        _iter_sampled_paths(graph, n_samples, weight_mode, eligible_nodes)
    )


# ------------------------------------------------------------------
# Paired-end insert size estimation
# ------------------------------------------------------------------

def _seg_name_index(graph: rx.PyGraph) -> dict[str, int]:
    """Return a mapping from segment name to graph node index.

    Args:
        graph: Parsed GFA graph.

    Returns:
        Dict mapping ``Segment.name`` to node index.
    """
    return {graph[idx].name: idx for idx in graph.node_indices()}


def _load_pe_pairs(bam_path: Path) -> dict[str, tuple[str, str]]:
    """Load paired-end read-to-segment mappings from a BAM file.

    Reads are grouped by template name.  Only primary, mapped
    alignments are considered.  Returns only templates where
    both R1 and R2 have a primary alignment.

    Args:
        bam_path: BAM file with reads aligned to assembly segments.
            The reference sequence names must match segment names
            in the GFA.

    Returns:
        Mapping of template name to ``(R1_segment, R2_segment)``.

    Raises:
        typer.BadParameter: If pysam is not installed.
    """
    if _pysam is None:
        raise typer.BadParameter(
            "pysam is required for --paired-end. "
            "Install it with: pip install pysam"
        )

    r1_seg: dict[str, str] = {}
    r2_seg: dict[str, str] = {}

    reads_seen = 0
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task("Loading PE pairs", total=None)
        with _pysam.AlignmentFile(str(bam_path), "rb") as bam:
            for read in bam.fetch(until_eof=True):
                if (
                    read.is_unmapped
                    or read.is_secondary
                    or read.is_supplementary
                    or not read.is_paired
                ):
                    continue
                seg = read.reference_name
                name = read.query_name
                if seg is None or name is None:
                    continue
                if read.is_read1:
                    r1_seg[name] = seg
                elif read.is_read2:
                    r2_seg[name] = seg
                reads_seen += 1
                if reads_seen % 100_000 == 0:
                    progress.update(task, completed=reads_seen)
        progress.update(task, completed=reads_seen)

    pairs = {
        name: (r1_seg[name], r2_seg[name])
        for name in r1_seg
        if name in r2_seg
    }
    logger.info(
        "Loaded %d read pairs from %s", len(pairs), bam_path,
    )
    return pairs


def _path_pair_distances(
    graph: rx.PyGraph,
    path: _OrientedPath,
    pair_seg_indices: list[tuple[int, int]],
) -> list[int]:
    """Return bp distances for read pairs found within a path.

    For each pair where both segments appear in *path*, the
    distance is the sum of segment lengths from the earlier
    segment to the later one (inclusive).  This approximates
    the DNA fragment length spanned by the pair.

    Pairs where both reads map to the same segment are included;
    their distance equals that segment's length.

    Args:
        graph: The assembled sequence graph.
        path: Oriented path as a list of ``(node_idx, is_forward)``
            tuples.
        pair_seg_indices: List of ``(R1_node_idx, R2_node_idx)``
            pairs to search for.

    Returns:
        List of bp distances, one per pair found in the path.
    """
    node_indices = [idx for idx, _ in path]
    node_pos = {node: i for i, node in enumerate(node_indices)}
    distances: list[int] = []

    for seg1_idx, seg2_idx in pair_seg_indices:
        if seg1_idx not in node_pos or seg2_idx not in node_pos:
            continue
        i = node_pos[seg1_idx]
        j = node_pos[seg2_idx]
        if i > j:
            i, j = j, i
        dist = sum(graph[node_indices[k]].length for k in range(i, j + 1))
        distances.append(dist)

    return distances


def sample_pair_distances(
    graph: rx.PyGraph,
    pair_seg_indices: list[tuple[int, int]],
    n_samples: int,
    weight_mode: str = "kmer",
) -> list[int]:
    """Sample random paths and collect paired-end bp distances.

    For each sampled path, finds all read pairs where both
    segments appear in the path and records the bp distance
    between them (sum of segment lengths, inclusive).  The
    distribution of these distances estimates the fragment
    length (insert size).

    Args:
        graph: The assembled sequence graph.
        pair_seg_indices: List of ``(R1_node_idx, R2_node_idx)``
            pairs, pre-converted from segment names.
        n_samples: Number of random paths to sample.
        weight_mode: Neighbour selection weighting —
            ``"kmer"``, ``"overlap"``, or ``"unweighted"``.

    Returns:
        List of bp distances across all sampled paths.
    """
    leaves = leaf_nodes(graph)
    if leaves:
        start_nodes = leaves
    else:
        logger.warning(
            "Graph has no leaf nodes; using all nodes as "
            "start candidates."
        )
        start_nodes = list(graph.node_indices())

    distances: list[int] = []

    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Sampling PE distances", total=n_samples,
        )
        for _ in range(n_samples):
            path = _random_simple_path(graph, start_nodes, weight_mode)
            distances.extend(
                _path_pair_distances(graph, path, pair_seg_indices)
            )
            progress.advance(task)

    return distances


def longest_simple_path(graph: rx.PyGraph) -> list[int]:
    """Return a longest simple path in the graph.

    Uses rustworkx's exact algorithm, which exhaustively
    searches all pairs of simple paths and is expensive for
    large graphs.

    Args:
        graph: The graph to search.

    Returns:
        Ordered list of node indices, or empty list if the
        graph is empty.
    """
    result = rx.longest_simple_path(graph)
    return list(result) if result is not None else []


# ------------------------------------------------------------------
# Summary output
# ------------------------------------------------------------------

def _compute_summary(
    graph: rx.PyGraph,
    path_lengths: list[int] | None,
    weight_mode: str = "kmer",
    pair_distances: list[int] | None = None,
) -> dict[str, object]:
    """Compute a summary of graph properties as a dict.

    Args:
        graph: Parsed GFA graph.
        path_lengths: Sampled path lengths in bp, or ``None``
            if path sampling was skipped.
        weight_mode: Weight mode used for path sampling.
        pair_distances: Estimated insert sizes in bp from
            paired-end analysis, or ``None`` if not performed.

    Returns:
        Dict of summary statistics suitable for JSON serialisation.
    """
    n_nodes = graph.num_nodes()
    n_edges = graph.num_edges()
    result: dict[str, object] = {
        "nodes": n_nodes,
        "edges": n_edges,
    }

    if n_nodes == 0:
        return result

    components = rx.connected_components(graph)
    component_sizes = sorted(
        (len(c) for c in components), reverse=True,
    )
    n_comp = len(component_sizes)
    comp_mean = sum(component_sizes) / n_comp
    comp_variance = sum((x - comp_mean) ** 2 for x in component_sizes) / n_comp
    mid = n_comp // 2
    comp_median: float = (
        component_sizes[mid]
        if n_comp % 2 == 1
        else (component_sizes[mid - 1] + component_sizes[mid]) / 2
    )
    result["connected_components"] = n_comp
    result["component_nodes_min"] = component_sizes[-1]
    result["component_nodes_max"] = component_sizes[0]
    result["component_nodes_mean"] = comp_mean
    result["component_nodes_median"] = comp_median
    result["component_nodes_std_dev"] = math.sqrt(comp_variance)

    segments: list[Segment] = [
        graph[idx] for idx in graph.node_indices()
    ]
    seg_lengths = [s.length for s in segments]
    total_bp = sum(seg_lengths)
    result["total_assembly_bp"] = total_bp
    result["segment_length_min"] = min(seg_lengths)
    result["segment_length_max"] = max(seg_lengths)
    result["segment_length_mean"] = total_bp / n_nodes

    degrees = [
        graph.degree(idx) for idx in graph.node_indices()
    ]
    result["degree_min"] = min(degrees)
    result["degree_max"] = max(degrees)
    result["degree_mean"] = sum(degrees) / n_nodes

    kmer_counts = [
        s.kmer_count for s in segments
        if s.kmer_count is not None
    ]
    if kmer_counts:
        result["kmer_count_min"] = min(kmer_counts)
        result["kmer_count_max"] = max(kmer_counts)
        result["kmer_count_mean"] = (
            sum(kmer_counts) / len(kmer_counts)
        )

    if path_lengths is not None:
        n = len(path_lengths)
        mean = sum(path_lengths) / n
        variance = (
            sum((x - mean) ** 2 for x in path_lengths) / n
        )
        result["path_lengths"] = {
            "samples": n,
            "weight_mode": weight_mode,
            "min": min(path_lengths),
            "max": max(path_lengths),
            "mean": mean,
            "std_dev": math.sqrt(variance),
            "variance": variance,
        }

    if pair_distances is not None:
        if not pair_distances:
            result["insert_sizes"] = None
        else:
            n = len(pair_distances)
            mean = sum(pair_distances) / n
            variance = (
                sum((x - mean) ** 2 for x in pair_distances) / n
            )
            result["insert_sizes"] = {
                "pairs": n,
                "min": min(pair_distances),
                "max": max(pair_distances),
                "mean": mean,
                "variance": variance,
            }

    return result


def print_summary(
    graph: rx.PyGraph,
    path_lengths: list[int] | None,
    weight_mode: str = "kmer",
    pair_distances: list[int] | None = None,
) -> None:
    """Print a summary of graph properties to stdout.

    Args:
        graph: Parsed GFA graph.
        path_lengths: Sampled path lengths in bp, or ``None``
            if path sampling was skipped.
        weight_mode: Weight mode used for path sampling —
            shown in the output header.
        pair_distances: Estimated insert sizes in bp from
            paired-end analysis, or ``None`` if not performed.
    """
    s = _compute_summary(graph, path_lengths, weight_mode, pair_distances)

    print(f"Nodes (segments): {s['nodes']}")
    print(f"Edges (links):    {s['edges']}")

    if s["nodes"] == 0:
        return

    print(
        f"Connected components: n={s['connected_components']}, "
        f"min={s['component_nodes_min']}, "
        f"max={s['component_nodes_max']}, "
        f"mean={s['component_nodes_mean']:.0f}, "  # type: ignore[str-format]
        f"median={s['component_nodes_median']:.0f}, "  # type: ignore[str-format]
        f"std_dev={s['component_nodes_std_dev']:.0f}"  # type: ignore[str-format]
    )

    print(f"Total assembly span: {s['total_assembly_bp']:,} bp")
    print(
        f"Segment lengths:  "
        f"min={s['segment_length_min']:,}, "
        f"max={s['segment_length_max']:,}, "
        f"mean={s['segment_length_mean']:,.0f}"  # type: ignore[str-format]
    )
    print(
        f"Node degrees:     "
        f"min={s['degree_min']}, "
        f"max={s['degree_max']}, "
        f"mean={s['degree_mean']:.1f}"  # type: ignore[str-format]
    )

    if "kmer_count_min" in s:
        print(
            f"K-mer counts:     "
            f"min={s['kmer_count_min']:,}, "
            f"max={s['kmer_count_max']:,}, "
            f"mean={s['kmer_count_mean']:,.0f}"  # type: ignore[str-format]
        )

    if "path_lengths" in s:
        pl = s["path_lengths"]
        assert isinstance(pl, dict)
        print(
            f"\nSampled path lengths "
            f"({pl['samples']:,} samples, "
            f"{pl['weight_mode']} weights, bp):"
        )
        print(f"  min:      {pl['min']:,}")
        print(f"  max:      {pl['max']:,}")
        print(f"  mean:     {pl['mean']:,.0f}")
        print(f"  std dev:  {pl['std_dev']:,.0f}")
        print(f"  variance: {pl['variance']:,.0f}")

    if "insert_sizes" in s:
        ins = s["insert_sizes"]
        if ins is None:
            print(
                "\nPaired-end insert size: "
                "no pairs found in sampled paths."
            )
        else:
            assert isinstance(ins, dict)
            print(
                f"\nEstimated insert sizes "
                f"({ins['pairs']:,} pairs, bp):"
            )
            print(f"  min:      {ins['min']:,}")
            print(f"  max:      {ins['max']:,}")
            print(f"  mean:     {ins['mean']:,.0f}")
            print(f"  variance: {ins['variance']:,.0f}")


def _report_chosen_components(
    graph: rx.PyGraph,
    chosen_comps: list[set[int]],
    seg_min_index: list[tuple[np.ndarray, np.ndarray] | None] | None = None,
) -> None:
    """Print a per-component size table (nodes, bp, optionally minimizers).

    Args:
        graph: Parsed GFA graph.
        chosen_comps: Components to display, ordered largest-first by bp.
        seg_min_index: When provided, minimizer counts per segment are
            summed and displayed as an additional column.
    """
    n_total = graph.num_nodes()
    for i, comp in enumerate(chosen_comps):
        if i == 10:
            typer.echo(
                f"  ... and {len(chosen_comps) - 10} more",
                err=True,
            )
            break
        comp_bp = sum(graph[idx].length for idx in comp)
        line = (
            f"  [{i + 1}] {len(comp):,} nodes, "
            f"{comp_bp:,} bp "
            f"= {len(comp) / n_total:.1%} of total"
        )
        if seg_min_index is not None:
            def _smi_len(entry: tuple[np.ndarray, np.ndarray] | None) -> int:
                return len(entry[0]) if entry is not None else 0
            comp_min = sum(
                _smi_len(seg_min_index[idx])
                for idx in comp
                if idx < len(seg_min_index)
            )
            line += f", {comp_min:,} minimizers"
        typer.echo(line, err=True)


# ------------------------------------------------------------------
# Fragment length / insert size estimation via combination minimizers
# ------------------------------------------------------------------

@dataclass
class FragmentLengthEstimate:
    """Posterior estimate of the log-normal fragment length distribution.

    Attributes:
        mu_log: Posterior mean of the log-scale mean parameter.
        sigma_log: Posterior mean of the log-scale std-dev parameter.
        mu_log_ci: 95 % credible interval for ``mu_log``.
        sigma_log_ci: 95 % credible interval for ``sigma_log``.
        median: Estimated median fragment length = ``exp(mu_log)`` bp.
        mean: Estimated mean fragment length = ``exp(mu_log + sigma_log**2 / 2)`` bp.
        n_bins_used: Number of distance bins that passed the minimum-hash filter.
        inference: Inference algorithm used (``"nuts"`` or ``"advi"``).
    """

    mu_log: float
    sigma_log: float
    mu_log_ci: tuple[float, float]
    sigma_log_ci: tuple[float, float]
    median: float
    mean: float
    n_bins_used: int
    inference: str
    signal_reliable: bool = True  # False when containment rates are nearly flat


def _load_node_minimizer_seqs(
    seqs_prefix: Path,
) -> tuple[dict[int, np.ndarray], int]:
    """Load per-node minimizer ID arrays from rust-mdbg LZ4 sequences files.

    The sequences files written by rust-mdbg (``{prefix}.{thread}.sequences``)
    are LZ4-frame-compressed tab-separated text.  Each data row has the form::

        node_name\\t[h1, h2, ..., hk]\\tsequence\\t...

    Args:
        seqs_prefix: rust-mdbg output prefix; files are matched as
            ``{prefix.parent}/{prefix.name}.*.sequences``.

    Returns:
        ``(node_name_int → minimizer_id_array, k_value)`` where ``k_value``
        is read from the file header.  Returns ``({}, 1)`` if no files are
        found or the ``lz4`` package is unavailable.
    """
    if _lz4 is None:
        logger.warning(
            "lz4 package not available; cannot load sequences files. "
            "Install with: pip install lz4"
        )
        return {}, 1

    seqs_files = sorted(
        seqs_prefix.parent.glob(f"{seqs_prefix.name}.*.sequences")
    )
    if not seqs_files:
        logger.warning(
            "No sequences files found matching %s.*.sequences", seqs_prefix
        )
        return {}, 1

    node_seqs: dict[int, np.ndarray] = {}
    k_val = 1

    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task("Loading sequences files", total=len(seqs_files))
        for seqs_file in seqs_files:
            try:
                with _lz4.open(str(seqs_file), mode="rt") as fh:
                    for line in fh:
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        if line.startswith("# k ="):
                            try:
                                k_val = int(line.split("=", 1)[1].strip())
                            except ValueError:
                                pass
                            continue
                        if line.startswith("#"):
                            continue
                        fields = line.split("\t")
                        if len(fields) < 2:
                            continue
                        try:
                            node_name = int(fields[0])
                        except ValueError:
                            continue
                        min_str = fields[1].strip()
                        if min_str.startswith("[") and min_str.endswith("]"):
                            min_str = min_str[1:-1]
                            try:
                                mids = np.array(
                                    [
                                        int(x.strip())
                                        for x in min_str.split(",")
                                        if x.strip()
                                    ],
                                    dtype=np.uint64,
                                )
                            except ValueError:
                                continue
                            node_seqs[node_name] = mids
            except Exception as exc:
                logger.warning(
                    "Failed to read sequences file %s: %s", seqs_file, exc
                )
            progress.advance(task)

    logger.info(
        "Loaded minimizer sequences for %d nodes (k=%d) from %d file(s)",
        len(node_seqs), k_val, len(seqs_files),
    )
    return node_seqs, k_val


def _load_minimizer_table(table_path: Path) -> dict[int, str]:
    """Load the rust-mdbg minimizer table TSV into a ``hash_id → l-mer`` dict.

    Lines starting with ``#`` are treated as comments.  Each data row must
    have at least two tab-separated fields: ``minimizer_id`` (integer) and
    ``lmer`` (DNA string).

    Args:
        table_path: Path to the ``{prefix}.minimizer_table`` TSV file.

    Returns:
        Dict mapping integer minimizer hash ID to l-mer string.
    """
    table: dict[int, str] = {}
    entries = 0
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task("Loading minimizer table", total=None)
        with table_path.open() as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                fields = line.split("\t")
                if len(fields) < 2:
                    continue
                try:
                    table[int(fields[0])] = fields[1]
                    entries += 1
                    if entries % 500_000 == 0:
                        progress.update(task, completed=entries)
                except ValueError:
                    continue
        progress.update(task, completed=entries)
    logger.info("Loaded %d entries from minimizer table %s", len(table), table_path)
    return table


def aggregate_path_bin_sketches(
    path_sketches: list[list[CombSketchIndex]],
    max_pool_hashes: int = 500_000,
) -> list[set[int]]:
    """Union per-path per-bin sketches into one set per bin.

    For each distance bin, the hash sets from all paths are unioned.  If the
    resulting set exceeds *max_pool_hashes*, it is uniformly subsampled.

    Args:
        path_sketches: ``list[list[CombSketchIndex]]`` — one inner list per
            path, one ``CombSketchIndex`` per distance bin.
        max_pool_hashes: Saturation cap per bin after pooling.

    Returns:
        ``list[set[int]]`` — one set of hash values per bin.
    """
    if not path_sketches:
        return []
    n_bins = len(path_sketches[0])
    pooled: list[set[int]] = [set() for _ in range(n_bins)]

    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task("Pooling path sketches", total=len(path_sketches))
        for sketch_list in path_sketches:
            for b, sketch in enumerate(sketch_list):
                if b >= n_bins:
                    break
                # Path sketches are always in-memory (no LMDB).
                pooled[b].update(sketch._buf.keys())
            progress.advance(task)

    for b in range(n_bins):
        if len(pooled[b]) > max_pool_hashes:
            pooled[b] = set(random.sample(list(pooled[b]), max_pool_hashes))

    return pooled


def compute_containment_rates(
    pe_sketch: CombSketchIndex,
    path_bin_sets: list[set[int]],
    intra_sketch: CombSketchIndex | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Compute per-bin containment of path hashes in the PE sketch.

    For each distance bin *b*::

        c_obs[b]   = |pe_sketch ∩ path_bin_sets[b]| / |path_bin_sets[b]|
        c_noise[b] = |intra_sketch ∩ path_bin_sets[b]| / |path_bin_sets[b]|

    Bins with an empty path set are assigned ``nan``.

    Args:
        pe_sketch: Cross-pair (R1×R2) combination minimizer sketch.
        path_bin_sets: Pooled path hash sets, one per distance bin.
        intra_sketch: Within-read combination minimizer sketch used as a
            noise baseline.  Pass ``None`` to skip noise estimation.

    Returns:
        ``(c_obs, c_noise, n_pe_hashes)`` where the arrays have length
        equal to ``len(path_bin_sets)``.
    """
    def _sketch_keys(sketch: CombSketchIndex) -> set[int]:
        if sketch._env is None:
            return set(sketch._buf.keys())
        sketch._flush()
        keys: set[int] = set()
        with sketch._env.begin() as txn:
            for key, _ in txn.cursor(db=sketch._lmdb_db):
                keys.add(struct.unpack("<Q", key)[0])
        return keys

    pe_keys = _sketch_keys(pe_sketch)
    n_pe = len(pe_keys)
    intra_keys: set[int] | None = (
        _sketch_keys(intra_sketch) if intra_sketch is not None else None
    )

    n_bins = len(path_bin_sets)
    c_obs = np.full(n_bins, np.nan)
    c_noise = np.zeros(n_bins)

    for b, path_set in enumerate(path_bin_sets):
        if not path_set:
            continue
        n_path = len(path_set)
        c_obs[b] = len(pe_keys & path_set) / n_path
        if intra_keys is not None:
            c_noise[b] = len(intra_keys & path_set) / n_path

    return c_obs, c_noise, n_pe


def _fragment_length_pyro_model(
    c_adjusted: "_torch.Tensor",
    bin_lo: "_torch.Tensor",
    bin_hi: "_torch.Tensor",
    n_path_hashes: "_torch.Tensor",
    observed_mask: "_torch.Tensor",
) -> None:
    """Pyro generative model for fragment length from containment rates.

    Priors::

        mu_log    ~ Normal(log(8000), 1.0)
        sigma_log ~ HalfNormal(0.5)
        rho       ~ Beta(2, 20)          # residual noise floor

    Likelihood per active bin *b*::

        p[b]        = LogNormal(mu_log, sigma_log).CDF(bin_hi[b])
                    - LogNormal(mu_log, sigma_log).CDF(bin_lo[b])
        expected[b] = p[b] * (1 - rho) + rho
        sigma_obs[b]= sqrt(expected[b] * (1 - expected[b]) / n_path_hashes[b])
        obs[b]      ~ Normal(expected[b], sigma_obs[b])    [masked by observed_mask]

    Args:
        c_adjusted: Noise-subtracted containment rates, shape ``[B]``.
        bin_lo: Lower bp edge of each bin, shape ``[B]``.
        bin_hi: Upper bp edge of each bin, shape ``[B]``.
        n_path_hashes: Number of path hashes per bin, shape ``[B]``.
        observed_mask: Boolean mask — ``True`` means the bin is active, shape ``[B]``.
    """
    mu_log = _pyro.sample(
        "mu_log",
        _pyro_dist.Normal(
            _torch.tensor(math.log(8000.0)), _torch.tensor(1.0)
        ),
    )
    sigma_log = _pyro.sample(
        "sigma_log", _pyro_dist.HalfNormal(_torch.tensor(0.5))
    )
    rho = _pyro.sample(
        "rho", _pyro_dist.Beta(_torch.tensor(2.0), _torch.tensor(20.0))
    )

    # Log-normal CDF: P(X <= x) = Phi((log(x) - mu) / sigma).
    normal = _torch.distributions.Normal(mu_log, sigma_log)
    log_lo = _torch.log(bin_lo.clamp(min=1.0))
    log_hi = _torch.log(bin_hi.clamp(min=1.0))
    p = (normal.cdf(log_hi) - normal.cdf(log_lo)).clamp(0.0, 1.0)

    expected = p * (1.0 - rho) + rho
    # Clamp n_path_hashes to ≥ 1 to avoid division by zero for masked bins
    # (the mask prevents those bins from contributing to the gradient).
    sigma_obs = (
        expected * (1.0 - expected) / n_path_hashes.clamp(min=1.0)
    ).sqrt().clamp(min=1e-6)

    B = c_adjusted.shape[0]
    with _pyro.plate("bins", B, dim=-1):
        with _pyro.poutine.mask(mask=observed_mask):
            _pyro.sample(
                "obs", _pyro_dist.Normal(expected, sigma_obs), obs=c_adjusted
            )


def estimate_fragment_length(
    pe_sketch: CombSketchIndex,
    path_bin_sketches: list[list[CombSketchIndex]],
    bin_distances: list[float],
    intra_sketch: CombSketchIndex | None = None,
    read_length: int = 150,
    min_path_hashes_per_bin: int = 50,
    inference: str = "nuts",
    num_samples: int = 500,
    num_warmup: int = 200,
    max_pool_hashes: int = 500_000,
) -> FragmentLengthEstimate:
    """Estimate the fragment length distribution from combination minimizer containment.

    Pools path distance-bin sketches across paths, computes containment rates
    relative to the PE sketch, subtracts intra-read noise, and fits a log-normal
    distribution via Bayesian inference (NUTS or ADVI).

    Args:
        pe_sketch: Cross-pair (R1×R2) combination minimizer sketch.
        path_bin_sketches: One inner list per path; each inner list has one
            :class:`CombSketchIndex` per distance bin.
        bin_distances: Sorted bp breakpoints defining bins
            (``len(bin_distances) - 1`` bins total).
        intra_sketch: Within-read sketch for noise baseline.  ``None`` → skip.
        read_length: Sequenced read length in bp (informational only).
        min_path_hashes_per_bin: Bins with fewer path hashes are excluded.
        inference: ``"nuts"`` (default) or ``"advi"``.
        num_samples: Posterior samples to draw (NUTS) or predictive samples
            (ADVI).
        num_warmup: NUTS warm-up steps.
        max_pool_hashes: Saturation cap for pooled path bin sets.

    Returns:
        :class:`FragmentLengthEstimate` with posterior statistics.

    Raises:
        RuntimeError: If ``pyro-ppl`` or ``torch`` are not installed.
        ValueError: If *inference* is not ``"nuts"`` or ``"advi"``.
    """
    if _pyro is None or _torch is None:
        raise RuntimeError(
            "pyro-ppl and torch are required for insert size estimation; "
            "install with: pip install pyro-ppl torch"
        )

    # 1. Pool path bin sketches.
    path_bin_sets = aggregate_path_bin_sketches(path_bin_sketches, max_pool_hashes)

    # 2. Compute containment rates.
    c_obs, c_noise, n_pe = compute_containment_rates(
        pe_sketch, path_bin_sets, intra_sketch
    )
    logger.info(
        "PE sketch: %d unique hashes; containment rates per bin: %s",
        n_pe,
        np.round(c_obs, 4),
    )

    # 3. Build bin-edge arrays.
    n_bins = len(bin_distances) - 1
    bin_lo = np.array(bin_distances[:-1], dtype=np.float32)
    bin_hi = np.array(bin_distances[1:], dtype=np.float32)
    n_path_hashes = np.array([len(s) for s in path_bin_sets], dtype=np.float32)

    # 4. Adjust for noise; replace NaN (empty bins) with 0.
    c_adjusted = np.clip(np.where(np.isnan(c_obs), 0.0, c_obs) - c_noise, 0.0, 1.0)

    # 5. Build mask.
    observed_mask = n_path_hashes >= min_path_hashes_per_bin
    n_bins_used = int(observed_mask.sum())

    # 5b. Flat-containment check: if the containment rates across active bins
    # have very low variance, the signal is dominated by repeat-induced noise.
    # This can occur with low-k graphs, tandem-repeat-rich genomes, or when the
    # true insert size is outside the bin range.
    active_c = c_adjusted[observed_mask] if n_bins_used > 1 else np.array([])
    signal_reliable = True
    if len(active_c) > 1:
        c_range = float(active_c.max() - active_c.min())
        if c_range < 0.10:
            signal_reliable = False
            logger.warning(
                "Containment rates are nearly flat across active bins "
                "(range=%.3f, threshold=0.10). The insert-size signal is weak — "
                "the estimate will be dominated by the prior. "
                "Possible causes: repeat-rich genome, low k value in the mdBG, "
                "or insert size outside the specified bin range. "
                "Consider increasing k or widening --insert-size-bins.",
                c_range,
            )

    if n_bins_used == 0:
        logger.warning(
            "No bins have >= %d path hashes; returning uninformative estimate",
            min_path_hashes_per_bin,
        )
        mu0 = math.log(8000.0)
        return FragmentLengthEstimate(
            mu_log=mu0,
            sigma_log=0.5,
            mu_log_ci=(mu0 - 1.0, mu0 + 1.0),
            sigma_log_ci=(0.0, 1.0),
            median=math.exp(mu0),
            mean=math.exp(mu0 + 0.5**2 / 2.0),
            n_bins_used=0,
            inference=inference,
            signal_reliable=False,
        )

    # 6. Convert to tensors.
    c_adj_t = _torch.tensor(c_adjusted, dtype=_torch.float32)
    bin_lo_t = _torch.tensor(bin_lo, dtype=_torch.float32)
    bin_hi_t = _torch.tensor(bin_hi, dtype=_torch.float32)
    n_path_t = _torch.tensor(n_path_hashes, dtype=_torch.float32)
    mask_t = _torch.tensor(observed_mask, dtype=_torch.bool)
    model_args = (c_adj_t, bin_lo_t, bin_hi_t, n_path_t, mask_t)

    # 7. Run inference.
    if inference == "nuts":
        kernel = NUTS(_fragment_length_pyro_model)
        mcmc = MCMC(
            kernel,
            num_samples=num_samples,
            warmup_steps=num_warmup,
            disable_progbar=False,
        )
        mcmc.run(*model_args)
        samples = mcmc.get_samples()
        mu_log_samps = samples["mu_log"].numpy()
        sigma_log_samps = samples["sigma_log"].numpy()

    elif inference == "advi":
        _pyro.clear_param_store()
        guide = AutoNormal(_fragment_length_pyro_model)
        optimizer = _PyroAdam({"lr": 0.01})
        svi_obj = SVI(
            _fragment_length_pyro_model, guide, optimizer, loss=Trace_ELBO()
        )
        _advi_steps = 2000
        with Progress(*_PROGRESS_COLUMNS) as progress:
            task = progress.add_task("ADVI inference", total=_advi_steps)
            for step in range(_advi_steps):
                loss = svi_obj.step(*model_args)
                if step % 500 == 0:
                    logger.debug("ADVI step %d, ELBO=%.4f", step, -loss)
                progress.advance(task)
        predictive = Predictive(
            _fragment_length_pyro_model,
            guide=guide,
            num_samples=num_samples,
            return_sites=["mu_log", "sigma_log"],
        )
        post = predictive(*model_args)
        mu_log_samps = post["mu_log"].squeeze().numpy()
        sigma_log_samps = post["sigma_log"].squeeze().numpy()

    else:
        raise ValueError(
            f"Unknown inference method {inference!r}; use 'nuts' or 'advi'"
        )

    # 8. Summarise posterior.
    mu_log_mean = float(np.mean(mu_log_samps))
    sigma_log_mean = float(np.mean(sigma_log_samps))
    mu_log_ci = (
        float(np.percentile(mu_log_samps, 2.5)),
        float(np.percentile(mu_log_samps, 97.5)),
    )
    sigma_log_ci = (
        float(np.percentile(sigma_log_samps, 2.5)),
        float(np.percentile(sigma_log_samps, 97.5)),
    )

    return FragmentLengthEstimate(
        mu_log=mu_log_mean,
        sigma_log=sigma_log_mean,
        mu_log_ci=mu_log_ci,
        sigma_log_ci=sigma_log_ci,
        median=math.exp(mu_log_mean),
        mean=math.exp(mu_log_mean + sigma_log_mean**2 / 2.0),
        n_bins_used=n_bins_used,
        inference=inference,
        signal_reliable=signal_reliable,
    )


def build_top_path_combo_sketches(
    graph: rx.PyGraph,
    lmdb_path: Path | None,
    bin_distances: list[float],
    density: float,
    n_paths: int = 50,
    top_paths: list[tuple[_OrientedPath, int]] | None = None,
    k: int = 7,
    use_exact_distances: bool = False,
    min_table: dict[int, str] | None = None,
    l: int = 12,
    bp_scale: float | None = None,
    seqs_prefix: Path | None = None,
) -> tuple[list[list[CombSketchIndex]], float]:
    """Build distance-binned combination minimizer sketches for the top paths.

    For each path in *top_paths* (up to *n_paths*), the minimizer sequence is
    reconstructed from the rust-mdbg LZ4 sequences files (if *seqs_prefix* is
    given) and passed to :func:`build_path_combo_sketch`.  The resulting per-bin
    sketches are collected and returned alongside the ``bp_scale`` used.

    **Minimizer sequence construction** (when *seqs_prefix* is provided):

    Each GFA segment corresponds to a minimizer-space k-mer of length ``k``.
    For a forward-traversed segment the last minimizer is the new element;
    for a backward-traversed segment the first element of the reversed array is
    new.  The first segment in the path contributes all ``k`` minimizers; each
    subsequent segment contributes one new minimizer.

    If no sequences files are available, node graph indices are used as proxy
    minimizer IDs (combinations will not intersect with the PE sketch — use
    only for testing purposes).

    Args:
        graph: Parsed GFA graph (``Segment`` node data required).
        lmdb_path: rust-mdbg reads LMDB directory (used for bp_scale
            estimation only; thinning threshold is fixed at
            ``int(density × _UINT64_MAX)``).
        bin_distances: Sorted bp breakpoints (``≥ 2`` values).
        density: Combination hash thinning density.
        n_paths: Maximum number of paths to process.
        top_paths: Pre-selected ``(path, bp_length)`` pairs, sorted
            longest-first.  Pass ``None`` to return empty results.
        use_exact_distances: Forward to :func:`build_path_combo_sketch`.
        min_table: Minimizer hash → l-mer string mapping for exact distances.
        l: l-mer length; forwarded to :func:`build_path_combo_sketch`.
        bp_scale: Average bp per minimizer position.  Estimated from path
            data when ``None``.
        seqs_prefix: rust-mdbg output prefix for loading LZ4 sequences files.

    Returns:
        ``(list_of_per_path_sketch_lists, bp_scale_used)`` where each inner
        list has one :class:`CombSketchIndex` per distance bin.
    """
    if not top_paths:
        logger.warning("No top paths provided; skipping path combo sketch building")
        return [], 1.0

    # --- Load node minimizer sequences ------------------------------------------
    node_min_seqs: dict[int, np.ndarray] = {}
    k_val = 1
    if seqs_prefix is not None:
        node_min_seqs, k_val = _load_node_minimizer_seqs(seqs_prefix)
        if not node_min_seqs:
            logger.warning(
                "Sequences files empty or unreadable; "
                "falling back to node-index proxy minimizer IDs"
            )
    else:
        logger.warning(
            "No sequences prefix provided; "
            "using node graph indices as proxy minimizer IDs. "
            "Containment rates will not be meaningful."
        )

    selected = top_paths[:n_paths]

    # --- Estimate bp_scale if not provided --------------------------------------
    if bp_scale is None:
        total_bp = 0
        total_min = 0
        for path, bp_len in selected:
            total_bp += bp_len
            if node_min_seqs:
                for seg_i, (node_idx, _is_fwd) in enumerate(path):
                    mids = node_min_seqs.get(int(graph[node_idx].name))
                    if mids is not None:
                        if seg_i == 0:
                            total_min += len(mids)
                        else:
                            total_min += max(1, len(mids) - (k_val - 1))
                    else:
                        total_min += 1
            else:
                total_min += len(path)
        bp_scale = total_bp / max(total_min, 1)
        logger.info(
            "Estimated bp_scale=%.2f from %d top path(s)", bp_scale, len(selected)
        )

    # --- Build sketches per path ------------------------------------------------
    path_sketch_lists: list[list[CombSketchIndex]] = []
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Building path combo sketches", total=len(selected)
        )
        for path, _bp_len in selected:
            # Build path_min_seq.
            if node_min_seqs:
                min_ids: list[int] = []
                valid = True
                for seg_i, (node_idx, is_fwd) in enumerate(path):
                    mids = node_min_seqs.get(int(graph[node_idx].name))
                    if mids is None or len(mids) == 0:
                        valid = False
                        break
                    # Reversed segment: flip the minimizer order.
                    node_mids: np.ndarray = mids if is_fwd else mids[::-1]
                    if seg_i == 0:
                        min_ids.extend(node_mids.tolist())
                    else:
                        # Each subsequent node adds one new minimizer beyond
                        # the k-1 shared overlap with the previous node.
                        overlap = k_val - 1
                        new_mids = node_mids[overlap:] if len(node_mids) > overlap else node_mids[-1:]
                        min_ids.extend(new_mids.tolist())
                if not valid or len(min_ids) < 2:
                    progress.advance(task)
                    continue
                path_min_seq = np.array(min_ids, dtype=np.uint64)
            else:
                path_min_seq = np.array(
                    [node_idx for node_idx, _ in path], dtype=np.uint64
                )
                if len(path_min_seq) < 2:
                    progress.advance(task)
                    continue

            sketches = build_path_combo_sketch(
                path_min_seq=path_min_seq,
                bin_distances=bin_distances,
                density=density,
                bp_scale=bp_scale,
                k=k,
                use_exact_distances=use_exact_distances,
                path=path if use_exact_distances else None,
                graph=graph if use_exact_distances else None,
                min_table=min_table if use_exact_distances else None,
                l=l,
            )
            path_sketch_lists.append(sketches)
            progress.advance(task)

    logger.info(
        "Built path combo sketches for %d/%d paths",
        len(path_sketch_lists), len(selected),
    )
    return path_sketch_lists, bp_scale


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

@app.command()
def main(
    gfa: Annotated[Path, typer.Argument(
        help="Path to the GFA file",
        exists=True,
        dir_okay=False,
    )],
    samples: Annotated[int, typer.Option(
        "-n", "--samples",
        help="Number of random paths to sample",
    )] = 1000,
    no_sample: Annotated[bool, typer.Option(
        "--no-sample",
        help="Skip path sampling entirely",
    )] = False,
    weight: Annotated[WeightMode, typer.Option(
        help="Neighbour selection weighting "
        "for random walks",
        case_sensitive=False,
    )] = WeightMode.kmer,
    paired_end: Annotated[bool, typer.Option(
        "--paired-end/--no-paired-end",
        help="Estimate insert sizes from "
        "paired-end read alignments",
    )] = False,
    pe_bam: Annotated[Path | None, typer.Option(
        help="BAM file of reads aligned to assembly "
        "segments (required with --paired-end). "
        "Reference names must match GFA segment names",
        exists=True,
        dir_okay=False,
    )] = None,
    debug: Annotated[bool, typer.Option(
        "--debug/--no-debug",
        help="Enable debug logging",
    )] = False,
    json_out: Annotated[Path | None, typer.Option(
        "--json",
        help="Write summary statistics as JSON to this path",
        dir_okay=False,
    )] = None,
    sample_component_proportion: Annotated[float, typer.Option(
        "--sample-component-proportion",
        help="Sample paths only from the largest connected components "
             "whose combined node count covers at least this fraction "
             "of all graph nodes (0.0–1.0). Set to 0 (default) to "
             "sample from all components.",
    )] = 0.0,
    read_minimizers_prefix: Annotated[Path | None, typer.Option(
        "--read-minimizers",
        help="rust-mdbg output prefix for the reads LMDB "
             "({prefix}.index.lmdb); enables combination sketch building",
        dir_okay=False,
    )] = None,
    interleaved_pairs: Annotated[bool, typer.Option(
        "--interleaved-pairs/--no-interleaved-pairs",
        help="Reads in the LMDB are interleaved paired-end "
             "(arithmetic pairing: read 0↔1, 2↔3, …). "
             "Required to build the PE combination sketch.",
    )] = False,
    combo_k: Annotated[int, typer.Option(
        "--combo-k",
        help="Number of consecutive minimizer IDs per combination k-mer. "
             "Set to match the rust-mdbg -k value so each k-mer maps to one "
             "assembly graph node.",
        min=1,
    )] = 7,
    combo_density: Annotated[float, typer.Option(
        "--combo-density",
        help="Thinning density: fraction of combination hashes to retain",
        min=0.0,
        max=1.0,
    )] = 0.001,
    pe_combo_lmdb_out: Annotated[Path | None, typer.Option(
        "--pe-combo-lmdb-out",
        help="Persist the PE combination sketch to this LMDB directory",
        file_okay=False,
    )] = None,
    intra_combo_lmdb_out: Annotated[Path | None, typer.Option(
        "--intra-combo-lmdb-out",
        help="Persist the intra-read combination sketch to this LMDB directory",
        file_okay=False,
    )] = None,
    minimizer_table: Annotated[Path | None, typer.Option(
        "--minimizer-table",
        help="rust-mdbg minimizer table TSV ({prefix}.minimizer_table); "
             "enables exact bp-distance mode for path combo sketches",
        exists=True,
        dir_okay=False,
    )] = None,
    estimate_insert_size: Annotated[bool, typer.Option(
        "--estimate-insert-size/--no-estimate-insert-size",
        help="Estimate the fragment length distribution via combination "
             "minimizer containment (requires --read-minimizers and "
             "--interleaved-pairs)",
    )] = False,
    insert_size_bins: Annotated[str, typer.Option(
        "--insert-size-bins",
        help="Comma-separated bp bin edges for insert size estimation",
    )] = "150,200,250,300,400,500,600,800,1000",
    insert_size_paths: Annotated[int, typer.Option(
        "--insert-size-paths",
        help="Number of top (longest) paths to use for path combo sketches",
        min=1,
    )] = 50,
    read_length: Annotated[int, typer.Option(
        "--read-length",
        help="Sequenced read length in bp (used in insert size prior)",
        min=1,
    )] = 150,
    insert_size_inference: Annotated[str, typer.Option(
        "--insert-size-inference",
        help="Inference algorithm for insert size estimation: 'nuts' or 'advi'",
    )] = "nuts",
    insert_size_min_bin_hashes: Annotated[int, typer.Option(
        "--insert-size-min-bin-hashes",
        help="Minimum number of path hashes per bin to include in fitting",
        min=1,
    )] = 50,
) -> None:
    """Parse a rust-mdbg GFA output into a graph and report properties."""
    if paired_end and pe_bam is None:
        raise typer.BadParameter(
            "--pe-bam PATH is required when "
            "--paired-end is set."
        )

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    graph = parse_gfa(gfa)

    weight_str = weight.value

    path_lengths: list[int] | None = None
    sampled_paths: list[tuple[_OrientedPath, int]] | None = None
    chosen_comps: list[set[int]] = []

    if no_sample:
        logger.debug("Path sampling skipped by --no-sample.")
    elif graph.num_nodes() == 0:
        typer.echo(
            "Graph has no nodes; skipping path sampling.",
            err=True,
        )
    else:
        _en, chosen_comps = large_component_nodes(
            graph, sample_component_proportion,
        )
        eligible_nodes: set[int] | None = _en if sample_component_proportion > 0.0 else None
        n_total = graph.num_nodes()
        n_chosen_nodes = sum(len(c) for c in chosen_comps)
        _prop_label = (
            f" (>= {sample_component_proportion:.0%} threshold)"
            if sample_component_proportion > 0.0 else ""
        )
        typer.echo(
            f"Sampling from {len(chosen_comps)} component(s), "
            f"{n_chosen_nodes:,} nodes "
            f"= {n_chosen_nodes / n_total:.1%} of {n_total:,} total"
            f"{_prop_label}"
            + ":",
            err=True,
        )
        _report_chosen_components(graph, chosen_comps)
        if estimate_insert_size:
            # Preserve full paths (needed for path combo sketches).
            sampled_paths = sample_paths(graph, samples, weight_str, eligible_nodes)
            path_lengths = [bp for _, bp in sampled_paths]
        else:
            sampled_paths = None
            path_lengths = sample_path_lengths(graph, samples, weight_str, eligible_nodes)

    pair_distances: list[int] | None = None
    if paired_end and pe_bam is not None:
        pairs = _load_pe_pairs(pe_bam)
        seg_idx = _seg_name_index(graph)
        pair_seg_indices = [
            (seg_idx[r1], seg_idx[r2])
            for r1, r2 in pairs.values()
            if r1 in seg_idx and r2 in seg_idx
        ]
        if not pair_seg_indices:
            typer.echo(
                "No read pairs map to known "
                "graph segments.",
                err=True,
            )
            pair_distances = []
        else:
            logger.info(
                "%d of %d pairs map to graph segments",
                len(pair_seg_indices), len(pairs),
            )
            pair_distances = sample_pair_distances(
                graph, pair_seg_indices,
                samples, weight_str,
            )

    print_summary(
        graph, path_lengths, weight_str, pair_distances,
    )

    # --- Combination minimizer sketches ------------------------------------
    pe_combo_unique: int | None = None
    intra_combo_unique: int | None = None
    pe_sketch_obj: CombSketchIndex | None = None
    intra_sketch_obj: CombSketchIndex | None = None
    _lmdb_path: Path | None = None

    if read_minimizers_prefix is not None:
        _lmdb_path = Path(str(read_minimizers_prefix) + ".index.lmdb")

        if interleaved_pairs:
            pe_sketch_obj = build_pe_combo_sketch(
                _lmdb_path,
                density=combo_density,
                k=combo_k,
                out_lmdb=pe_combo_lmdb_out,
            )
            pe_combo_unique = len(pe_sketch_obj)
            typer.echo(
                f"PE combo sketch: {pe_combo_unique:,} unique hashes "
                f"(density={combo_density})",
                err=True,
            )

        intra_sketch_obj = build_intra_combo_sketch(
            _lmdb_path,
            density=combo_density,
            k=combo_k,
            out_lmdb=intra_combo_lmdb_out,
        )
        intra_combo_unique = len(intra_sketch_obj)
        typer.echo(
            f"Intra combo sketch: {intra_combo_unique:,} unique hashes "
            f"(density={combo_density})",
            err=True,
        )

    # --- Insert size estimation via combination minimizer containment -------
    insert_size_result: FragmentLengthEstimate | None = None

    if estimate_insert_size:
        if pe_sketch_obj is None:
            typer.echo(
                "ERROR: --estimate-insert-size requires --read-minimizers and "
                "--interleaved-pairs to build the PE combo sketch.",
                err=True,
            )
        elif sampled_paths is None:
            typer.echo(
                "WARNING: No sampled paths available for insert size estimation "
                "(path sampling was skipped).",
                err=True,
            )
        else:
            # Parse bin edges.
            try:
                _bin_edges = sorted(
                    float(x.strip()) for x in insert_size_bins.split(",")
                )
            except ValueError as exc:
                raise typer.BadParameter(
                    f"--insert-size-bins must be comma-separated numbers: {exc}"
                ) from exc

            # Select top paths by bp length.
            top_paths_sorted = sorted(sampled_paths, key=lambda t: t[1], reverse=True)

            # Load optional minimizer table.
            _min_table: dict[int, str] | None = None
            if minimizer_table is not None:
                _min_table = _load_minimizer_table(minimizer_table)

            typer.echo(
                f"Building path combo sketches for top "
                f"{insert_size_paths} paths "
                f"(bins: {_bin_edges})…",
                err=True,
            )
            path_bin_sketches, _bp_scale = build_top_path_combo_sketches(
                graph=graph,
                lmdb_path=_lmdb_path,
                bin_distances=_bin_edges,
                density=combo_density,
                n_paths=insert_size_paths,
                top_paths=top_paths_sorted,
                k=combo_k,
                min_table=_min_table,
                l=read_length,
                seqs_prefix=read_minimizers_prefix,
            )

            if path_bin_sketches:
                typer.echo(
                    f"Estimating fragment length "
                    f"(inference={insert_size_inference})…",
                    err=True,
                )
                insert_size_result = estimate_fragment_length(
                    pe_sketch=pe_sketch_obj,
                    path_bin_sketches=path_bin_sketches,
                    bin_distances=_bin_edges,
                    intra_sketch=intra_sketch_obj,
                    read_length=read_length,
                    min_path_hashes_per_bin=insert_size_min_bin_hashes,
                    inference=insert_size_inference,
                )
                _reliable_tag = (
                    "" if insert_size_result.signal_reliable else " [UNRELIABLE]"
                )
                typer.echo(
                    f"Fragment length estimate{_reliable_tag}: "
                    f"median={insert_size_result.median:.0f} bp, "
                    f"mean={insert_size_result.mean:.0f} bp, "
                    f"mu_log={insert_size_result.mu_log:.3f} "
                    f"[{insert_size_result.mu_log_ci[0]:.3f}, "
                    f"{insert_size_result.mu_log_ci[1]:.3f}], "
                    f"sigma_log={insert_size_result.sigma_log:.3f} "
                    f"[{insert_size_result.sigma_log_ci[0]:.3f}, "
                    f"{insert_size_result.sigma_log_ci[1]:.3f}], "
                    f"bins_used={insert_size_result.n_bins_used}",
                    err=True,
                )
            else:
                typer.echo(
                    "WARNING: No path combo sketches were built; "
                    "skipping insert size estimation.",
                    err=True,
                )

    # Close sketches now that estimation is done.
    if pe_sketch_obj is not None:
        pe_sketch_obj.close()
    if intra_sketch_obj is not None:
        intra_sketch_obj.close()

    if json_out is not None:
        stats = _compute_summary(
            graph, path_lengths, weight_str, pair_distances,
        )
        del graph  # last use; release graph memory
        stats["gfa"] = str(gfa)
        if pe_combo_unique is not None:
            stats["pe_combo_sketch_unique_hashes"] = pe_combo_unique
            stats["combo_density"] = combo_density
        if intra_combo_unique is not None:
            stats["intra_combo_sketch_unique_hashes"] = intra_combo_unique
            stats.setdefault("combo_density", combo_density)
        if insert_size_result is not None:
            stats["insert_size"] = {
                "median": insert_size_result.median,
                "mean": insert_size_result.mean,
                "mu_log": insert_size_result.mu_log,
                "sigma_log": insert_size_result.sigma_log,
                "mu_log_ci": list(insert_size_result.mu_log_ci),
                "sigma_log_ci": list(insert_size_result.sigma_log_ci),
                "n_bins_used": insert_size_result.n_bins_used,
                "inference": insert_size_result.inference,
                "signal_reliable": insert_size_result.signal_reliable,
            }
        json_out.write_text(json.dumps(stats, indent=2) + "\n")
        typer.echo(f"Stats written to {json_out}", err=True)


if __name__ == "__main__":
    app()
