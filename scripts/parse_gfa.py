#!/usr/bin/env python3
"""Parse a rust-mdbg GFA output into a rustworkx graph and report properties.

Optionally loads per-read minimizer IDs produced by rust-mdbg
``--dump-read-minimizers`` for downstream alignment to sampled paths.

Usage::

    python scripts/parse_gfa.py rust_mdbg_out.gfa
    python scripts/parse_gfa.py rust_mdbg_out.gfa --samples 5000
    python scripts/parse_gfa.py rust_mdbg_out.gfa --no-sample
    python scripts/parse_gfa.py rust_mdbg_out.gfa --paired-end --pe-bam reads.bam
    python scripts/parse_gfa.py rust_mdbg_out.gfa --read-minimizers rust_mdbg_out
"""

from __future__ import annotations

import bisect
import glob
import io
import json
import logging
import math
import random
import re
import struct
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Protocol

import lmdb  # type: ignore[import-untyped]
import lz4.frame  # type: ignore[import-untyped]
import numpy as np

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
# rust-mdbg read minimizers
# ------------------------------------------------------------------


def load_minimizer_table(table_path: Path) -> dict[int, str]:
    """Load a rust-mdbg minimizer_table file into a hash-to-lmer dict.

    Args:
        table_path: Path to ``{prefix}.minimizer_table`` (plain-text
            TSV: ``hash<TAB>lmer`` per line).

    Returns:
        Mapping of NT-hash (u64) to l-mer string.
    """
    table: dict[int, str] = {}
    with open(table_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                table[int(parts[0])] = parts[1]
    logger.info(
        "Loaded %d minimizers from %s", len(table), table_path,
    )
    return table


# ------------------------------------------------------------------
# Read index
# ------------------------------------------------------------------

# Strips common paired-end read-name suffixes to obtain the template name.
# Handles: /1 /2, /R1 /R2, _R1 _R2, .1 .2, .R1 .R2,
#          and Illumina CASAVA space suffixes (e.g. " 1:N:0:BARCODE").

@dataclass
class ReadIndex:
    """Integer-keyed read registry.

    Read IDs are 0-based; v2 binary indices are 1-based, so
    ``read_id = read_index - 1``.

    Attributes:
        n_reads: Total number of reads.
        pairs: Read ID → mate read ID for paired-end reads.
        l: Minimizer length from LMDB meta; 0 if not written by rust-mdbg.
    """

    n_reads: int
    pairs: dict[int, int]
    l: int = 0


def build_read_index(
    lmdb_path: Path,
    paired_interleaved: bool = False,
) -> ReadIndex:
    """Build a read registry from the LMDB index written by rust-mdbg.

    Reads n_reads from the ``meta`` sub-database (key ``b"n_reads"``,
    value u64 LE), then builds arithmetic pairs when *paired_interleaved*
    is True.

    Args:
        lmdb_path: Path to the LMDB environment directory (``{prefix}.index.lmdb``).
        paired_interleaved: Build arithmetic pairs (0↔1, 2↔3, …).

    Returns:
        A :class:`ReadIndex` with ``n_reads`` and ``pairs``.

    Raises:
        FileNotFoundError: If the LMDB directory does not exist.
        RuntimeError: If ``n_reads`` is missing from the LMDB meta sub-db.
    """
    if not lmdb_path.exists():
        raise FileNotFoundError(
            f"LMDB index not found at {lmdb_path}; run rust-mdbg with "
            f"--dump-read-minimizers first"
        )
    env = lmdb.open(
        str(lmdb_path), readonly=True, lock=False,
        max_dbs=_LMDB_MAX_DBS, map_size=_LMDB_MAP_SIZE,
    )
    try:
        meta_db = env.open_db(_DB_META)
        with env.begin() as txn:
            nr = txn.get(_META_N_READS, db=meta_db)
            l_bytes = txn.get(_META_L, db=meta_db)
        if nr is None:
            raise RuntimeError(
                f"n_reads not found in LMDB meta at {lmdb_path}; "
                "re-run rust-mdbg with --dump-read-minimizers"
            )
        n_reads = struct.unpack("<Q", nr)[0]
        l_val = struct.unpack("<I", l_bytes)[0] if l_bytes else 0
    finally:
        env.close()

    pairs: dict[int, int] = {}
    if paired_interleaved:
        for i in range(0, n_reads - 1, 2):
            pairs[i] = i + 1
            pairs[i + 1] = i
        logger.info(
            "Interleaved pairing: %d pairs from %d reads",
            len(pairs) // 2, n_reads,
        )
    logger.info("Read index: %d reads, %d paired templates", n_reads, len(pairs) // 2)
    return ReadIndex(n_reads=n_reads, pairs=pairs, l=l_val)


def path_minimizer_sequence(
    path: _OrientedPath,
    seg_min_index: list[tuple[np.ndarray, np.ndarray] | None],
    k: int,
) -> np.ndarray:
    """Build the minimizer sequence for a sampled graph path.

    Adjacent segments in a rust-mdbg path share *k* − 1 minimizers
    (the graph is a k-mer de Bruijn graph over minimizer sequences).
    The overlapping prefix of each segment after the first is dropped
    before concatenation.

    Because rust-mdbg stores nodes in their canonical (normalised) form
    (``REVCOMP_AWARE = true``), about half of all nodes are stored in
    reversed order relative to the traversal direction.  The reversed
    copy is precomputed in *seg_min_index* so this function selects
    ``fwd`` or ``rev`` by integer index with no runtime reversal.

    Args:
        path: Ordered list of ``(node_idx, is_forward)`` pairs as
            returned by :func:`_random_simple_path`.
        seg_min_index: Node-index-keyed list of ``(fwd, rev)`` uint64
            arrays from :func:`build_seg_min_index_from_lmdb`.
        k: rust-mdbg k-mer size (overlap between adjacent segments is
            *k* − 1 minimizers).

    Returns:
        1-D uint64 numpy array of minimizer IDs spanning the entire
        path in traversal order.
    """
    overlap = k - 1
    parts: list[np.ndarray] = []
    for node_idx, is_forward in path:
        pair = seg_min_index[node_idx]
        if pair is None:
            continue
        arr = pair[0] if is_forward else pair[1]
        parts.append(arr if not parts else arr[overlap:])
    if not parts:
        return np.empty(0, dtype=np.uint64)
    return np.concatenate(parts)


# ------------------------------------------------------------------
# Unified on-disk LMDB index
# ------------------------------------------------------------------

_LMDB_MAP_SIZE: int = 32 * 1024**3    # 32 GiB — fallback for read-only opens (virtual only)
_LMDB_MAX_DBS: int = 6

_DB_META  = b"meta"   # metadata
_DB_READS = b"reads"  # read_name → packed u64 minimizer IDs
_DB_SEGS  = b"segs"   # seg_name  → packed u64 fwd minimizer IDs
_DB_KINV  = b"kinv"   # (k, kmer) → packed (read_id u32, read_pos u32) pairs
_DB_KCNT  = b"kcnt"   # (read_id u32, k u8) → n_distinct_kmers u16

_META_K_LEVELS = b"k_levels"
_META_N_READS  = b"n_reads"
_META_SEG_K    = b"seg_k"
_META_N_LEVELS = b"n_levels"
_META_L        = b"l"           # minimizer length written by rust-mdbg


def _kinv_key(k: int, kmer: tuple[int, ...]) -> bytes:
    """Pack a k-mer into an LMDB kinv key."""
    return struct.pack(f"<B{k}Q", k, *kmer)


def _kcnt_key(read_id: int, k: int) -> bytes:
    """Pack a (read_id, k) pair into an LMDB kcnt key."""
    return struct.pack("<IB", read_id, k)


def _check_lmdb_valid(
    db_path: Path, n_levels: int,
) -> tuple[bool, list[int], int | None]:
    """Return (is_valid, k_levels, seg_k).

    Returns (False, [], None) if the index is absent, corrupt, or kinv/kcnt
    were not built (n_levels missing or mismatched).  The reads sub-db is
    populated by rust-mdbg; this check only validates the kinv/kcnt layer
    built on top by :func:`build_lmdb_index`.

    Args:
        db_path: LMDB environment directory.
        n_levels: Expected number of k levels.

    Returns:
        Tuple of (is_valid, k_levels, seg_k).
    """
    if not db_path.exists():
        return False, [], None
    try:
        env = lmdb.open(
            str(db_path), readonly=True, lock=False,
            max_dbs=_LMDB_MAX_DBS, map_size=_LMDB_MAP_SIZE,
        )
        meta_db = env.open_db(_DB_META)
        with env.begin() as txn:
            kl  = txn.get(_META_K_LEVELS, db=meta_db)
            nl  = txn.get(_META_N_LEVELS, db=meta_db)
            sk  = txn.get(_META_SEG_K,    db=meta_db)
        env.close()
        if not (kl and nl):
            return False, [], None
        if struct.unpack("<H", nl)[0] != n_levels:
            return False, [], None
        k_levels: list[int] = json.loads(kl.decode())
        raw_sk = struct.unpack("<h", sk)[0] if sk else -1
        seg_k: int | None = raw_sk if raw_sk != -1 else None
        return True, k_levels, seg_k
    except Exception as exc:
        logger.warning("Cannot read LMDB index at %s: %s", db_path, exc)
        return False, [], None


def _compute_lmdb_map_size(n_reads: int, l: int, n_levels: int = 1) -> int:
    """Return a map_size (bytes) sufficient for the LMDB given n_reads, l, n_levels.

    Cost model (4× B-tree overhead):
      reads_db: n_reads × avg_min × 8 B
      kinv:     n_reads × avg_min × n_levels × 25 B
      kcnt:     n_reads × n_levels × 7 B
    where avg_min = l / 1.25.
    Result is rounded up to the next GiB with a minimum of 4 GiB.
    """
    gib = 1024 ** 3
    avg_min = l / 1.25
    raw = int(n_reads * (avg_min * 8 + avg_min * n_levels * 25 + n_levels * 7) * 4)
    rounded = math.ceil(raw / gib) * gib
    return max(4 * gib, rounded)


def build_lmdb_index(
    db_path: Path,
    read_prefix: Path,
    index: ReadIndex,
    n_levels: int = 5,
) -> tuple[list[int], int | None]:
    """Build (or reuse) the unified on-disk LMDB index.

    Stores three datasets in named sub-databases within one LMDB
    environment:

    * ``reads`` — per-read minimizer ID arrays from ``.read_minimizers``.
    * ``segs``  — per-segment forward minimizer arrays from ``.sequences``.
    * ``kinv`` / ``kcnt`` — inverted k-mer index for read-to-path matching.

    If *db_path* already contains a valid index (matching *n_reads* and
    *n_levels*) it is returned without rebuilding.

    Args:
        db_path: Destination directory for the LMDB environment.
        read_prefix: rust-mdbg output prefix.
        index: :class:`ReadIndex` supplying the name-to-ID mapping.
        n_levels: Number of k-mer size levels for the matching index.

    Returns:
        Tuple of ``(k_levels, seg_k)``.
    """
    valid, k_levels, seg_k = _check_lmdb_valid(db_path, n_levels)
    if valid:
        logger.info(
            "Reusing LMDB index at %s (k_levels=%s, seg_k=%s)",
            db_path, k_levels, seg_k,
        )
        return k_levels, seg_k

    logger.info("Building LMDB index at %s", db_path)
    map_size = (
        _compute_lmdb_map_size(index.n_reads, index.l, n_levels)
        if index.l > 0
        else _LMDB_MAP_SIZE
    )
    # Rust already created the LMDB directory and populated the reads sub-db.
    env = lmdb.open(
        str(db_path),
        map_size=map_size,
        max_dbs=_LMDB_MAX_DBS,
        sync=False,
    )
    reads_db = env.open_db(_DB_READS)
    segs_db  = env.open_db(_DB_SEGS)
    kinv_db  = env.open_db(_DB_KINV)
    kcnt_db  = env.open_db(_DB_KCNT)
    meta_db  = env.open_db(_DB_META)

    _BATCH = 20_000

    # -- Stage 1 REMOVED: reads sub-db already populated by rust-mdbg --------

    # -- Stage 2: .sequences → segs sub-db; extract k -----------------------
    seg_k_found: int | None = None
    seq_files = sorted(glob.glob(str(read_prefix) + ".*.sequences"))

    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "LMDB: segment minimizers", total=len(seq_files),
        )
        for fpath in seq_files:
            raw = lz4.frame.decompress(Path(fpath).read_bytes())
            with env.begin(write=True) as txn:
                for line in io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"):
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    if line.startswith("# k = ") and seg_k_found is None:
                        try:
                            seg_k_found = int(line[6:].split()[0])
                        except ValueError:
                            pass
                        continue
                    if line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    bracket = parts[1].strip()
                    seg_mids = (
                        [int(x.strip()) for x in bracket[1:-1].split(",") if x.strip()]
                        if bracket.startswith("[") and bracket.endswith("]")
                        else []
                    )
                    val = struct.pack(f"<{len(seg_mids)}Q", *seg_mids) if seg_mids else b""
                    txn.put(parts[0].encode(), val, db=segs_db)
            progress.advance(task)

    # -- Compute min/max minimizer count from reads_db (populated by Rust) ----
    min_len = 10 ** 9
    max_len = 0
    with env.begin() as txn:
        cursor = txn.cursor(db=reads_db)
        for _key, val_bytes in cursor:
            n = len(val_bytes) // 8
            if n:
                min_len = min(min_len, n)
                max_len = max(max_len, n)

    # -- Stage 3: reads sub-db → kinv + kcnt sub-dbs ------------------------
    if max_len == 0:
        logger.warning("No reads with minimizers; k-mer index will be empty.")
        k_levels = []
    else:
        k_levels = _compute_k_levels(min_len, max_len, n_levels)
    logger.info("LMDB k levels: %s", k_levels)

    if k_levels:
        inv_acc: dict[bytes, list[int]] = {}
        cnt_acc: dict[bytes, int] = {}
        n_processed = 0

        with Progress(*_PROGRESS_COLUMNS) as progress:
            task = progress.add_task(
                "LMDB: k-mer index", total=index.n_reads,
            )
            with env.begin() as rtxn:
                cursor = rtxn.cursor(db=reads_db)
                for name_bytes, val_bytes in cursor:
                    read_id = int(name_bytes.decode()) - 1
                    if read_id >= index.n_reads:
                        progress.advance(task)
                        continue
                    n = len(val_bytes) // 8
                    mids: tuple[int, ...] = (
                        struct.unpack(f"<{n}Q", val_bytes) if n else ()
                    )
                    k = _floor_k_level(n, k_levels)
                    if k is not None and mids:
                        seen: dict[tuple[int, ...], int] = {}
                        for pos in range(n - k + 1):
                            km = tuple(mids[pos:pos + k])
                            if km not in seen:
                                seen[km] = pos
                        for km, pos in seen.items():
                            ikey = _kinv_key(k, km)
                            entry = inv_acc.setdefault(ikey, [])
                            entry.append(read_id)
                            entry.append(pos)
                        cnt_acc[_kcnt_key(read_id, k)] = len(seen)
                    n_processed += 1
                    progress.advance(task)

                    if n_processed % _BATCH == 0:
                        with env.begin(write=True) as wtxn:
                            for ikey, flat in inv_acc.items():
                                existing = wtxn.get(ikey, db=kinv_db) or b""
                                wtxn.put(
                                    ikey,
                                    existing + struct.pack(f"<{len(flat)}I", *flat),
                                    db=kinv_db,
                                )
                            for ckey, cnt in cnt_acc.items():
                                wtxn.put(
                                    ckey,
                                    struct.pack("<H", min(cnt, 65535)),
                                    db=kcnt_db,
                                )
                        inv_acc.clear()
                        cnt_acc.clear()

        if inv_acc or cnt_acc:
            with env.begin(write=True) as wtxn:
                for ikey, flat in inv_acc.items():
                    existing = wtxn.get(ikey, db=kinv_db) or b""
                    wtxn.put(
                        ikey,
                        existing + struct.pack(f"<{len(flat)}I", *flat),
                        db=kinv_db,
                    )
                for ckey, cnt in cnt_acc.items():
                    wtxn.put(
                        ckey, struct.pack("<H", min(cnt, 65535)), db=kcnt_db,
                    )

    # -- Stage 4: metadata ---------------------------------------------------
    with env.begin(write=True) as txn:
        txn.put(_META_K_LEVELS, json.dumps(k_levels).encode(), db=meta_db)
        txn.put(_META_N_READS,  struct.pack("<Q", index.n_reads),  db=meta_db)
        txn.put(_META_SEG_K,
                struct.pack("<h", seg_k_found if seg_k_found is not None else -1),
                db=meta_db)
        txn.put(_META_N_LEVELS, struct.pack("<H", n_levels), db=meta_db)

    env.sync()
    env.close()
    logger.info("LMDB index built at %s", db_path)
    return k_levels, seg_k_found


def build_seg_min_index_from_lmdb(
    graph: rx.PyGraph,
    lmdb_env: lmdb.Environment,
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Build a node-indexed segment minimizer list directly from the LMDB index.

    Replaces :func:`load_segment_minimizers` + :func:`build_seg_min_index`
    by querying the ``segs`` sub-database rather than re-reading the
    ``.sequences`` files, eliminating the intermediate in-memory dict.

    Args:
        graph: Parsed GFA graph.
        lmdb_env: Open LMDB environment (from :func:`build_lmdb_index`).

    Returns:
        List of ``(fwd, rev)`` uint64 array pairs indexed by graph node
        index, with ``None`` for segments absent from the index.
    """
    segs_db = lmdb_env.open_db(_DB_SEGS)
    n = graph.num_nodes()
    result: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n
    n_found = 0

    with lmdb_env.begin() as txn:
        for node_idx in graph.node_indices():
            seg: Segment = graph[node_idx]
            val = txn.get(seg.name.encode(), db=segs_db)
            if val is None:
                continue
            n_mids = len(val) // 8
            if n_mids:
                fwd = np.frombuffer(val, dtype=np.uint64).copy()
                result[node_idx] = (fwd, fwd[::-1].copy())
            else:
                empty = np.empty(0, dtype=np.uint64)
                result[node_idx] = (empty, empty)
            n_found += 1

    logger.info(
        "Loaded minimizers for %d/%d segments from LMDB", n_found, n,
    )
    return result


# ------------------------------------------------------------------
# Matcher mode and shared protocol
# ------------------------------------------------------------------


class MatcherMode(str, Enum):
    """Substring-matching strategy for read-to-path alignment."""

    exact = "exact"
    pseudo_match = "pseudo-match"


class ReadMatcher(Protocol):
    """Protocol satisfied by every read-to-path matcher.

    A matcher is built once from the read minimizer data and then
    queried once per sampled path via :meth:`search_path`.
    """

    def search_path(self, path_seq: np.ndarray) -> list[PathMatch]:
        """Return all reads found in *path_seq*.

        Args:
            path_seq: 1-D uint64 array of minimizer IDs for one path.

        Returns:
            List of :class:`PathMatch` records (one per hit occurrence).
        """
        ...

    def describe(self) -> str:
        """Return a short human-readable description of the index."""
        ...


# ------------------------------------------------------------------
# Ordered-match matcher (ordered k-mer subsequence, index paths)
# ------------------------------------------------------------------


def _compute_k_levels(min_len: int, max_len: int, n_levels: int) -> list[int]:
    """Compute distinct k values evenly distributed across [min_len, max_len].

    Args:
        min_len: Minimum read length in minimizers (lower bound for k).
        max_len: Maximum read length in minimizers (upper bound for k).
        n_levels: Desired total number of levels including the endpoints.

    Returns:
        Sorted list of distinct integer k values, length ≤ n_levels.
    """
    if n_levels <= 1 or min_len >= max_len:
        return [max(1, min_len)]
    raw = np.linspace(min_len, max_len, n_levels)
    seen: set[int] = set()
    result: list[int] = []
    for val in np.round(raw).astype(int):
        k_int = int(val)
        if k_int >= 1 and k_int not in seen:
            seen.add(k_int)
            result.append(k_int)
    return result


def _floor_k_level(read_len: int, k_levels: list[int]) -> int | None:
    """Return the largest k in k_levels that does not exceed read_len.

    Args:
        read_len: Number of minimizers in the read.
        k_levels: Sorted ascending list of candidate k values.

    Returns:
        Largest k ≤ read_len, or None if every level exceeds read_len.
    """
    best: int | None = None
    for k in k_levels:
        if k <= read_len:
            best = k
    return best


class OrderedMatchMatcher:
    """Read-to-path matcher using ordered k-mer subsequence matching.

    Indexes sampled paths at multiple k-mer sizes spanning the range of
    observed read lengths.  For each read the k value is floored to the
    nearest indexed level; a read matches a path when every k-mer formed
    from its minimizer sequence appears in that path **in the same
    left-to-right order** (i.e. the read k-mer sequence is a subsequence
    of the path k-mer sequence).

    Both forward and reversed read orientations are tested; each
    ``(read_id, path_idx)`` pair is emitted at most once.

    Pre-computed results are served sequentially via :meth:`search_path`
    in the same order as the *path_seqs* passed to
    :func:`build_ordered_matcher`.

    Attributes:
        _per_path_matches: Per-path :class:`PathMatch` lists.
        _n_total: Total read–path match events recorded.
        _k_levels: K values used for the multi-level index.
        _n_reads_indexed: Reads processed during the build pass.
        _search_idx: Next sequential index for :meth:`search_path`.
    """

    def __init__(
        self,
        per_path_matches: list[list[PathMatch]],
        n_total: int,
        k_levels: list[int],
        n_reads_indexed: int,
    ) -> None:
        self._per_path_matches = per_path_matches
        self._n_total = n_total
        self._k_levels = k_levels
        self._n_reads_indexed = n_reads_indexed
        self._search_idx = 0

    def search_path(self, path_seq: np.ndarray) -> list[PathMatch]:
        """Return pre-computed matches for the next path in sequence.

        Args:
            path_seq: Ignored; present for :class:`ReadMatcher` protocol
                compatibility.

        Returns:
            List of :class:`PathMatch` records for this path.
        """
        result = self._per_path_matches[self._search_idx]
        self._search_idx += 1
        return result

    def describe(self) -> str:
        """Return a short human-readable description of the index."""
        k_str = ", ".join(str(k) for k in self._k_levels)
        return (
            f"Ordered-match: {len(self._k_levels)} k levels [{k_str}], "
            f"{self._n_reads_indexed:,} reads indexed, "
            f"{self._n_total:,} total match events"
        )


def build_ordered_matcher(
    path_seqs: list[np.ndarray],
    lmdb_env: lmdb.Environment,
    k_levels: list[int],
    index: ReadIndex,
    eligible: set[int] | None = None,
) -> OrderedMatchMatcher:
    """Build an :class:`OrderedMatchMatcher` by querying the on-disk LMDB index.

    For each sampled path at each k level, builds an in-memory positional
    k-mer map, then queries the LMDB inverted index for candidate reads.
    Each candidate is verified by a greedy subsequence check: the read's
    k-mers (in their original order, reconstructed from LMDB positions)
    must appear at strictly increasing positions in the path.

    No streaming over read minimizer files is performed after the LMDB
    index has been built.

    Args:
        path_seqs: Minimizer sequences for the sampled paths.
        lmdb_env: Open LMDB environment from :func:`build_lmdb_index`.
        k_levels: K values loaded from LMDB metadata.
        index: :class:`ReadIndex` for read-count reporting.
        eligible: Optional set of read IDs to include; others skipped.

    Returns:
        An :class:`OrderedMatchMatcher` with all matches pre-computed,
        ready for sequential :meth:`~OrderedMatchMatcher.search_path` calls.
    """
    if not k_levels:
        logger.warning("No k levels; returning empty ordered matcher.")
        return OrderedMatchMatcher([[] for _ in path_seqs], 0, [], 0)

    kinv_db = lmdb_env.open_db(_DB_KINV)
    kcnt_db = lmdb_env.open_db(_DB_KCNT)
    path_lens = [len(seq) for seq in path_seqs]
    per_path: list[list[PathMatch]] = [[] for _ in path_seqs]
    seen_matches: set[tuple[int, int]] = set()
    n_total = 0

    with lmdb_env.begin() as txn, Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Ordered-match (LMDB)", total=len(path_seqs),
        )
        for path_idx, seq in enumerate(path_seqs):
            for k in k_levels:
                n_seq = len(seq)
                if n_seq < k:
                    continue

                # Per-path positional index: kmer → sorted positions in path
                path_kmer_pos: dict[tuple[int, ...], list[int]] = {}
                for pos in range(n_seq - k + 1):
                    km = tuple(int(m) for m in seq[pos:pos + k])
                    path_kmer_pos.setdefault(km, []).append(pos)
                path_kmer_set = set(path_kmer_pos.keys())

                # Collect candidate reads and their (kmer, read_pos) hits
                # candidate_hits: read_id → [(km, read_pos)]
                candidate_hits: dict[int, list[tuple[tuple[int, ...], int]]] = {}
                candidate_counts: dict[int, int] = {}
                for km in path_kmer_set:
                    inv_data = txn.get(_kinv_key(k, km), db=kinv_db)
                    if inv_data is None:
                        continue
                    arr = np.frombuffer(inv_data, dtype=np.uint32).reshape(-1, 2)
                    for row in arr:
                        rid = int(row[0])
                        rpos = int(row[1])
                        if eligible is not None and rid not in eligible:
                            continue
                        candidate_counts[rid] = candidate_counts.get(rid, 0) + 1
                        candidate_hits.setdefault(rid, []).append((km, rpos))

                for rid, match_count in candidate_counts.items():
                    cnt_data = txn.get(_kcnt_key(rid, k), db=kcnt_db)
                    if cnt_data is None:
                        continue
                    n_total_kmers = struct.unpack("<H", cnt_data)[0]
                    if match_count != n_total_kmers:
                        continue
                    # All read k-mers in path (pseudo-match passed).
                    # Now verify order: read k-mers by read_pos must map
                    # to strictly increasing path positions.
                    hits_sorted = sorted(
                        candidate_hits[rid], key=lambda x: x[1],
                    )
                    last_path_pos = -1
                    valid = True
                    for km, _ in hits_sorted:
                        positions = path_kmer_pos[km]
                        idx = bisect.bisect_right(positions, last_path_pos)
                        if idx >= len(positions):
                            valid = False
                            break
                        last_path_pos = positions[idx]
                    if not valid:
                        continue
                    key = (rid, path_idx)
                    if key not in seen_matches:
                        seen_matches.add(key)
                        per_path[path_idx].append(
                            PathMatch(
                                read_id=rid,
                                path_start=0,
                                path_end=path_lens[path_idx],
                            )
                        )
                        n_total += 1
            progress.advance(task)

    logger.info(
        "Ordered-match (LMDB): %d total match events across %d paths",
        n_total, len(path_seqs),
    )
    return OrderedMatchMatcher(per_path, n_total, k_levels, index.n_reads)


# ------------------------------------------------------------------
# Pseudo-match matcher (unordered k-mer set membership)
# ------------------------------------------------------------------


class PseudoMatchMatcher:
    """Read-to-path matcher using unordered k-mer set membership.

    Indexes sampled paths at multiple k-mer sizes spanning the range of
    observed read lengths.  For each read the k value is floored to the
    nearest indexed level; a read matches a path when every k-mer formed
    from its minimizer sequence appears somewhere in that path, with no
    constraint on order.

    Both forward and reversed read orientations are tested; each
    (read_id, path_idx) pair is emitted at most once.

    Pre-computed results are served sequentially via :meth:`search_path`
    in the same order as the *path_seqs* passed to :func:`build_pseudo_matcher`.

    Attributes:
        _per_path_matches: Per-path :class:`PathMatch` lists.
        _n_total: Total read–path match events recorded.
        _k_levels: K values used for the multi-level index.
        _n_reads_indexed: Reads processed during the build pass.
        _search_idx: Next sequential index for :meth:`search_path`.
    """

    def __init__(
        self,
        per_path_matches: list[list[PathMatch]],
        n_total: int,
        k_levels: list[int],
        n_reads_indexed: int,
    ) -> None:
        self._per_path_matches = per_path_matches
        self._n_total = n_total
        self._k_levels = k_levels
        self._n_reads_indexed = n_reads_indexed
        self._search_idx = 0

    def search_path(self, path_seq: np.ndarray) -> list[PathMatch]:
        """Return pre-computed matches for the next path in sequence.

        Args:
            path_seq: Ignored; present for :class:`ReadMatcher` protocol
                compatibility.

        Returns:
            List of :class:`PathMatch` records for this path.
        """
        result = self._per_path_matches[self._search_idx]
        self._search_idx += 1
        return result

    def describe(self) -> str:
        """Return a short human-readable description of the index."""
        k_str = ", ".join(str(k) for k in self._k_levels)
        return (
            f"Pseudo-match: {len(self._k_levels)} k levels [{k_str}], "
            f"{self._n_reads_indexed:,} reads indexed, "
            f"{self._n_total:,} total match events"
        )


def build_pseudo_matcher(
    path_seqs: list[np.ndarray],
    lmdb_env: lmdb.Environment,
    k_levels: list[int],
    index: ReadIndex,
    eligible: set[int] | None = None,
) -> PseudoMatchMatcher:
    """Build a :class:`PseudoMatchMatcher` by querying the on-disk LMDB index.

    For each sampled path at each k level, builds an in-memory k-mer set,
    then queries the LMDB inverted index for candidate reads.  A candidate
    matches when the number of its distinct k-mers found in the path equals
    its total distinct k-mer count (stored in the ``kcnt`` sub-database).

    No streaming over read minimizer files is performed after the LMDB
    index has been built.

    Args:
        path_seqs: Minimizer sequences for the sampled paths.
        lmdb_env: Open LMDB environment from :func:`build_lmdb_index`.
        k_levels: K values loaded from LMDB metadata.
        index: :class:`ReadIndex` for read-count reporting.
        eligible: Optional set of read IDs to include; others skipped.

    Returns:
        A :class:`PseudoMatchMatcher` with all matches pre-computed,
        ready for sequential :meth:`~PseudoMatchMatcher.search_path` calls.
    """
    if not k_levels:
        logger.warning("No k levels; returning empty pseudo-match matcher.")
        return PseudoMatchMatcher([[] for _ in path_seqs], 0, [], 0)

    kinv_db = lmdb_env.open_db(_DB_KINV)
    kcnt_db = lmdb_env.open_db(_DB_KCNT)
    path_lens = [len(seq) for seq in path_seqs]
    per_path: list[list[PathMatch]] = [[] for _ in path_seqs]
    seen_matches: set[tuple[int, int]] = set()
    n_total = 0

    with lmdb_env.begin() as txn, Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Pseudo-match (LMDB)", total=len(path_seqs),
        )
        for path_idx, seq in enumerate(path_seqs):
            for k in k_levels:
                n_seq = len(seq)
                if n_seq < k:
                    continue

                # Build path k-mer set at this level
                path_kmer_set: set[tuple[int, ...]] = set()
                for pos in range(n_seq - k + 1):
                    path_kmer_set.add(
                        tuple(int(m) for m in seq[pos:pos + k])
                    )

                # Collect candidate reads: count matching k-mers per read_id
                candidate_counts: dict[int, int] = {}
                for km in path_kmer_set:
                    inv_data = txn.get(_kinv_key(k, km), db=kinv_db)
                    if inv_data is None:
                        continue
                    arr = np.frombuffer(inv_data, dtype=np.uint32).reshape(-1, 2)
                    for row in arr:
                        rid = int(row[0])
                        if eligible is None or rid in eligible:
                            candidate_counts[rid] = (
                                candidate_counts.get(rid, 0) + 1
                            )

                # Verify: read matches iff all its k-mers are in path_kmer_set
                for rid, match_count in candidate_counts.items():
                    cnt_data = txn.get(_kcnt_key(rid, k), db=kcnt_db)
                    if cnt_data is None:
                        continue
                    if match_count != struct.unpack("<H", cnt_data)[0]:
                        continue
                    key = (rid, path_idx)
                    if key not in seen_matches:
                        seen_matches.add(key)
                        per_path[path_idx].append(
                            PathMatch(
                                read_id=rid,
                                path_start=0,
                                path_end=path_lens[path_idx],
                            )
                        )
                        n_total += 1
            progress.advance(task)

    logger.info(
        "Pseudo-match (LMDB): %d total match events across %d paths",
        n_total, len(path_seqs),
    )
    return PseudoMatchMatcher(per_path, n_total, k_levels, index.n_reads)


# ------------------------------------------------------------------
# Path matching
# ------------------------------------------------------------------


@dataclass
class PathMatch:
    """A read whose minimizer sequence was found within a path.

    Attributes:
        read_id: Integer read ID from :class:`ReadIndex`.
        path_start: Start index in the path's minimizer sequence.
        path_end: Exclusive end index in the path's minimizer sequence.
    """

    read_id: int
    path_start: int
    path_end: int


def match_reads_to_path(
    path_min_seq: np.ndarray,
    matcher: ReadMatcher,
) -> list[PathMatch]:
    """Find all reads whose minimizer sequences occur in *path_min_seq*.

    Args:
        path_min_seq: Concatenated minimizer sequence for a sampled
            graph path, from :func:`path_minimizer_sequence`.
        matcher: A compiled :class:`ReadMatcher`.

    Returns:
        List of :class:`PathMatch` records, one per occurrence.
    """
    return matcher.search_path(path_min_seq)


def _minimizer_to_bp_scale(
    graph: rx.PyGraph,
    seg_min_index: list[tuple[np.ndarray, np.ndarray] | None],
) -> float | None:
    """Estimate average base pairs per minimizer from assembly data.

    Sums segment lengths and minimizer counts across all segments that
    have minimizer data in *seg_min_index*.  Segments with no loaded
    minimizers are excluded from both totals.

    Args:
        graph: Parsed GFA graph.
        seg_min_index: Node-index-keyed list of ``(fwd, rev)`` arrays
            from :func:`build_seg_min_index_from_lmdb`.

    Returns:
        Scale factor (bp per minimizer), or ``None`` if no segments
        with minimizers were found.
    """
    total_bp = 0
    total_min = 0
    for idx in graph.node_indices():
        pair = seg_min_index[idx]
        if pair is not None and len(pair[0]):
            total_bp += graph[idx].length
            total_min += len(pair[0])
    if total_min == 0:
        return None
    return total_bp / total_min


def _path_pair_insert_sizes(
    path_matches: list[PathMatch],
    index: ReadIndex,
) -> list[tuple[int, int, int]]:
    """Enumerate all (r1_id, r2_id, minimizer_span) tuples for read pairs.

    Groups match positions by read ID, then for every pair where both
    mates appear in the current path returns the Cartesian product of
    their hit positions.  The minimizer-space insert size is::

        max(r1_end, r2_end) - min(r1_start, r2_start)

    Multiple hits per read (when a read's minimizer sequence appears
    more than once in the path) produce multiple entries — all
    permutations are reported.

    Args:
        path_matches: All :class:`PathMatch` records from one path.
        index: :class:`ReadIndex` with pair information.

    Returns:
        List of ``(r1_id, r2_id, minimizer_span)`` tuples, one per
        (r1_hit × r2_hit) combination found.
    """
    positions: dict[int, list[tuple[int, int]]] = {}
    for m in path_matches:
        positions.setdefault(m.read_id, []).append(
            (m.path_start, m.path_end)
        )

    results: list[tuple[int, int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()

    for r1_id, r1_hits in positions.items():
        r2_id = index.pairs.get(r1_id)
        if r2_id is None or r2_id not in positions:
            continue
        pair_key = (min(r1_id, r2_id), max(r1_id, r2_id))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        r2_hits = positions[r2_id]
        for r1_start, r1_end in r1_hits:
            for r2_start, r2_end in r2_hits:
                span = max(r1_end, r2_end) - min(r1_start, r2_start)
                results.append((r1_id, r2_id, span))

    return results


def _insert_size_stats(
    sizes_min: list[int],
    scale: float | None,
) -> dict[str, object]:
    """Compute insert size statistics in minimizer space and base pairs.

    Args:
        sizes_min: Minimizer-space insert sizes.
        scale: Base pairs per minimizer scale factor from
            :func:`_minimizer_to_bp_scale`, or ``None`` if unavailable.

    Returns:
        Dict with ``"observations"`` count, a ``"minimizer_space"`` sub-dict
        (min/max/mean/median/std_dev), and optionally a ``"bp"`` sub-dict
        with the same keys scaled by *scale* plus
        ``"scale_bp_per_minimizer"``.  Returns an empty dict when
        *sizes_min* is empty.
    """
    n = len(sizes_min)
    if n == 0:
        return {}
    sorted_s = sorted(sizes_min)
    total = sum(sorted_s)
    mean_s = total / n
    var_s = sum((x - mean_s) ** 2 for x in sorted_s) / n
    median_s: float = (
        sorted_s[n // 2]
        if n % 2 == 1
        else (sorted_s[n // 2 - 1] + sorted_s[n // 2]) / 2
    )
    stats: dict[str, object] = {
        "observations": n,
        "minimizer_space": {
            "min": sorted_s[0],
            "max": sorted_s[-1],
            "mean": mean_s,
            "median": median_s,
            "std_dev": math.sqrt(var_s),
        },
    }
    if scale is not None:
        bp = [x * scale for x in sizes_min]
        bp_sorted = sorted(bp)
        bp_mean = sum(bp_sorted) / n
        bp_var = sum((x - bp_mean) ** 2 for x in bp_sorted) / n
        bp_median: float = (
            bp_sorted[n // 2]
            if n % 2 == 1
            else (bp_sorted[n // 2 - 1] + bp_sorted[n // 2]) / 2
        )
        stats["bp"] = {
            "scale_bp_per_minimizer": scale,
            "min": bp_sorted[0],
            "max": bp_sorted[-1],
            "mean": bp_mean,
            "median": bp_median,
            "std_dev": math.sqrt(bp_var),
        }
    return stats


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
        help="Enable debug logging (includes per-read name and "
             "template name output when --read-minimizers is used)",
    )] = False,
    json_out: Annotated[Path | None, typer.Option(
        "--json",
        help="Write summary statistics as JSON to this path",
        dir_okay=False,
    )] = None,
    read_minimizers_prefix: Annotated[Path | None, typer.Option(
        "--read-minimizers",
        help="rust-mdbg output prefix for loading per-read minimizer IDs "
             "(e.g. rust_mdbg_out); reads all matching "
             "{prefix}.*.read_minimizers files",
    )] = None,
    lmdb_index: Annotated[Path | None, typer.Option(
        "--lmdb-index",
        help="Path to the unified on-disk LMDB index directory. Built "
             "automatically on first run from the .read_minimizers and "
             ".sequences files; reused on subsequent runs when the read "
             "count and k-level count match. Default: "
             "{read-minimizers-prefix}.index.lmdb",
    )] = None,
    minimizer_table: Annotated[Path | None, typer.Option(
        "--minimizer-table",
        help="Path to the rust-mdbg {prefix}.minimizer_table file "
             "(hash → l-mer lookup). Optional; not required for "
             "read-to-path matching (segment minimizers are loaded "
             "from the .sequences files instead)",
        exists=True,
        dir_okay=False,
    )] = None,
    k: Annotated[int, typer.Option(
        "--k",
        help="rust-mdbg k-mer size. Used as fallback if the value "
             "cannot be read from the .sequences file header",
    )] = 7,
    insert_sizes_out: Annotated[Path | None, typer.Option(
        "--insert-sizes-out",
        help="Write per-pair insert size records to this TSV file "
             "(columns: read1_name, read2_name, insert_size_minimizers, "
             "insert_size_bp). Requires --read-minimizers. All "
             "permutations of multi-position hits are reported.",
        dir_okay=False,
    )] = None,
    top_paths: Annotated[int, typer.Option(
        "--top-paths",
        help="Number of longest sampled paths to use for read mapping "
             "and insert-size estimation. 0 = use all sampled paths.",
    )] = 10,
    sample_component_proportion: Annotated[float, typer.Option(
        "--sample-component-proportion",
        help="Sample paths only from the largest connected components "
             "whose combined node count covers at least this fraction "
             "of all graph nodes (0.0–1.0). Set to 0 (default) to "
             "sample from all components.",
    )] = 0.0,
    read_mappings_out: Annotated[Path | None, typer.Option(
        "--read-mappings-out",
        help="Write a TSV of all read-to-path mappings to this file. "
             "Columns: read_name, pair (1/2/.), path_index (1-based).",
        dir_okay=False,
    )] = None,
    paths_out: Annotated[Path | None, typer.Option(
        "--paths-out",
        help="Write a TSV of the chosen paths to this file. "
             "Columns: path_index (1-based), minimizers "
             "(comma-separated IDs), nodes (comma-separated name+/-), "
             "path_length_bp.",
        dir_okay=False,
    )] = None,
    matcher: Annotated[MatcherMode, typer.Option(
        "--matcher",
        help="Read-to-path matching strategy. "
             "'exact' builds a multi-level k-mer positional index over paths "
             "and matches reads whose k-mer sequence is a subsequence of the "
             "path k-mer sequence (ordered, two streaming passes). "
             "'pseudo-match' builds a multi-level k-mer set index over paths; "
             "a read matches when all its k-mers are present in the path "
             "regardless of order (two streaming passes).",
        case_sensitive=False,
    )] = MatcherMode.pseudo_match,
    pseudo_match_levels: Annotated[int, typer.Option(
        "--pseudo-match-levels",
        help="Number of k-mer size levels for the 'pseudo-match' matcher. "
             "Levels are evenly spaced between the minimum and maximum "
             "observed read length (in minimizers). Each read is assigned "
             "to the largest level not exceeding its minimizer count.",
        min=1,
    )] = 5,
    interleaved_pairs: Annotated[bool, typer.Option(
        "--interleaved-pairs/--no-interleaved-pairs",
        help="Pair reads arithmetically assuming interleaved paired-end "
             "input: read 1 pairs with read 2, read 3 with read 4, etc. "
             "Intended for v2 read_minimizers files (RMBG\\x02) where "
             "names are sequential integer indices rather than FASTQ "
             "read identifiers. Falls back to template-name pairing if "
             "names are not integers.",
    )] = False,
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

    # When minimizer matching is requested we need the actual paths,
    # not just their lengths, so sample once and derive both.
    sampled: list[tuple[_OrientedPath, int]] | None = None
    path_lengths: list[int] | None = None
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
            + (":" if read_minimizers_prefix is None else ""),
            err=True,
        )
        if read_minimizers_prefix is None:
            _report_chosen_components(graph, chosen_comps)
        sampled = sample_paths(graph, samples, weight_str, eligible_nodes)
        path_lengths = [bp for _, bp in sampled]

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

    read_index: ReadIndex | None = None
    n_total_matches = 0
    ins_stats: dict[str, object] = {}
    bp_scale: float | None = None

    # --- Select top N paths by length before read mapping ---------------------
    if sampled and top_paths:
        sampled.sort(key=lambda t: t[1], reverse=True)
        sampled = sampled[:top_paths]

    if read_minimizers_prefix is not None:
        # Compute lmdb_path first — build_read_index reads n_reads from it.
        _lmdb_path = (
            lmdb_index
            if lmdb_index is not None
            else Path(str(read_minimizers_prefix) + ".index.lmdb")
        )
        read_index = build_read_index(_lmdb_path, interleaved_pairs)
        typer.echo(
            f"Read index: {read_index.n_reads:,} reads, "
            f"{len(read_index.pairs) // 2:,} paired templates",
            err=True,
        )

        # Build (or reuse) the unified on-disk LMDB index (kinv/kcnt/segs layers).
        k_levels, lmdb_seg_k = build_lmdb_index(
            _lmdb_path, read_minimizers_prefix, read_index,
            n_levels=pseudo_match_levels,
        )
        lmdb_env = lmdb.open(
            str(_lmdb_path), readonly=True, lock=False,
            max_dbs=_LMDB_MAX_DBS, map_size=_LMDB_MAP_SIZE,
        )

        # Precompute path sequences for the top paths only.
        path_seqs: list[np.ndarray] = []
        if sampled:
            seg_min_index = build_seg_min_index_from_lmdb(graph, lmdb_env)
            effective_k = lmdb_seg_k if lmdb_seg_k is not None else k
            bp_scale = _minimizer_to_bp_scale(graph, seg_min_index)
            typer.echo(f"Segment minimizers: (k={effective_k})", err=True)
            typer.echo(
                f"bp/minimizer scale: "
                + (f"{bp_scale:.3f}" if bp_scale else "unavailable"),
                err=True,
            )

            if chosen_comps:
                typer.echo(
                    f"Chosen components ({len(chosen_comps)} total):", err=True,
                )
                _report_chosen_components(graph, chosen_comps, seg_min_index)

            for path, _ in sampled:
                seq = path_minimizer_sequence(path, seg_min_index, effective_k)
                path_seqs.append(seq)
            del seg_min_index

        read_matcher: ReadMatcher
        if matcher == MatcherMode.exact:
            read_matcher = build_ordered_matcher(
                path_seqs, lmdb_env, k_levels, read_index,
            )
        else:
            read_matcher = build_pseudo_matcher(
                path_seqs, lmdb_env, k_levels, read_index,
            )
        lmdb_env.close()
        typer.echo(read_matcher.describe(), err=True)

        # --- Report and write paths --------------------------------------------
        if sampled:
            n_paths = len(sampled)
            unique_path_nodes = {
                node_idx
                for path, _ in sampled
                for node_idx, _ in path
            }
            n_unique = len(unique_path_nodes)
            n_total = graph.num_nodes()
            typer.echo(
                f"\nChosen paths ({n_paths} total, "
                f"covering {n_unique:,} unique nodes "
                f"= {n_unique / n_total:.1%} of {n_total:,}):",
                err=True,
            )
            for i, (seq, (path, bp)) in enumerate(zip(path_seqs, sampled)):
                if i == 10:
                    typer.echo(f"  ... and {n_paths - 10} more", err=True)
                    break
                typer.echo(
                    f"  [{i + 1}] {len(seq):,} minimizers, "
                    f"{bp:,} bp, {len(path)} nodes",
                    err=True,
                )

            if paths_out is not None:
                with open(paths_out, "w") as paths_fh:
                    paths_fh.write(
                        "path_index\tminimizers\tnodes\tpath_length_bp\n"
                    )
                    for path_idx, (seq, (path, bp)) in enumerate(
                        zip(path_seqs, sampled), start=1,
                    ):
                        minimizers_col = ",".join(str(int(m)) for m in seq)
                        nodes_col = ",".join(
                            graph[node_idx].name + ("+" if fwd else "-")
                            for node_idx, fwd in path
                        )
                        paths_fh.write(
                            f"{path_idx}\t{minimizers_col}\t"
                            f"{nodes_col}\t{bp}\n"
                        )
                typer.echo(f"Paths written to {paths_out}", err=True)

        # --- Match reads to the top paths -------------------------------------
        if sampled:
            match_counts: dict[int, int] = {}
            all_insert_sizes_min: list[int] = []
            insert_out_fh = (
                open(insert_sizes_out, "w")  # noqa: WPS515
                if insert_sizes_out is not None
                else None
            )
            if insert_out_fh is not None:
                header_cols = (
                    "read1_index\tread2_index\t"
                    "insert_size_minimizers"
                    + ("\tinsert_size_bp" if bp_scale else "")
                )
                insert_out_fh.write(header_cols + "\n")

            mappings_out_fh = (
                open(read_mappings_out, "w")  # noqa: WPS515
                if read_mappings_out is not None
                else None
            )
            if mappings_out_fh is not None:
                mappings_out_fh.write("read_index\tpair\tpath_index\n")

            del sampled
            try:
                with Progress(*_PROGRESS_COLUMNS) as progress:
                    task = progress.add_task(
                        "Matching reads to paths", total=len(path_seqs),
                    )
                    for path_idx, seq in enumerate(path_seqs, start=1):
                        path_matches = read_matcher.search_path(seq)
                        for m in path_matches:
                            match_counts[m.read_id] = (
                                match_counts.get(m.read_id, 0) + 1
                            )
                            n_total_matches += 1
                            if mappings_out_fh is not None:
                                pair_num = (
                                    "1" if m.read_id % 2 == 0 else "2"
                                ) if m.read_id in read_index.pairs else "."
                                mappings_out_fh.write(
                                    f"{m.read_id + 1}\t"
                                    f"{pair_num}\t"
                                    f"{path_idx}\n"
                                )
                        for r1_id, r2_id, span in _path_pair_insert_sizes(
                            path_matches, read_index,
                        ):
                            all_insert_sizes_min.append(span)
                            if insert_out_fh is not None:
                                row = (
                                    f"{r1_id + 1}\t{r2_id + 1}\t{span}"
                                )
                                if bp_scale is not None:
                                    row += f"\t{span * bp_scale:.1f}"
                                insert_out_fh.write(row + "\n")
                        progress.advance(task)
            finally:
                if insert_out_fh is not None:
                    insert_out_fh.close()
                if mappings_out_fh is not None:
                    mappings_out_fh.close()

            n_matched_reads = len(match_counts)
            typer.echo(
                f"Reads matched to paths: {n_matched_reads:,} reads, "
                f"{n_total_matches:,} total occurrences",
                err=True,
            )

            ins_stats = _insert_size_stats(all_insert_sizes_min, bp_scale)
            if ins_stats:  # noqa: SIM102
                ms = ins_stats["minimizer_space"]
                assert isinstance(ms, dict)
                typer.echo(
                    f"\nInsert size (minimizer space, "
                    f"{ins_stats['observations']:,} observations):",
                    err=True,
                )
                typer.echo(
                    f"  min={ms['min']:,}  max={ms['max']:,}  "
                    f"mean={ms['mean']:.1f}  "
                    f"median={ms['median']:.1f}  "
                    f"std_dev={ms['std_dev']:.1f}",
                    err=True,
                )
                if "bp" in ins_stats:
                    bp_s = ins_stats["bp"]
                    assert isinstance(bp_s, dict)
                    typer.echo(
                        f"Insert size (base pairs, "
                        f"scale={bp_s['scale_bp_per_minimizer']:.3f} "
                        f"bp/minimizer):",
                        err=True,
                    )
                    typer.echo(
                        f"  min={bp_s['min']:.0f}  "
                        f"max={bp_s['max']:.0f}  "
                        f"mean={bp_s['mean']:.0f}  "
                        f"median={bp_s['median']:.0f}  "
                        f"std_dev={bp_s['std_dev']:.0f}",
                        err=True,
                    )
            else:
                typer.echo(
                    "No paired reads matched the same path; "
                    "insert size could not be estimated.",
                    err=True,
                )

    min_table: dict[int, str] | None = None
    if minimizer_table is not None:
        min_table = load_minimizer_table(minimizer_table)
        typer.echo(
            f"Minimizer table: {len(min_table):,} unique minimizers "
            f"from {minimizer_table}",
            err=True,
        )

    if json_out is not None:
        stats = _compute_summary(
            graph, path_lengths, weight_str, pair_distances,
        )
        del graph  # last use; release graph memory
        stats["gfa"] = str(gfa)
        if read_index is not None:
            stats["read_index_reads"] = read_index.n_reads
            stats["read_index_pairs"] = len(read_index.pairs) // 2
        if n_total_matches:
            stats["path_match_occurrences"] = n_total_matches
        if ins_stats:
            stats["insert_size_from_minimizers"] = ins_stats
        if min_table is not None:
            stats["unique_minimizers"] = len(min_table)
        json_out.write_text(json.dumps(stats, indent=2) + "\n")
        typer.echo(f"Stats written to {json_out}", err=True)


if __name__ == "__main__":
    app()
