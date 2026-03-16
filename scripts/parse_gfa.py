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
import gzip
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

import lz4.frame
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

# Binary format: LZ4-compressed
# v1 header = b"RMBG\x01"; per-record: u32 LE name_len | name bytes | u32 LE n | n×u64 LE ids | n×u64 LE pos
# v2 header = b"RMBG\x02"; per-record: u32 LE read_index | u32 LE n | n×u32 LE compact_ids | n×u32 LE pos
_RM_MAGIC = b"RMBG"
_RM_HDR_LEN = 5  # 4 magic + 1 version

# Compact minimizer map format: b"RMCM\x01" | u32 LE n | n×u64 LE hashes (indexed by compact_id)
_RMCM_MAGIC = b"RMCM\x01"


def _load_compact_map(path: Path) -> np.ndarray:
    """Load a compact minimizer map file; return array of u64 hashes indexed by compact u32 ID."""
    raw = path.read_bytes()
    if raw[:5] != _RMCM_MAGIC:
        raise ValueError(f"Not a compact minimizer map file: {path}")
    (n,) = struct.unpack_from("<I", raw, 5)
    return np.frombuffer(raw, dtype="<u8", count=n, offset=9)


def _maybe_load_compact_map(prefix: Path) -> np.ndarray | None:
    """Load compact map for *prefix* if the file exists, else return None."""
    path = Path(str(prefix) + ".compact_map")
    if path.exists():
        cm = _load_compact_map(path)
        logger.info("Loaded compact minimizer map: %d entries from %s", len(cm), path)
        return cm
    return None


def _iter_tsv_rm(raw: bytes) -> Iterator[tuple[str, tuple[int, ...]]]:
    """Yield ``(name, minimizer_ids)`` from legacy LZ4-TSV bytes."""
    for line in io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        yield parts[0], tuple(int(x) for x in parts[1].split(",") if x)


def _iter_binary_rm(
    raw: bytes,
    compact_map: np.ndarray | None = None,
) -> Iterator[tuple[str, tuple[int, ...]]]:
    """Yield ``(name, minimizer_ids)`` from binary-format bytes.

    Supports v1 (name-string records) and v2 (compact-id records).  For v2,
    *compact_map* must be provided to expand compact u32 IDs back to u64 hashes.
    """
    mv = memoryview(raw)
    version = raw[4]
    offset = _RM_HDR_LEN
    total = len(raw)
    while offset < total:
        if version == 1:
            (name_len,) = struct.unpack_from("<I", mv, offset)
            offset += 4
            name = bytes(mv[offset: offset + name_len]).decode()
            offset += name_len
            (n,) = struct.unpack_from("<I", mv, offset)
            offset += 4
            ids: tuple[int, ...] = struct.unpack_from(f"<{n}Q", mv, offset)
            offset += n * 8 + n * 8  # skip positions (not needed here)
            yield name, ids
        else:  # version == 2
            (read_index,) = struct.unpack_from("<I", mv, offset)
            offset += 4
            (n,) = struct.unpack_from("<I", mv, offset)
            offset += 4
            compact_ids: tuple[int, ...] = struct.unpack_from(f"<{n}I", mv, offset)
            offset += n * 4
            offset += n * 4  # skip positions
            if compact_map is not None:
                ids = tuple(int(compact_map[cid]) for cid in compact_ids)
            else:
                ids = compact_ids  # fallback: emit compact ids as-is
            yield str(read_index), ids


def _iter_binary_rm_names(raw: bytes) -> Iterator[str]:
    """Yield read names only from binary-format bytes, skipping minimizer data."""
    mv = memoryview(raw)
    version = raw[4]
    offset = _RM_HDR_LEN
    total = len(raw)
    while offset < total:
        if version == 1:
            (name_len,) = struct.unpack_from("<I", mv, offset)
            offset += 4
            name = bytes(mv[offset: offset + name_len]).decode()
            offset += name_len
            (n,) = struct.unpack_from("<I", mv, offset)
            offset += 4 + n * 16  # skip ids (n×8) + positions (n×8)
            yield name
        else:  # version == 2
            (read_index,) = struct.unpack_from("<I", mv, offset)
            offset += 4
            (n,) = struct.unpack_from("<I", mv, offset)
            offset += 4 + n * 8  # skip compact_ids (n×4) + positions (n×4)
            yield str(read_index)


def _iter_tsv_rm_names(raw: bytes) -> Iterator[str]:
    """Yield read names only from legacy LZ4-TSV bytes."""
    for line in io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if parts:
            yield parts[0]


def _iter_fastx_names(path: Path) -> Iterator[str]:
    """Yield the first whitespace-delimited token from each sequence header.

    Handles both FASTQ (records starting with ``@``) and FASTA (records
    starting with ``>``).  Gzip-compressed files are detected automatically
    from a ``.gz`` suffix.  Format is auto-detected from the first non-empty
    line.

    Args:
        path: Path to a FASTQ or FASTA file, optionally gzip-compressed.

    Yields:
        Read/sequence names (header token before the first space).
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        is_fastq: bool | None = None
        skip_remaining = 0
        for raw_line in fh:
            line = raw_line.rstrip("\n")
            if skip_remaining > 0:
                skip_remaining -= 1
                continue
            if is_fastq is None:
                if not line:
                    continue
                is_fastq = line.startswith("@")
            if is_fastq:
                if line.startswith("@"):
                    yield line[1:].split()[0]
                    skip_remaining = 3  # seq, +, qual
            else:
                if line.startswith(">"):
                    yield line[1:].split()[0]


def iter_read_minimizers(
    prefix: Path,
) -> Iterator[tuple[str, tuple[int, ...]]]:
    """Stream ``(name, minimizer_ids)`` pairs from all ``.read_minimizers`` files.

    Decompresses one file at a time so only one file's data resides in
    memory simultaneously.  Supports binary v1/v2 (``RMBG`` magic) and
    legacy LZ4-TSV formats, auto-detected per file.  For v2 files, the
    compact minimizer map (``{prefix}.compact_map``) is loaded once and
    used to expand compact u32 IDs back to u64 hashes.

    Args:
        prefix: rust-mdbg output prefix (e.g. ``Path("rust_mdbg_out")``).

    Yields:
        ``(read_name, minimizer_id_tuple)`` for each read record.
    """
    pattern = str(prefix) + ".*.read_minimizers"
    files = sorted(glob.glob(pattern))
    compact_map = _maybe_load_compact_map(prefix)
    for fpath in files:
        raw = lz4.frame.decompress(Path(fpath).read_bytes())
        if raw[:4] == _RM_MAGIC:
            yield from _iter_binary_rm(raw, compact_map)
        else:
            yield from _iter_tsv_rm(raw)


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
# Bare _1/_2 (underscore without R) are intentionally excluded: they are
# indistinguishable from accession suffixes (e.g. MGYG000290000_1).
_PAIR_SUFFIX_RE = re.compile(
    r"/R?[12]$"             # /1  /2  /R1  /R2
    r"|_R[12]$"             # _R1  _R2
    r"|\.R?[12]$"           # .1  .2  .R1  .R2
    r"|[ \t][12]:[^ \t]+$"  # CASAVA: " 1:N:0:BARCODE"
)

_PAIR_NUMBER_RE = re.compile(
    r"/R?([12])$"           # /1  /2  /R1  /R2
    r"|_R([12])$"           # _R1  _R2
    r"|\.R?([12])$"         # .1  .2  .R1  .R2
    r"|[ \t]([12]):[^ \t]+$"  # CASAVA: " 1:N:0:BARCODE"
)


def _template_name(name: str) -> str:
    """Strip paired-end suffix from *name* to get the template name.

    Args:
        name: Raw read name from a FASTQ/FASTA file.

    Returns:
        Template name shared by both mates of a pair.
    """
    return _PAIR_SUFFIX_RE.sub("", name.split()[0])


def _pair_number(name: str) -> str:
    """Return the mate number (``"1"`` or ``"2"``) encoded in *name*.

    Returns ``"."`` when no recognised paired-end suffix is found.

    Args:
        name: Raw read name from a FASTQ/FASTA file.

    Returns:
        ``"1"``, ``"2"``, or ``"."`` for unpaired reads.
    """
    m = _PAIR_NUMBER_RE.search(name)
    if m is None:
        return "."
    return next(g for g in m.groups() if g is not None)


@dataclass
class ReadIndex:
    """Integer-keyed read registry for memory-efficient downstream use.

    Each read is assigned a compact integer ID so that index structures
    (Aho-Corasick output lists, pair maps, position tables) store small
    ints rather than variable-length strings.

    Read names are written to a name-index file by
    :func:`build_read_index` and can be loaded back with
    :func:`load_name_index` when output requires them.  After the
    matcher is built, ``name_to_id`` should be cleared (``name_to_id.clear()``)
    to release the string keys from memory.

    Attributes:
        n_reads: Total number of reads indexed.
        name_to_id: Read name to compact integer ID.  Clear after the
            matcher is built to free string memory.
        pairs: Read ID to mate read ID for paired-end reads.  Unpaired
            reads have no entry.
    """

    n_reads: int
    name_to_id: dict[str, int]
    pairs: dict[int, int]


def _write_name_index(sorted_names: list[str], path: Path) -> None:
    """Write read names to an LZ4-compressed line-per-name file.

    Line number *i* (0-based) is the name for read ID *i*.

    Args:
        sorted_names: Read names in ascending sort order.
        path: Destination path for the compressed index file.
    """
    content = "\n".join(sorted_names).encode()
    path.write_bytes(lz4.frame.compress(content))


def load_name_index(path: Path) -> list[str]:
    """Load a name index written by :func:`_write_name_index`.

    Args:
        path: Path to the LZ4-compressed name-index file.

    Returns:
        List of read names where index *i* is the name for read ID *i*.
    """
    content = lz4.frame.decompress(path.read_bytes())
    return content.decode().splitlines()


def build_read_index(
    prefix: Path,
    name_index_path: Path,
    paired_interleaved: bool = False,
    reads_path: Path | None = None,
) -> ReadIndex:
    """Build an integer-indexed read registry from ``.read_minimizers`` files.

    Performs a names-only streaming pass — minimizer data is not loaded
    into memory.  Reads are assigned IDs in sorted name order.  The
    sorted name list is written to *name_index_path* (LZ4-compressed,
    one name per line) and then freed; only ``name_to_id`` and ``pairs``
    are retained in the returned :class:`ReadIndex`.

    For v2 binary files (``RMBG\\x02``), names are integer read indices.
    When *paired_interleaved* is ``True``, pairs are assigned arithmetically:
    odd-indexed reads (1, 3, 5, …) are paired with the next even-indexed read
    (2, 4, 6, …) — this matches interleaved paired-end input ordering.

    When *reads_path* is provided, read names are sourced from the FASTQ/FASTA
    file instead of from ``.read_minimizers`` files.

    Args:
        prefix: rust-mdbg output prefix (e.g. ``Path("rust_mdbg_out")``).
        name_index_path: Destination for the name-index file.
        paired_interleaved: Use arithmetic pairing for interleaved input.
        reads_path: Optional FASTQ or FASTA file (plain or ``.gz``) whose
            read names override the ``.read_minimizers`` name scan.

    Returns:
        A :class:`ReadIndex` with ``n_reads``, ``name_to_id``, and
        ``pairs``.  Call ``name_to_id.clear()`` once the matcher is
        built to release string memory.
    """
    all_names: list[str] = []

    if reads_path is not None:
        logger.info("Loading read names from %s", reads_path)
        all_names.extend(_iter_fastx_names(reads_path))
    else:
        pattern = str(prefix) + ".*.read_minimizers"
        files = sorted(glob.glob(pattern))
        if not files:
            logger.warning(
                "No .read_minimizers files found for prefix: %s", prefix,
            )
            _write_name_index([], name_index_path)
            return ReadIndex(n_reads=0, name_to_id={}, pairs={})

        with Progress(*_PROGRESS_COLUMNS) as progress:
            task = progress.add_task("Loading read names", total=len(files))
            for fpath in files:
                raw = lz4.frame.decompress(Path(fpath).read_bytes())
                iter_fn = (
                    _iter_binary_rm_names
                    if raw[:4] == _RM_MAGIC
                    else _iter_tsv_rm_names
                )
                all_names.extend(iter_fn(raw))
                progress.advance(task)

    sorted_names = sorted(all_names)
    del all_names  # free unsorted list before building the dict
    _write_name_index(sorted_names, name_index_path)
    logger.info("Name index written to %s (%d reads)", name_index_path, len(sorted_names))

    name_to_id: dict[str, int] = {
        name: i for i, name in enumerate(sorted_names)
    }

    pairs: dict[int, int] = {}
    if paired_interleaved:
        # Arithmetic pairing for interleaved input: 1↔2, 3↔4, 5↔6, …
        # Names are 1-based read indices (str); pair odd with the next even.
        try:
            int_ids = sorted(int(n) for n in sorted_names)
        except ValueError:
            logger.warning(
                "paired_interleaved=True but names are not integers; "
                "falling back to template-name pairing"
            )
            int_ids = []
        if int_ids:
            id_set = set(int_ids)
            for idx in int_ids:
                if idx % 2 == 1 and (idx + 1) in id_set:
                    r1 = name_to_id[str(idx)]
                    r2 = name_to_id[str(idx + 1)]
                    pairs[r1] = r2
                    pairs[r2] = r1
            logger.info(
                "Interleaved pairing: %d pairs from %d reads",
                len(pairs) // 2, len(sorted_names),
            )
    else:
        template_to_ids: dict[str, list[int]] = {}
        for read_id, name in enumerate(sorted_names):
            tmpl = _template_name(name)
            logger.debug("Read name: %s  template: %s", name, tmpl)
            template_to_ids.setdefault(tmpl, []).append(read_id)

        for tmpl, ids in template_to_ids.items():
            if len(ids) == 2:
                pairs[ids[0]] = ids[1]
                pairs[ids[1]] = ids[0]
            elif len(ids) > 2:
                logger.warning(
                    "Template %r has %d reads; skipping pair detection",
                    tmpl, len(ids),
                )
    del sorted_names  # free the sorted list; dict keys are the canonical copy

    n_reads = len(name_to_id)
    logger.info(
        "Read index: %d reads, %d paired templates",
        n_reads, len(pairs) // 2,
    )
    return ReadIndex(n_reads=n_reads, name_to_id=name_to_id, pairs=pairs)


# ------------------------------------------------------------------
# Segment minimizers
# ------------------------------------------------------------------


_SegMinDict = dict[str, tuple[np.ndarray, np.ndarray]]
"""Mapping of segment name to ``(fwd, rev)`` uint64 numpy arrays.

Both arrays are 1-D and contiguous.  ``rev`` is the reversed copy of
``fwd``, precomputed once so that path traversal never needs to reverse
at query time.
"""


def load_segment_minimizers(
    prefix: Path,
) -> tuple[_SegMinDict, int | None]:
    """Load per-segment minimizer IDs from rust-mdbg ``.sequences`` files.

    Reads all ``{prefix}.*.sequences`` LZ4-compressed files.  Each
    non-comment line has the format::

        node_name<TAB>[hash1, hash2, ...]<TAB>sequence<TAB>...

    The node name matches the GFA S-record segment name exactly.  The
    ``k`` value is extracted from the ``# k = N`` header comment in the
    first file that contains it.

    For each segment both the forward and the reversed minimizer arrays
    are precomputed and stored as contiguous ``uint64`` numpy arrays so
    that path construction never needs to perform a reversal at query
    time.

    Args:
        prefix: rust-mdbg output prefix (e.g. ``Path("rust_mdbg_out")``).

    Returns:
        Tuple of ``(segment_minimizers, k)`` where *segment_minimizers*
        maps each segment name to a ``(fwd, rev)`` pair of uint64 numpy
        arrays, and *k* is the k-mer size read from the file header
        (``None`` if the header line is absent).
    """
    pattern = str(prefix) + ".*.sequences"
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(
            "No .sequences files found for prefix: %s", prefix,
        )
        return {}, None

    seg_min: _SegMinDict = {}
    k: int | None = None

    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Loading segment minimizers", total=len(files),
        )
        for fpath in files:
            raw = lz4.frame.decompress(Path(fpath).read_bytes())
            for line in io.TextIOWrapper(
                io.BytesIO(raw), encoding="utf-8",
            ):
                line = line.rstrip("\n")
                if not line:
                    continue
                if line.startswith("# k = ") and k is None:
                    try:
                        k = int(line[6:].split()[0])
                    except ValueError:
                        pass
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                node_name = parts[0]
                bracket = parts[1].strip()
                if bracket.startswith("[") and bracket.endswith("]"):
                    fwd = np.fromiter(
                        (
                            int(x.strip())
                            for x in bracket[1:-1].split(",")
                            if x.strip()
                        ),
                        dtype=np.uint64,
                    )
                else:
                    fwd = np.empty(0, dtype=np.uint64)
                seg_min[node_name] = (fwd, fwd[::-1].copy())
            progress.advance(task)

    logger.info(
        "Loaded minimizers for %d segments from %s.* (k=%s)",
        len(seg_min), prefix, k,
    )
    return seg_min, k


def build_seg_min_index(
    graph: rx.PyGraph,
    seg_min: _SegMinDict,
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Build a node-index-keyed list for O(1) minimizer lookup.

    Returns a list where position *i* holds the ``(fwd, rev)`` arrays
    for graph node *i*, or ``None`` if the segment has no minimizer data.
    The list length equals the number of graph nodes, so lookups use
    a direct integer index rather than a string dict lookup.

    Args:
        graph: Parsed GFA graph.
        seg_min: Output of :func:`load_segment_minimizers`.

    Returns:
        List of ``(fwd, rev)`` array pairs indexed by graph node index,
        with ``None`` for segments absent from *seg_min*.
    """
    n = graph.num_nodes()
    index: list[tuple[np.ndarray, np.ndarray] | None] = [None] * n
    for node_idx in graph.node_indices():
        seg: Segment = graph[node_idx]
        pair = seg_min.get(seg.name)
        if pair is not None:
            index[node_idx] = pair
    return index


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
            arrays from :func:`build_seg_min_index`.
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


def _build_ordered_path_index(
    path_seqs: list[np.ndarray],
    k_levels: list[int],
) -> dict[int, dict[tuple[int, ...], dict[int, list[int]]]]:
    """Build a positional inverted index for ordered k-mer matching.

    For each k level and each k-mer that appears in any path, records the
    sorted list of positions where that k-mer occurs within each path.

    Args:
        path_seqs: Minimizer ID arrays for the sampled paths.
        k_levels: Distinct k values to index.

    Returns:
        Mapping ``k → kmer_tuple → {path_idx: [sorted positions]}``.
    """
    index: dict[int, dict[tuple[int, ...], dict[int, list[int]]]] = {}
    for k in k_levels:
        k_index: dict[tuple[int, ...], dict[int, list[int]]] = {}
        for path_idx, seq in enumerate(path_seqs):
            n = len(seq)
            if n < k:
                continue
            for pos in range(n - k + 1):
                kmer = tuple(int(m) for m in seq[pos:pos + k])
                k_index.setdefault(kmer, {}).setdefault(path_idx, []).append(pos)
        index[k] = k_index
    return index


def _ordered_match_read(
    read_mids: tuple[int, ...],
    path_index: dict[int, dict[tuple[int, ...], dict[int, list[int]]]],
    k_levels: list[int],
    read_id: int,
    path_lens: list[int],
) -> list[tuple[int, PathMatch]]:
    """Return paths where read k-mers appear in the same order as in the read.

    Uses the k level whose size is the largest value ≤ the number of read
    minimizers (floor assignment).  Candidate paths are those containing all
    read k-mers (set intersection); each candidate is then verified by a
    greedy left-to-right scan that requires path positions to be strictly
    increasing (subsequence check).

    Args:
        read_mids: Ordered minimizer IDs for the read.
        path_index: Positional inverted index from
            :func:`_build_ordered_path_index`.
        k_levels: Sorted list of indexed k values.
        read_id: Integer read ID to embed in match records.
        path_lens: Minimizer lengths of each path (for ``PathMatch`` bounds).

    Returns:
        List of ``(path_idx, PathMatch)`` for every ordered occurrence.
    """
    k = _floor_k_level(len(read_mids), k_levels)
    if k is None:
        return []
    n = len(read_mids)
    read_kmers = [tuple(read_mids[i:i + k]) for i in range(n - k + 1)]
    if not read_kmers:
        return []
    k_index = path_index.get(k, {})

    # Candidate paths: those that contain every read k-mer.
    candidate_paths: set[int] | None = None
    for km in read_kmers:
        hits = k_index.get(km)
        if hits is None:
            return []
        if candidate_paths is None:
            candidate_paths = set(hits.keys())
        else:
            candidate_paths &= hits.keys()
        if not candidate_paths:
            return []

    matches: list[tuple[int, PathMatch]] = []
    for path_idx in candidate_paths:
        # Greedy subsequence check: for each read k-mer in order, advance
        # to the smallest path position that is strictly after the last.
        last_pos = -1
        valid = True
        for km in read_kmers:
            positions = k_index[km][path_idx]
            idx = bisect.bisect_right(positions, last_pos)
            if idx >= len(positions):
                valid = False
                break
            last_pos = positions[idx]
        if valid:
            matches.append((
                path_idx,
                PathMatch(
                    read_id=read_id,
                    path_start=0,
                    path_end=path_lens[path_idx],
                ),
            ))
    return matches


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
    minimizer_iter_factory: Callable[[], Iterable[tuple[str, tuple[int, ...]]]],
    index: ReadIndex,
    eligible: set[int] | None = None,
    n_levels: int = 5,
) -> OrderedMatchMatcher:
    """Build an :class:`OrderedMatchMatcher` with a two-pass streaming approach.

    Pass 1 scans read lengths to determine the k-level boundaries.
    The path positional index is then built for those levels.  Pass 2 streams
    reads again and checks each against the path index via ordered k-mer
    subsequence matching.

    Both forward and reversed orientations are tested; each
    ``(read_id, path_idx)`` pair is recorded at most once.

    Args:
        path_seqs: Minimizer sequences for the sampled paths.
        minimizer_iter_factory: Callable that returns a fresh
            ``(read_name, minimizer_ids)`` iterator on each invocation;
            called twice for the two streaming passes.
        index: :class:`ReadIndex` providing the name-to-ID mapping.
        eligible: Optional set of read IDs to include; others skipped.
        n_levels: Number of distinct k values to index, evenly spaced
            from the minimum to the maximum observed read length.

    Returns:
        An :class:`OrderedMatchMatcher` with all matches pre-computed,
        ready for sequential :meth:`~OrderedMatchMatcher.search_path` calls.
    """
    # -- Pass 1: determine the read-length range for k-level boundaries ------
    min_len = 10 ** 9
    max_len = 0
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Scanning read lengths (ordered-match)", total=index.n_reads,
        )
        for read_name, read_mids in minimizer_iter_factory():
            read_id = index.name_to_id.get(read_name)
            if read_id is None:
                continue
            n = len(read_mids)
            if n > 0 and (eligible is None or read_id in eligible):
                if n < min_len:
                    min_len = n
                if n > max_len:
                    max_len = n
            progress.advance(task)

    if max_len == 0:
        logger.warning("No eligible reads with minimizers found; returning empty matcher.")
        return OrderedMatchMatcher([[] for _ in path_seqs], 0, [], 0)

    k_levels = _compute_k_levels(min_len, max_len, n_levels)
    logger.info("Ordered-match k levels: %s", k_levels)

    # -- Build the path positional k-mer index for each level ----------------
    path_index = _build_ordered_path_index(path_seqs, k_levels)
    path_lens = [len(seq) for seq in path_seqs]

    # -- Pass 2: stream reads and match against the path index ---------------
    per_path: list[list[PathMatch]] = [[] for _ in path_seqs]
    seen_matches: set[tuple[int, int]] = set()
    n_total = 0
    n_indexed = 0
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Building ordered-match index", total=index.n_reads,
        )
        for read_name, read_mids in minimizer_iter_factory():
            read_id = index.name_to_id.get(read_name)
            if read_id is None:
                continue
            progress.advance(task)
            if eligible is not None and read_id not in eligible:
                continue
            if not read_mids:
                continue
            n_indexed += 1
            rev_mids: tuple[int, ...] = read_mids[::-1]
            orientations = (read_mids, rev_mids) if rev_mids != read_mids else (read_mids,)
            for mids in orientations:
                for path_idx, match in _ordered_match_read(
                    mids, path_index, k_levels, read_id, path_lens,
                ):
                    key = (read_id, path_idx)
                    if key not in seen_matches:
                        seen_matches.add(key)
                        per_path[path_idx].append(match)
                        n_total += 1

    logger.info(
        "Ordered-match: %d total match events across %d paths, %d reads indexed",
        n_total, len(path_seqs), n_indexed,
    )
    return OrderedMatchMatcher(per_path, n_total, k_levels, n_indexed)


# ------------------------------------------------------------------
# Pseudo-match matcher (unordered k-mer set membership)
# ------------------------------------------------------------------


def _build_pseudo_path_index(
    path_seqs: list[np.ndarray],
    k_levels: list[int],
) -> dict[int, dict[tuple[int, ...], set[int]]]:
    """Build an inverted k-mer index over sampled paths for each k level.

    Args:
        path_seqs: Minimizer ID arrays for each sampled path.
        k_levels: Distinct k values to index.

    Returns:
        Mapping ``k → kmer_tuple → {path_idx, …}`` covering every k-mer
        present in any sampled path at each level.
    """
    index: dict[int, dict[tuple[int, ...], set[int]]] = {}
    for k in k_levels:
        kmer_map: dict[tuple[int, ...], set[int]] = {}
        for path_idx, seq in enumerate(path_seqs):
            for i in range(len(seq) - k + 1):
                kmer: tuple[int, ...] = tuple(int(seq[i + j]) for j in range(k))
                kmer_map.setdefault(kmer, set()).add(path_idx)
        index[k] = kmer_map
    return index


def _pseudo_match_read(
    read_mids: tuple[int, ...],
    path_index: dict[int, dict[tuple[int, ...], set[int]]],
    k_levels: list[int],
    read_id: int,
    path_lens: list[int],
) -> list[tuple[int, PathMatch]]:
    """Find paths that contain every k-mer of *read_mids* (unordered).

    Selects the largest k ≤ len(read_mids) from k_levels, generates all
    consecutive k-mers from the read, then intersects the candidate path
    sets from the inverted index.  A path matches when it contains all
    k-mers; positional order within the path is not required.

    Args:
        read_mids: Tuple of minimizer IDs for the read.
        path_index: Inverted index from :func:`_build_pseudo_path_index`.
        k_levels: Sorted list of k values matching those in path_index.
        read_id: Integer read ID to embed in match records.
        path_lens: Minimizer-sequence length of each path, used as the
            placeholder ``path_end`` in returned :class:`PathMatch` records.

    Returns:
        List of ``(path_idx, PathMatch)`` for every matching path.
    """
    k = _floor_k_level(len(read_mids), k_levels)
    if k is None:
        return []
    kmer_map = path_index[k]
    candidate_paths: set[int] | None = None
    for i in range(len(read_mids) - k + 1):
        kmer: tuple[int, ...] = read_mids[i:i + k]
        paths_for_kmer = kmer_map.get(kmer)
        if not paths_for_kmer:
            return []
        if candidate_paths is None:
            candidate_paths = set(paths_for_kmer)
        else:
            candidate_paths &= paths_for_kmer
            if not candidate_paths:
                return []
    if candidate_paths is None:
        return []
    return [
        (path_idx, PathMatch(read_id=read_id, path_start=0, path_end=path_lens[path_idx]))
        for path_idx in candidate_paths
    ]


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
    minimizer_iter_factory: Callable[[], Iterable[tuple[str, tuple[int, ...]]]],
    index: ReadIndex,
    eligible: set[int] | None = None,
    n_levels: int = 5,
) -> PseudoMatchMatcher:
    """Build a :class:`PseudoMatchMatcher` with a two-pass streaming approach.

    Pass 1 scans read lengths to determine the k-level boundaries, which
    span from the minimum to the maximum observed eligible read length.
    The path k-mer index is then built for those levels.  Pass 2 streams
    reads again and checks each against the path index via unordered
    k-mer set membership.

    Both forward and reversed orientations are tested; each
    ``(read_id, path_idx)`` pair is recorded at most once.

    Args:
        path_seqs: Minimizer sequences for the sampled paths.
        minimizer_iter_factory: Callable that returns a fresh
            ``(read_name, minimizer_ids)`` iterator on each invocation;
            called twice for the two streaming passes.
        index: :class:`ReadIndex` providing the name-to-ID mapping.
        eligible: Optional set of read IDs to include; others skipped.
        n_levels: Number of distinct k values to index, evenly spaced
            from the minimum to the maximum observed read length.

    Returns:
        A :class:`PseudoMatchMatcher` with all matches pre-computed,
        ready for sequential :meth:`~PseudoMatchMatcher.search_path` calls.
    """
    # -- Pass 1: determine the read-length range for k-level boundaries ------
    min_len = 10 ** 9
    max_len = 0
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Scanning read lengths (pseudo-match)", total=index.n_reads,
        )
        for read_name, read_mids in minimizer_iter_factory():
            read_id = index.name_to_id.get(read_name)
            if read_id is None:
                continue
            n = len(read_mids)
            if n > 0 and (eligible is None or read_id in eligible):
                if n < min_len:
                    min_len = n
                if n > max_len:
                    max_len = n
            progress.advance(task)

    if max_len == 0:
        logger.warning("No eligible reads with minimizers found; returning empty matcher.")
        return PseudoMatchMatcher([[] for _ in path_seqs], 0, [], 0)

    k_levels = _compute_k_levels(min_len, max_len, n_levels)
    logger.info("Pseudo-match k levels: %s", k_levels)

    # -- Build the path k-mer index for each level ---------------------------
    path_index = _build_pseudo_path_index(path_seqs, k_levels)
    path_lens = [len(seq) for seq in path_seqs]

    # -- Pass 2: stream reads and match against the path index ---------------
    per_path: list[list[PathMatch]] = [[] for _ in path_seqs]
    seen_matches: set[tuple[int, int]] = set()
    n_total = 0
    n_indexed = 0
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Building pseudo-match index", total=index.n_reads,
        )
        for read_name, read_mids in minimizer_iter_factory():
            read_id = index.name_to_id.get(read_name)
            if read_id is None:
                continue
            progress.advance(task)
            if eligible is not None and read_id not in eligible:
                continue
            if not read_mids:
                continue
            n_indexed += 1
            rev_mids: tuple[int, ...] = read_mids[::-1]
            orientations = (read_mids, rev_mids) if rev_mids != read_mids else (read_mids,)
            for mids in orientations:
                for path_idx, match in _pseudo_match_read(
                    mids, path_index, k_levels, read_id, path_lens,
                ):
                    key = (read_id, path_idx)
                    if key not in seen_matches:
                        seen_matches.add(key)
                        per_path[path_idx].append(match)
                        n_total += 1

    logger.info(
        "Pseudo-match: %d total match events across %d paths, %d reads indexed",
        n_total, len(path_seqs), n_indexed,
    )
    return PseudoMatchMatcher(per_path, n_total, k_levels, n_indexed)


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
            from :func:`build_seg_min_index`.

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
            comp_min = sum(
                len(seg_min_index[idx][0])
                for idx in comp
                if idx < len(seg_min_index) and seg_min_index[idx] is not None
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
    reads: Annotated[Path | None, typer.Option(
        "--reads",
        help="FASTQ or FASTA file (plain or .gz) whose read names populate "
             "the read index; overrides the .read_minimizers name scan when "
             "provided alongside --read-minimizers",
        exists=True,
        dir_okay=False,
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
    min_paired_matches: Annotated[int, typer.Option(
        "--min-paired-matches",
        help="Keep only paths where at least this many paired-end read "
             "pairs (both mates mapped to the same path) are found. "
             "Applied after matching, before reporting. 0 disables "
             "the filter. Requires --read-minimizers.",
    )] = 5,
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

    if read_minimizers_prefix is not None:
        _name_index_path = Path(
            str(read_minimizers_prefix) + ".read_name_index"
        )
        read_index = build_read_index(
            read_minimizers_prefix, _name_index_path, interleaved_pairs, reads,
        )
        typer.echo(
            f"Read index: {read_index.n_reads:,} reads, "
            f"{len(read_index.pairs) // 2:,} paired templates",
            err=True,
        )

        # Precompute path sequences for the matcher.
        path_seqs: list[np.ndarray] = []
        if sampled:
            seg_min, k_from_file = load_segment_minimizers(
                read_minimizers_prefix,
            )
            effective_k = k_from_file if k_from_file is not None else k
            typer.echo(
                f"Segment minimizers: {len(seg_min):,} segments "
                f"(k={effective_k})",
                err=True,
            )

            seg_min_index = build_seg_min_index(graph, seg_min)
            bp_scale = _minimizer_to_bp_scale(graph, seg_min_index)
            typer.echo(
                f"bp/minimizer scale: "
                + (f"{bp_scale:.3f}" if bp_scale else "unavailable"),
                err=True,
            )

            if chosen_comps:
                typer.echo(
                    f"Chosen components ({len(chosen_comps)} total):",
                    err=True,
                )
                _report_chosen_components(graph, chosen_comps, seg_min_index)

            for path, _ in sampled:
                seq = path_minimizer_sequence(path, seg_min_index, effective_k)
                path_seqs.append(seq)
            del seg_min_index  # not accessed again after path sequences are built

        read_matcher: ReadMatcher
        if matcher == MatcherMode.exact:
            read_matcher = build_ordered_matcher(
                path_seqs,
                lambda: iter_read_minimizers(read_minimizers_prefix),
                read_index,
                n_levels=pseudo_match_levels,
            )
        else:
            read_matcher = build_pseudo_matcher(
                path_seqs,
                lambda: iter_read_minimizers(read_minimizers_prefix),
                read_index,
                n_levels=pseudo_match_levels,
            )
        typer.echo(read_matcher.describe(), err=True)
        read_index.name_to_id.clear()
        logger.info("name_to_id cleared; string keys released from memory")

        # --- Score all paths, filter by paired matches, rank, select top N ----
        # Matches are cached here so the reporting loop below reuses them.
        cached_matches: list[list[PathMatch]] = []
        path_pair_counts: list[int] = []
        if sampled:
            _scored: list[
                tuple[int, int, np.ndarray, tuple[_OrientedPath, int], list[PathMatch]]
            ] = []
            with Progress(*_PROGRESS_COLUMNS) as progress:
                task = progress.add_task(
                    "Scoring paths by paired matches", total=len(sampled),
                )
                for seq, entry in zip(path_seqs, sampled):
                    pm = match_reads_to_path(seq, read_matcher)
                    n_pairs = len(_path_pair_insert_sizes(pm, read_index))
                    _scored.append((n_pairs, entry[1], seq, entry, pm))
                    progress.advance(task)

            if min_paired_matches > 0:
                n_before = len(_scored)
                _scored = [t for t in _scored if t[0] >= min_paired_matches]
                n_removed = n_before - len(_scored)
                typer.echo(
                    f"Path filter (≥{min_paired_matches} paired matches): "
                    f"{len(_scored)} of {n_before} paths kept "
                    f"({n_removed} removed)",
                    err=True,
                )

            # Sort by paired match count descending; length descending as tiebreaker.
            _scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            if top_paths:
                _scored = _scored[:top_paths]

            path_seqs = [t[2] for t in _scored]
            sampled = [t[3] for t in _scored]
            cached_matches = [t[4] for t in _scored]
            path_pair_counts = [t[0] for t in _scored]
            del _scored

        # --- Report and write paths after filtering ----------------------------
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
            for i, (seq, (path, bp), n_pairs) in enumerate(
                zip(path_seqs, sampled, path_pair_counts)
            ):
                if i == 10:
                    typer.echo(
                        f"  ... and {n_paths - 10} more", err=True,
                    )
                    break
                typer.echo(
                    f"  [{i + 1}] {len(seq):,} minimizers, "
                    f"{bp:,} bp, {len(path)} nodes, "
                    f"{n_pairs} paired matches",
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
                        minimizers_col = ",".join(
                            str(int(m)) for m in seq
                        )
                        nodes_col = ",".join(
                            graph[node_idx].name + ("+" if fwd else "-")
                            for node_idx, fwd in path
                        )
                        paths_fh.write(
                            f"{path_idx}\t{minimizers_col}\t"
                            f"{nodes_col}\t{bp}\n"
                        )
                typer.echo(
                    f"Paths written to {paths_out}", err=True,
                )

        if sampled:
            # Load name strings from disk only if output files need them;
            # held only for the duration of the matching loop then released.
            _need_names = (
                read_mappings_out is not None or insert_sizes_out is not None
            )
            _output_names: list[str] = (
                load_name_index(_name_index_path) if _need_names else []
            )
            match_counts: dict[int, int] = {}  # read_id → hit count
            all_insert_sizes_min: list[int] = []
            insert_out_fh = (
                open(insert_sizes_out, "w")  # noqa: WPS515
                if insert_sizes_out is not None
                else None
            )
            if insert_out_fh is not None:
                header_cols = (
                    "read1_name\tread2_name\t"
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
                mappings_out_fh.write("read_name\tpair\tpath_index\n")

            del sampled  # content fully consumed above; free path-node lists
            try:
                with Progress(*_PROGRESS_COLUMNS) as progress:
                    task = progress.add_task(
                        "Matching reads to paths", total=len(cached_matches),
                    )
                    for path_idx, (seq, path_matches) in enumerate(
                        zip(path_seqs, cached_matches), start=1,
                    ):
                        for m in path_matches:
                            match_counts[m.read_id] = (
                                match_counts.get(m.read_id, 0) + 1
                            )
                            n_total_matches += 1
                            if mappings_out_fh is not None:
                                rname = _output_names[m.read_id]
                                mappings_out_fh.write(
                                    f"{rname}\t"
                                    f"{_pair_number(rname)}\t"
                                    f"{path_idx}\n"
                                )
                        for r1_id, r2_id, span in _path_pair_insert_sizes(
                            path_matches, read_index,
                        ):
                            all_insert_sizes_min.append(span)
                            if insert_out_fh is not None:
                                r1_name = _output_names[r1_id]
                                r2_name = _output_names[r2_id]
                                row = (
                                    f"{r1_name}\t{r2_name}\t{span}"
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
                del _output_names  # release name strings from memory

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
