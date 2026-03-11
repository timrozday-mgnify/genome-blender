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

import glob
import io
import json
import logging
import math
import random
import re
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated

import lz4.frame

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
        from_orient: Orientation of the source segment.
        to_orient: Orientation of the target segment.
        overlap: CIGAR overlap string (e.g. ``"100M"``).
    """

    from_orient: str
    to_orient: str
    overlap: str


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
    )
    graph.add_edge(from_idx, to_idx, link)


# ------------------------------------------------------------------
# rust-mdbg read minimizers
# ------------------------------------------------------------------


def load_read_minimizers(prefix: Path) -> dict[str, tuple[int, ...]]:
    """Load per-read minimizer IDs from rust-mdbg output files.

    Reads all ``{prefix}.*.read_minimizers`` LZ4-compressed TSV files
    produced by rust-mdbg ``--dump-read-minimizers``.  Each file has
    one tab-separated row per read: ``read_id\tids\tpositions`` where
    *ids* is a comma-separated list of NT-hash minimizer IDs (u64).

    Args:
        prefix: rust-mdbg output prefix (e.g. ``Path("rust_mdbg_out")``).

    Returns:
        Mapping of read name to ordered tuple of NT-hash minimizer IDs.
    """
    pattern = str(prefix) + ".*.read_minimizers"
    files = sorted(glob.glob(pattern))
    if not files:
        logger.warning(
            "No .read_minimizers files found for prefix: %s", prefix,
        )
        return {}

    records: dict[str, tuple[int, ...]] = {}
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Loading read minimizers", total=len(files),
        )
        for fpath in files:
            raw = lz4.frame.decompress(Path(fpath).read_bytes())
            for line in io.TextIOWrapper(
                io.BytesIO(raw), encoding="utf-8",
            ):
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) != 3:
                    continue
                read_id = parts[0]
                ids = tuple(int(x) for x in parts[1].split(",") if x)
                records[read_id] = ids
                logger.debug("Read name: %s", read_id)
            progress.advance(task)

    logger.info(
        "Loaded minimizers for %d reads from %s.*",
        len(records), prefix,
    )
    return records


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


def _template_name(name: str) -> str:
    """Strip paired-end suffix from *name* to get the template name.

    Args:
        name: Raw read name from a FASTQ/FASTA file.

    Returns:
        Template name shared by both mates of a pair.
    """
    return _PAIR_SUFFIX_RE.sub("", name.split()[0])


@dataclass
class ReadIndex:
    """Integer-keyed read registry for memory-efficient downstream use.

    Each read is assigned a compact integer ID so that index structures
    (Aho-Corasick output lists, pair maps, position tables) store small
    ints rather than variable-length strings.

    Attributes:
        name_to_id: Read name to compact integer ID.
        names: Inverse map — index is the read ID, value is the name.
        minimizers: Per-read ordered minimizer-ID tuples, indexed by
            read ID.
        pairs: Read ID to mate read ID for paired-end reads.  Unpaired
            reads have no entry.
    """

    name_to_id: dict[str, int]
    names: list[str]
    minimizers: list[tuple[int, ...]]
    pairs: dict[int, int]


def build_read_index(
    read_minimizers: dict[str, tuple[int, ...]],
) -> ReadIndex:
    """Build an integer-indexed read registry from a minimizer dict.

    Reads are assigned IDs in sorted name order for reproducibility.
    Paired-end mates are detected by stripping common name suffixes
    and grouping reads that share the same template name.

    Args:
        read_minimizers: Mapping of read name to ordered minimizer IDs
            as returned by :func:`load_read_minimizers`.

    Returns:
        A :class:`ReadIndex` with integer IDs, minimizer lists stored
        by index, and paired mates cross-linked.
    """
    sorted_names = sorted(read_minimizers)
    n_reads = len(sorted_names)

    name_to_id: dict[str, int] = {}
    names: list[str] = sorted_names
    minimizers: list[tuple[int, ...]] = []
    template_to_ids: dict[str, list[int]] = {}

    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task("Building read index", total=n_reads)
        for read_id, name in enumerate(sorted_names):
            name_to_id[name] = read_id
            minimizers.append(read_minimizers[name])
            tmpl = _template_name(name)
            logger.debug("Read name: %s  template: %s", name, tmpl)
            template_to_ids.setdefault(tmpl, []).append(read_id)
            progress.advance(task)

    pairs: dict[int, int] = {}
    for tmpl, ids in template_to_ids.items():
        if len(ids) == 2:
            pairs[ids[0]] = ids[1]
            pairs[ids[1]] = ids[0]
        elif len(ids) > 2:
            logger.warning(
                "Template %r has %d reads; skipping pair detection",
                tmpl, len(ids),
            )

    logger.info(
        "Read index: %d reads, %d paired templates",
        len(names), len(pairs) // 2,
    )
    return ReadIndex(
        name_to_id=name_to_id,
        names=names,
        minimizers=minimizers,
        pairs=pairs,
    )


# ------------------------------------------------------------------
# Segment minimizers
# ------------------------------------------------------------------


def _segment_minimizers(
    sequence: str,
    reverse_table: dict[str, int],
    l: int,
) -> tuple[int, ...]:
    """Extract minimizer IDs from a segment sequence by l-mer lookup.

    Slides an l-mer window over *sequence* and looks up each l-mer in
    *reverse_table*.  Only l-mers present in the table (those selected
    as minimizers during graph construction) are included.  The table
    was built from HPC sequences, so *sequence* should also be in HPC
    space (as output by rust-mdbg).

    Args:
        sequence: Segment DNA sequence, typically HPC-encoded.
        reverse_table: lmer-string to NT-hash mapping (inverse of the
            minimizer_table file).
        l: l-mer length used during rust-mdbg graph construction.

    Returns:
        Ordered tuple of NT-hash minimizer IDs for this segment.
    """
    ids: list[int] = []
    for i in range(len(sequence) - l + 1):
        h = reverse_table.get(sequence[i: i + l])
        if h is not None:
            ids.append(h)
    return tuple(ids)


def build_segment_minimizers(
    graph: rx.PyGraph,
    reverse_table: dict[str, int],
    l: int,
) -> dict[str, tuple[int, ...]]:
    """Build a segment-name → minimizer-ID mapping for all graph nodes.

    Segments whose sequence is ``*`` (not stored in the GFA) are
    skipped with a warning and will be absent from path minimizer
    sequences.

    Args:
        graph: Parsed GFA graph.
        reverse_table: lmer → NT-hash mapping derived by inverting the
            minimizer_table (``{h: lmer for h, lmer in table.items()}``
            → ``{lmer: h ...}``).
        l: l-mer length.

    Returns:
        Dict mapping segment name to ordered tuple of minimizer IDs.
    """
    seg_min: dict[str, tuple[int, ...]] = {}
    n_missing = 0
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Extracting segment minimizers", total=graph.num_nodes(),
        )
        for idx in graph.node_indices():
            seg: Segment = graph[idx]
            if seg.sequence == "*":
                n_missing += 1
            else:
                seg_min[seg.name] = _segment_minimizers(
                    seg.sequence, reverse_table, l,
                )
            progress.advance(task)
    if n_missing:
        logger.warning(
            "%d segment(s) have sequence \"*\" and are excluded "
            "from path minimizer sequences",
            n_missing,
        )
    return seg_min


def path_minimizer_sequence(
    graph: rx.PyGraph,
    path: list[int],
    segment_minimizers: dict[str, tuple[int, ...]],
    k: int,
) -> tuple[int, ...]:
    """Build the minimizer sequence for a sampled graph path.

    Adjacent segments in a rust-mdbg path share *k* − 1 minimizers
    (the graph is a k-mer de Bruijn graph over minimizer sequences).
    The overlapping prefix of each segment after the first is therefore
    dropped before concatenation.

    Args:
        graph: Parsed GFA graph.
        path: Ordered list of node indices forming the path.
        segment_minimizers: Precomputed segment-name → minimizer IDs
            from :func:`build_segment_minimizers`.
        k: rust-mdbg k-mer size (overlap between adjacent segments is
            *k* − 1 minimizers).

    Returns:
        Ordered tuple of minimizer IDs spanning the entire path.
    """
    result: list[int] = []
    overlap = k - 1
    for i, node_idx in enumerate(path):
        seg: Segment = graph[node_idx]
        ids = segment_minimizers.get(seg.name, ())
        result.extend(ids if i == 0 else ids[overlap:])
    return tuple(result)


# ------------------------------------------------------------------
# Aho-Corasick automaton over integer sequences
# ------------------------------------------------------------------


@dataclass
class _AcNode:
    """Single node in the Aho-Corasick trie."""

    goto: dict[int, int] = field(default_factory=dict)
    failure: int = 0
    # Each entry: (pattern_id, pattern_length) emitted at this state.
    output: list[tuple[int, int]] = field(default_factory=list)


class AhoCorasick:
    """Aho-Corasick automaton for exact substring matching.

    Operates over arbitrary integer sequences (e.g. NT-hash minimizer
    IDs) rather than characters.  Trie transitions use dicts so the
    integer alphabet can be arbitrarily sparse.

    Usage::

        ac = AhoCorasick()
        for read_id, mids in enumerate(index.minimizers):
            ac.add_pattern(mids, read_id)
        ac.build()
        matches = ac.search(path_minimizer_sequence(...))
    """

    def __init__(self) -> None:
        self._nodes: list[_AcNode] = [_AcNode()]

    def _new_node(self) -> int:
        self._nodes.append(_AcNode())
        return len(self._nodes) - 1

    def add_pattern(self, pattern: tuple[int, ...], pattern_id: int) -> None:
        """Insert *pattern* into the trie, tagged with *pattern_id*.

        Args:
            pattern: Sequence of integer minimizer IDs.
            pattern_id: Caller-assigned identifier emitted on a match
                (use the read's integer ID from :class:`ReadIndex`).
        """
        state = 0
        for symbol in pattern:
            if symbol not in self._nodes[state].goto:
                self._nodes[state].goto[symbol] = self._new_node()
            state = self._nodes[state].goto[symbol]
        self._nodes[state].output.append((pattern_id, len(pattern)))

    def build(self) -> None:
        """Compute failure links and propagate outputs via BFS.

        Must be called once after all :meth:`add_pattern` calls and
        before any :meth:`search` calls.
        """
        q: deque[int] = deque()
        for child in self._nodes[0].goto.values():
            self._nodes[child].failure = 0
            q.append(child)

        while q:
            state = q.popleft()
            node = self._nodes[state]
            for symbol, next_state in node.goto.items():
                child = self._nodes[next_state]
                fall = node.failure
                while fall != 0 and symbol not in self._nodes[fall].goto:
                    fall = self._nodes[fall].failure
                child.failure = self._nodes[fall].goto.get(symbol, 0)
                if child.failure == next_state:
                    child.failure = 0
                child.output = (
                    child.output + self._nodes[child.failure].output
                )
                q.append(next_state)

    def search(
        self,
        text: tuple[int, ...],
    ) -> list[tuple[int, int, int]]:
        """Find all pattern occurrences in *text*.

        Args:
            text: Sequence of integer minimizer IDs to search.

        Returns:
            List of ``(start, end, pattern_id)`` tuples where
            ``text[start:end]`` matches the pattern for *pattern_id*.
            Results are ordered by end position.
        """
        state = 0
        matches: list[tuple[int, int, int]] = []
        for pos, symbol in enumerate(text):
            while state != 0 and symbol not in self._nodes[state].goto:
                state = self._nodes[state].failure
            state = self._nodes[state].goto.get(symbol, 0)
            for pat_id, pat_len in self._nodes[state].output:
                matches.append((pos - pat_len + 1, pos + 1, pat_id))
        return matches


def build_aho_corasick(index: ReadIndex) -> AhoCorasick:
    """Build an Aho-Corasick automaton from all reads in *index*.

    Each read's integer ID is used as the pattern ID so match results
    can be resolved back to read names and pairs without storing strings
    in the automaton.  Reads with empty minimizer sequences are skipped.

    Args:
        index: Populated :class:`ReadIndex`.

    Returns:
        A compiled :class:`AhoCorasick` automaton ready for searching.
    """
    ac = AhoCorasick()
    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Building Aho-Corasick", total=len(index.names),
        )
        for read_id, minimizers in enumerate(index.minimizers):
            if minimizers:
                ac.add_pattern(minimizers, read_id)
            progress.advance(task)
    ac.build()
    return ac


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
    path_min_seq: tuple[int, ...],
    ac: AhoCorasick,
) -> list[PathMatch]:
    """Find all reads whose minimizer sequences occur in *path_min_seq*.

    Args:
        path_min_seq: Concatenated minimizer sequence for a sampled
            graph path, from :func:`path_minimizer_sequence`.
        ac: Compiled :class:`AhoCorasick` automaton built from all reads.

    Returns:
        List of :class:`PathMatch` records, one per occurrence.
    """
    return [
        PathMatch(read_id=pat_id, path_start=start, path_end=end)
        for start, end, pat_id in ac.search(path_min_seq)
    ]


# ------------------------------------------------------------------
# Path sampling
# ------------------------------------------------------------------

_CIGAR_RE = re.compile(r"(\d+)[MIDNSHP=X]")


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
) -> list[int]:
    """Sample a random simple path by greedy random walk.

    Start from a randomly chosen leaf node, then repeatedly
    extend to an unvisited neighbour, chosen according to
    *weight_mode*:

    - ``"kmer"`` — weight by destination node's k-mer count
      (``Segment.kmer_count``); falls back to 1.0 when the
      tag is absent.
    - ``"overlap"`` — weight by the overlap length (sum of
      all lengths in the CIGAR string on the connecting edge).
    - ``"unweighted"`` — uniform random choice.

    ``graph.adj(node)`` returns ``{neighbour_idx: edge_data}``
    which provides both the neighbour index and the ``Link``
    edge data in one call, making all weight modes efficient.

    Args:
        graph: The graph to walk.
        start_nodes: Pool of nodes to pick a start from.
        weight_mode: One of ``"kmer"``, ``"overlap"``,
            ``"unweighted"``.

    Returns:
        Ordered list of node indices forming the path.
    """
    pick = _PICK_FN.get(weight_mode, _pick_uniformly)
    start = random.choice(start_nodes)
    path = [start]
    visited = {start}

    while True:
        neighbours = {
            n: edge
            for n, edge in graph.adj(path[-1]).items()
            if n not in visited
        }
        if not neighbours:
            break
        path.append(pick(graph, list(neighbours.keys()), neighbours))
        visited.add(path[-1])

    return path


def _iter_sampled_paths(
    graph: rx.PyGraph,
    n_samples: int,
    weight_mode: str = "kmer",
) -> Iterator[tuple[list[int], int]]:
    """Yield ``(path, bp_length)`` for *n_samples* random simple paths.

    Walks start from leaf nodes (degree == 1).  If the graph has no
    leaves, all nodes are used as start candidates instead.

    Args:
        graph: The assembled sequence graph.
        n_samples: Number of paths to sample.
        weight_mode: Neighbour selection weighting —
            ``"kmer"``, ``"overlap"``, or ``"unweighted"``.

    Yields:
        ``(path, bp_length)`` where *path* is an ordered list of node
        indices and *bp_length* is the sum of their segment lengths.
    """
    leaves = leaf_nodes(graph)
    if leaves:
        start_nodes = leaves
    else:
        logger.warning(
            "Graph has no leaf nodes (all degrees > 1); "
            "using all nodes as start candidates."
        )
        start_nodes = list(graph.node_indices())

    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task("Sampling paths", total=n_samples)
        for _ in range(n_samples):
            path = _random_simple_path(graph, start_nodes, weight_mode)
            bp = sum(graph[idx].length for idx in path)
            yield path, bp
            progress.advance(task)


def sample_path_lengths(
    graph: rx.PyGraph,
    n_samples: int,
    weight_mode: str = "kmer",
) -> list[int]:
    """Sample random simple paths and return their bp lengths.

    Args:
        graph: The assembled sequence graph.
        n_samples: Number of paths to sample.
        weight_mode: Neighbour selection weighting —
            ``"kmer"``, ``"overlap"``, or ``"unweighted"``.

    Returns:
        List of path lengths in base pairs.
    """
    return [bp for _, bp in _iter_sampled_paths(graph, n_samples, weight_mode)]


def sample_paths(
    graph: rx.PyGraph,
    n_samples: int,
    weight_mode: str = "kmer",
) -> list[tuple[list[int], int]]:
    """Sample random simple paths and return them with their bp lengths.

    Like :func:`sample_path_lengths` but also returns the path node
    sequences needed for minimizer matching.

    Args:
        graph: The assembled sequence graph.
        n_samples: Number of paths to sample.
        weight_mode: Neighbour selection weighting —
            ``"kmer"``, ``"overlap"``, or ``"unweighted"``.

    Returns:
        List of ``(path, bp_length)`` pairs.
    """
    return list(_iter_sampled_paths(graph, n_samples, weight_mode))


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
    path: list[int],
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
        path: Ordered list of node indices forming the path.
        pair_seg_indices: List of ``(R1_node_idx, R2_node_idx)``
            pairs to search for.

    Returns:
        List of bp distances, one per pair found in the path.
    """
    node_pos = {node: i for i, node in enumerate(path)}
    distances: list[int] = []

    for seg1_idx, seg2_idx in pair_seg_indices:
        if seg1_idx not in node_pos or seg2_idx not in node_pos:
            continue
        i = node_pos[seg1_idx]
        j = node_pos[seg2_idx]
        if i > j:
            i, j = j, i
        dist = sum(graph[path[k]].length for k in range(i, j + 1))
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
    result["connected_components"] = len(components)
    result["largest_component_nodes"] = component_sizes[0]
    if len(component_sizes) > 1:
        result["smallest_component_nodes"] = component_sizes[-1]

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

    print(f"Connected components: {s['connected_components']}")
    print(f"Largest component:  {s['largest_component_nodes']} nodes")
    if "smallest_component_nodes" in s:
        print(
            f"Smallest component: "
            f"{s['smallest_component_nodes']} nodes"
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
    minimizer_table: Annotated[Path | None, typer.Option(
        "--minimizer-table",
        help="Path to the rust-mdbg {prefix}.minimizer_table file. "
             "Required for read-to-path matching via Aho-Corasick; "
             "used to reconstruct per-segment minimizer sequences",
        exists=True,
        dir_okay=False,
    )] = None,
    k: Annotated[int, typer.Option(
        "--k",
        help="rust-mdbg k-mer size used during graph construction. "
             "Controls the minimizer overlap between adjacent segments "
             "(overlap = k - 1) when building path minimizer sequences",
    )] = 7,
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
    sampled: list[tuple[list[int], int]] | None = None
    path_lengths: list[int] | None = None

    if no_sample:
        logger.debug("Path sampling skipped by --no-sample.")
    elif graph.num_nodes() == 0:
        typer.echo(
            "Graph has no nodes; skipping path sampling.",
            err=True,
        )
    else:
        sampled = sample_paths(graph, samples, weight_str)
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
    ac: AhoCorasick | None = None
    n_total_matches = 0

    if read_minimizers_prefix is not None:
        raw_minimizers = load_read_minimizers(read_minimizers_prefix)
        read_index = build_read_index(raw_minimizers)
        typer.echo(
            f"Read index: {len(read_index.names):,} reads, "
            f"{len(read_index.pairs) // 2:,} paired templates",
            err=True,
        )
        ac = build_aho_corasick(read_index)
        typer.echo(
            f"Aho-Corasick automaton: {len(ac._nodes):,} states",
            err=True,
        )

        if minimizer_table is not None and sampled:
            table = load_minimizer_table(minimizer_table)
            l = len(next(iter(table.values())))  # infer l from table
            reverse_table: dict[str, int] = {
                lmer: h for h, lmer in table.items()
            }
            seg_min = build_segment_minimizers(graph, reverse_table, l)
            typer.echo(
                f"Segment minimizers: {len(seg_min):,} segments "
                f"with sequences (l={l})",
                err=True,
            )

            match_counts: dict[int, int] = {}  # read_id → hit count
            with Progress(*_PROGRESS_COLUMNS) as progress:
                task = progress.add_task(
                    "Matching reads to paths", total=len(sampled),
                )
                for path, _ in sampled:
                    seq = path_minimizer_sequence(
                        graph, path, seg_min, k,
                    )
                    for m in match_reads_to_path(seq, ac):
                        match_counts[m.read_id] = (
                            match_counts.get(m.read_id, 0) + 1
                        )
                        n_total_matches += 1
                    progress.advance(task)

            n_matched_reads = len(match_counts)
            typer.echo(
                f"Reads matched to paths: {n_matched_reads:,} reads, "
                f"{n_total_matches:,} total occurrences",
                err=True,
            )
        elif minimizer_table is None and read_minimizers_prefix is not None:
            typer.echo(
                "Provide --minimizer-table to enable read-to-path "
                "Aho-Corasick matching.",
                err=True,
            )

    if json_out is not None:
        stats = _compute_summary(
            graph, path_lengths, weight_str, pair_distances,
        )
        stats["gfa"] = str(gfa)
        if read_index is not None:
            stats["read_index_reads"] = len(read_index.names)
            stats["read_index_pairs"] = len(read_index.pairs) // 2
        if n_total_matches:
            stats["path_match_occurrences"] = n_total_matches
        json_out.write_text(json.dumps(stats, indent=2) + "\n")
        typer.echo(f"Stats written to {json_out}", err=True)


if __name__ == "__main__":
    app()
