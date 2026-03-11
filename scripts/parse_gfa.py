#!/usr/bin/env python3
"""Parse a GFA file into a rustworkx graph and report properties.

Usage::

    python scripts/parse_gfa.py assembly.gfa
    python scripts/parse_gfa.py assembly.gfa --samples 5000
    python scripts/parse_gfa.py assembly.gfa --no-sample
    python scripts/parse_gfa.py assembly.gfa --paired-end --pe-bam reads.bam
"""

from __future__ import annotations

import logging
import math
import random
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Annotated

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


def sample_path_lengths(
    graph: rx.PyGraph,
    n_samples: int,
    weight_mode: str = "kmer",
) -> list[int]:
    """Sample random simple paths and return their bp lengths.

    Walks start from leaf nodes (degree == 1) so that each
    path travels from a graph boundary inward.  If the graph
    has no leaves (e.g. a pure cycle), all nodes are used as
    start candidates instead.

    Each path length is the sum of segment lengths along the
    path.

    Args:
        graph: The assembled sequence graph.
        n_samples: Number of paths to sample.
        weight_mode: Neighbour selection weighting —
            ``"kmer"``, ``"overlap"``, or ``"unweighted"``.

    Returns:
        List of path lengths in base pairs.
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

    bp_lengths: list[int] = []

    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Sampling paths", total=n_samples,
        )
        for _ in range(n_samples):
            path = _random_simple_path(
                graph, start_nodes, weight_mode,
            )
            bp = sum(graph[idx].length for idx in path)
            bp_lengths.append(bp)
            progress.advance(task)

    return bp_lengths


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
    n_nodes = graph.num_nodes()
    n_edges = graph.num_edges()

    print(f"Nodes (segments): {n_nodes}")
    print(f"Edges (links):    {n_edges}")

    if n_nodes == 0:
        return

    # Connected components
    components = rx.connected_components(graph)
    component_sizes = sorted(
        (len(c) for c in components), reverse=True,
    )
    print(f"Connected components: {len(components)}")
    print(
        f"Largest component:  {component_sizes[0]} nodes",
    )
    if len(component_sizes) > 1:
        print(
            f"Smallest component: "
            f"{component_sizes[-1]} nodes"
        )

    # Segment length statistics
    segments: list[Segment] = [
        graph[idx] for idx in graph.node_indices()
    ]
    lengths = [s.length for s in segments]
    total_bp = sum(lengths)
    print(f"Total assembly span: {total_bp:,} bp")
    print(
        f"Segment lengths:  "
        f"min={min(lengths):,}, "
        f"max={max(lengths):,}, "
        f"mean={total_bp / n_nodes:,.0f}"
    )

    # Degree statistics
    degrees = [
        graph.degree(idx) for idx in graph.node_indices()
    ]
    print(
        f"Node degrees:     "
        f"min={min(degrees)}, "
        f"max={max(degrees)}, "
        f"mean={sum(degrees) / n_nodes:.1f}"
    )

    # K-mer count statistics (if available)
    kmer_counts = [
        s.kmer_count for s in segments
        if s.kmer_count is not None
    ]
    if kmer_counts:
        print(
            f"K-mer counts:     "
            f"min={min(kmer_counts):,}, "
            f"max={max(kmer_counts):,}, "
            f"mean={sum(kmer_counts) / len(kmer_counts):,.0f}"
        )

    # Sampled path length statistics
    if path_lengths is not None:
        n = len(path_lengths)
        mean = sum(path_lengths) / n
        variance = (
            sum((x - mean) ** 2 for x in path_lengths) / n
        )
        print(
            f"\nSampled path lengths "
            f"({n:,} samples, {weight_mode} weights, bp):"
        )
        print(f"  min:      {min(path_lengths):,}")
        print(f"  max:      {max(path_lengths):,}")
        print(f"  mean:     {mean:,.0f}")
        print(f"  std dev:  {math.sqrt(variance):,.0f}")
        print(f"  variance: {variance:,.0f}")

    # Paired-end insert size estimates
    if pair_distances is not None:
        if not pair_distances:
            print("\nPaired-end insert size: no pairs found in sampled paths.")
        else:
            n = len(pair_distances)
            mean = sum(pair_distances) / n
            variance = (
                sum((x - mean) ** 2 for x in pair_distances) / n
            )
            print(f"\nEstimated insert sizes ({n:,} pairs, bp):")
            print(f"  min:      {min(pair_distances):,}")
            print(f"  max:      {max(pair_distances):,}")
            print(f"  mean:     {mean:,.0f}")
            print(f"  variance: {variance:,.0f}")


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
    verbose: Annotated[bool, typer.Option(
        "--verbose/--no-verbose",
        help="Enable debug logging",
    )] = False,
) -> None:
    """Parse a GFA file into a graph and report properties."""
    if paired_end and pe_bam is None:
        raise typer.BadParameter(
            "--pe-bam PATH is required when "
            "--paired-end is set."
        )

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    graph = parse_gfa(gfa)

    weight_str = weight.value

    path_lengths: list[int] | None = None
    if no_sample:
        logger.debug("Path sampling skipped by --no-sample.")
    elif graph.num_nodes() == 0:
        typer.echo(
            "Graph has no nodes; skipping path sampling.",
            err=True,
        )
    else:
        path_lengths = sample_path_lengths(
            graph, samples, weight_str,
        )

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


if __name__ == "__main__":
    app()
