#!/usr/bin/env python3
"""Parse a GFA file into a rustworkx graph and report properties.

Usage::

    python scripts/parse_gfa.py assembly.gfa
    python scripts/parse_gfa.py assembly.gfa --samples 5000
    python scripts/parse_gfa.py assembly.gfa --no-sample
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from pathlib import Path

import re

import click
import rustworkx as rx
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

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
) -> None:
    """Print a summary of graph properties to stdout.

    Args:
        graph: Parsed GFA graph.
        path_lengths: Sampled path lengths in bp, or ``None``
            if path sampling was skipped.
        weight_mode: Weight mode used for path sampling —
            shown in the output header.
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


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

@click.command()
@click.argument("gfa", type=click.Path(
    exists=True, dir_okay=False, path_type=Path,
))
@click.option(
    "-n", "--samples", default=1000, show_default=True,
    help="Number of random paths to sample.",
)
@click.option(
    "--no-sample", is_flag=True, default=False,
    help="Skip path sampling entirely.",
)
@click.option(
    "--weight",
    type=click.Choice(["kmer", "overlap", "unweighted"]),
    default="kmer",
    show_default=True,
    help="Neighbour selection weighting for random walks.",
)
@click.option(
    "-v", "--verbose", is_flag=True, default=False,
    help="Enable debug logging.",
)
def main(
    gfa: Path,
    samples: int,
    no_sample: bool,
    weight: str,
    verbose: bool,
) -> None:
    """Parse a GFA file into a graph and report properties."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    graph = parse_gfa(gfa)

    path_lengths: list[int] | None = None
    if no_sample:
        logger.debug("Path sampling skipped by --no-sample.")
    elif graph.num_nodes() == 0:
        click.echo(
            "Graph has no nodes; skipping path sampling.",
            err=True,
        )
    else:
        path_lengths = sample_path_lengths(graph, samples, weight)

    print_summary(graph, path_lengths, weight)


if __name__ == "__main__":
    main()
