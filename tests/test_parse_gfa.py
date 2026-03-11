"""Tests for scripts/parse_gfa.py."""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import rustworkx as rx
from typer.testing import CliRunner

# Import from the script as a module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from parse_gfa import (
    Link,
    Segment,
    WeightMode,
    _overlap_length,
    _parse_tags,
    _path_pair_distances,
    _pick_by_kmer,
    _pick_by_overlap,
    _pick_uniformly,
    _random_simple_path,
    _seg_name_index,
    app,
    leaf_nodes,
    longest_simple_path,
    parse_gfa,
    print_summary,
    sample_pair_distances,
    sample_path_lengths,
)


runner = CliRunner()


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

def _write_gfa(path: Path, content: str) -> Path:
    """Write GFA content to a file."""
    path.write_text(content)
    return path


@pytest.fixture()
def linear_gfa(tmp_path) -> Path:
    """A linear chain: s1 -- s2 -- s3."""
    return _write_gfa(
        tmp_path / "linear.gfa",
        "H\tVN:Z:1.0\n"
        "S\ts1\tACGTACGT\tKC:i:100\n"
        "S\ts2\tGCTAGCTA\tKC:i:200\n"
        "S\ts3\tTTAGAATT\tKC:i:50\n"
        "L\ts1\t+\ts2\t+\t4M\n"
        "L\ts2\t+\ts3\t+\t3M\n",
    )


@pytest.fixture()
def cycle_gfa(tmp_path) -> Path:
    """A 3-node cycle (no leaves)."""
    return _write_gfa(
        tmp_path / "cycle.gfa",
        "S\ta\tAAAA\n"
        "S\tb\tCCCC\n"
        "S\tc\tGGGG\n"
        "L\ta\t+\tb\t+\t*\n"
        "L\tb\t+\tc\t+\t*\n"
        "L\tc\t+\ta\t+\t*\n",
    )


@pytest.fixture()
def star_len_gfa(tmp_path) -> Path:
    """Segments with * sequence and LN tags."""
    return _write_gfa(
        tmp_path / "star.gfa",
        "S\tx\t*\tLN:i:500\n"
        "S\ty\t*\tLN:i:300\n"
        "L\tx\t+\ty\t+\t10M\n",
    )


@pytest.fixture()
def empty_gfa(tmp_path) -> Path:
    """An empty GFA (header only)."""
    return _write_gfa(
        tmp_path / "empty.gfa",
        "H\tVN:Z:1.0\n",
    )


@pytest.fixture()
def linear_graph(linear_gfa) -> rx.PyGraph:
    """Parsed graph from linear_gfa."""
    return parse_gfa(linear_gfa)


# ------------------------------------------------------------------ #
# Unit tests: _parse_tags
# ------------------------------------------------------------------ #

class TestParseTags:
    """Tests for _parse_tags."""

    def test_typical_tags(self) -> None:
        tags = _parse_tags(["KC:i:42", "LN:i:100"])
        assert tags == {"KC": "42", "LN": "100"}

    def test_empty(self) -> None:
        assert _parse_tags([]) == {}

    def test_malformed_ignored(self) -> None:
        tags = _parse_tags(["KC:i:42", "badfield", "LN:i:5"])
        assert tags == {"KC": "42", "LN": "5"}

    def test_string_tag(self) -> None:
        tags = _parse_tags(["VN:Z:1.0"])
        assert tags["VN"] == "1.0"


# ------------------------------------------------------------------ #
# Unit tests: _overlap_length
# ------------------------------------------------------------------ #

class TestOverlapLength:
    """Tests for _overlap_length."""

    def test_simple_match(self) -> None:
        assert _overlap_length("100M") == 100.0

    def test_star_returns_one(self) -> None:
        assert _overlap_length("*") == 1.0

    def test_mixed_cigar(self) -> None:
        assert _overlap_length("10M5I3D") == 18.0

    def test_empty_returns_one(self) -> None:
        assert _overlap_length("") == 1.0


# ------------------------------------------------------------------ #
# Unit tests: leaf_nodes
# ------------------------------------------------------------------ #

class TestLeafNodes:
    """Tests for leaf_nodes."""

    def test_linear_chain(self, linear_graph) -> None:
        leaves = leaf_nodes(linear_graph)
        names = {linear_graph[i].name for i in leaves}
        assert names == {"s1", "s3"}

    def test_cycle_has_no_leaves(self, cycle_gfa) -> None:
        graph = parse_gfa(cycle_gfa)
        assert leaf_nodes(graph) == []

    def test_empty_graph(self) -> None:
        graph: rx.PyGraph = rx.PyGraph()
        assert leaf_nodes(graph) == []


# ------------------------------------------------------------------ #
# Unit tests: pick functions
# ------------------------------------------------------------------ #

class TestPickFunctions:
    """Tests for _pick_by_kmer, _pick_by_overlap, _pick_uniformly."""

    def _make_graph(
        self,
    ) -> tuple[rx.PyGraph, list[int], dict[int, Link]]:
        """Build a small graph and return (graph, candidates, edges)."""
        g: rx.PyGraph = rx.PyGraph()
        i0 = g.add_node(Segment("a", "AAAA", 4, kmer_count=100))
        i1 = g.add_node(Segment("b", "CCCC", 4, kmer_count=1))
        edges: dict[int, Link] = {
            i0: Link("+", "+", "2M"),
            i1: Link("+", "+", "10M"),
        }
        return g, [i0, i1], edges

    def test_pick_by_kmer_prefers_high_count(self) -> None:
        random.seed(42)
        g, cands, edges = self._make_graph()
        counts = {c: 0 for c in cands}
        for _ in range(200):
            chosen = _pick_by_kmer(g, cands, edges)
            counts[chosen] += 1
        # Node 0 has kmer_count=100, node 1 has 1
        assert counts[cands[0]] > counts[cands[1]]

    def test_pick_by_overlap_prefers_large_overlap(self) -> None:
        random.seed(42)
        g, cands, edges = self._make_graph()
        counts = {c: 0 for c in cands}
        for _ in range(200):
            chosen = _pick_by_overlap(g, cands, edges)
            counts[chosen] += 1
        # Node 1 has 10M overlap vs node 0's 2M
        assert counts[cands[1]] > counts[cands[0]]

    def test_pick_uniformly_returns_valid(self) -> None:
        random.seed(42)
        g, cands, edges = self._make_graph()
        for _ in range(20):
            chosen = _pick_uniformly(g, cands, edges)
            assert chosen in cands


# ------------------------------------------------------------------ #
# Unit tests: _path_pair_distances
# ------------------------------------------------------------------ #

class TestPathPairDistances:
    """Tests for _path_pair_distances."""

    def _make_chain(self, lengths: list[int]) -> rx.PyGraph:
        g: rx.PyGraph = rx.PyGraph()
        prev = None
        for i, ln in enumerate(lengths):
            idx = g.add_node(
                Segment(f"s{i}", "A" * ln, ln),
            )
            if prev is not None:
                g.add_edge(prev, idx, Link("+", "+", "*"))
            prev = idx
        return g

    def test_same_segment(self) -> None:
        g = self._make_chain([100, 200, 300])
        path = [0, 1, 2]
        dists = _path_pair_distances(g, path, [(1, 1)])
        assert dists == [200]

    def test_adjacent_segments(self) -> None:
        g = self._make_chain([100, 200, 300])
        path = [0, 1, 2]
        dists = _path_pair_distances(g, path, [(0, 1)])
        assert dists == [300]  # 100 + 200

    def test_spanning_full_path(self) -> None:
        g = self._make_chain([100, 200, 300])
        path = [0, 1, 2]
        dists = _path_pair_distances(g, path, [(0, 2)])
        assert dists == [600]  # 100 + 200 + 300

    def test_pair_not_in_path(self) -> None:
        g = self._make_chain([100, 200, 300])
        path = [0, 1]
        dists = _path_pair_distances(g, path, [(0, 2)])
        assert dists == []

    def test_reversed_pair_order(self) -> None:
        g = self._make_chain([100, 200, 300])
        path = [0, 1, 2]
        dists = _path_pair_distances(g, path, [(2, 0)])
        assert dists == [600]


# ------------------------------------------------------------------ #
# Unit tests: _seg_name_index
# ------------------------------------------------------------------ #

class TestSegNameIndex:
    """Tests for _seg_name_index."""

    def test_maps_names_to_indices(self, linear_graph) -> None:
        idx = _seg_name_index(linear_graph)
        assert set(idx.keys()) == {"s1", "s2", "s3"}
        for name, node_idx in idx.items():
            assert linear_graph[node_idx].name == name


# ------------------------------------------------------------------ #
# Integration tests: parse_gfa
# ------------------------------------------------------------------ #

class TestParseGfa:
    """Integration tests for parse_gfa."""

    def test_linear_graph(self, linear_gfa) -> None:
        g = parse_gfa(linear_gfa)
        assert g.num_nodes() == 3
        assert g.num_edges() == 2

    def test_segments_have_correct_data(self, linear_gfa) -> None:
        g = parse_gfa(linear_gfa)
        names = {g[i].name for i in g.node_indices()}
        assert names == {"s1", "s2", "s3"}
        idx = _seg_name_index(g)
        assert g[idx["s1"]].length == 8
        assert g[idx["s1"]].kmer_count == 100

    def test_star_sequence_uses_ln(self, star_len_gfa) -> None:
        g = parse_gfa(star_len_gfa)
        idx = _seg_name_index(g)
        assert g[idx["x"]].length == 500
        assert g[idx["y"]].length == 300

    def test_cycle_graph(self, cycle_gfa) -> None:
        g = parse_gfa(cycle_gfa)
        assert g.num_nodes() == 3
        assert g.num_edges() == 3

    def test_empty_gfa(self, empty_gfa) -> None:
        g = parse_gfa(empty_gfa)
        assert g.num_nodes() == 0
        assert g.num_edges() == 0

    def test_comments_and_blank_lines(self, tmp_path) -> None:
        gfa = _write_gfa(
            tmp_path / "comments.gfa",
            "# comment line\n"
            "\n"
            "H\tVN:Z:1.0\n"
            "S\ts1\tACGT\n"
            "\n",
        )
        g = parse_gfa(gfa)
        assert g.num_nodes() == 1

    def test_link_to_unknown_segment(self, tmp_path) -> None:
        gfa = _write_gfa(
            tmp_path / "bad_link.gfa",
            "S\ts1\tACGT\n"
            "L\ts1\t+\ts_missing\t+\t*\n",
        )
        g = parse_gfa(gfa)
        assert g.num_nodes() == 1
        assert g.num_edges() == 0


# ------------------------------------------------------------------ #
# Integration tests: sample_path_lengths
# ------------------------------------------------------------------ #

class TestSamplePathLengths:
    """Integration tests for sample_path_lengths."""

    def test_returns_correct_count(self, linear_graph) -> None:
        random.seed(42)
        lengths = sample_path_lengths(linear_graph, 10)
        assert len(lengths) == 10

    def test_all_lengths_positive(self, linear_graph) -> None:
        random.seed(42)
        lengths = sample_path_lengths(
            linear_graph, 20, "unweighted",
        )
        assert all(bp > 0 for bp in lengths)

    def test_cycle_uses_all_nodes(self, cycle_gfa) -> None:
        random.seed(42)
        g = parse_gfa(cycle_gfa)
        lengths = sample_path_lengths(g, 10, "unweighted")
        assert len(lengths) == 10

    def test_weight_modes(self, linear_graph) -> None:
        random.seed(42)
        for mode in ("kmer", "overlap", "unweighted"):
            lengths = sample_path_lengths(
                linear_graph, 5, mode,
            )
            assert len(lengths) == 5


# ------------------------------------------------------------------ #
# Integration tests: sample_pair_distances
# ------------------------------------------------------------------ #

class TestSamplePairDistances:
    """Integration tests for sample_pair_distances."""

    def test_finds_pairs_in_linear(self, linear_graph) -> None:
        random.seed(42)
        idx = _seg_name_index(linear_graph)
        pairs = [(idx["s1"], idx["s2"])]
        dists = sample_pair_distances(
            linear_graph, pairs, 50, "unweighted",
        )
        assert len(dists) > 0
        assert all(d > 0 for d in dists)

    def test_no_pairs_returns_empty(self, linear_graph) -> None:
        random.seed(42)
        dists = sample_pair_distances(
            linear_graph, [], 10, "unweighted",
        )
        assert dists == []


# ------------------------------------------------------------------ #
# Integration tests: print_summary
# ------------------------------------------------------------------ #

class TestPrintSummary:
    """Tests for print_summary output."""

    def test_basic_output(self, linear_graph, capsys) -> None:
        print_summary(linear_graph, None)
        out = capsys.readouterr().out
        assert "Nodes (segments): 3" in out
        assert "Edges (links):    2" in out
        assert "Connected components:" in out

    def test_with_path_lengths(self, linear_graph, capsys) -> None:
        print_summary(
            linear_graph, [100, 200, 300], "kmer",
        )
        out = capsys.readouterr().out
        assert "Sampled path lengths" in out
        assert "kmer" in out

    def test_with_pair_distances(
        self, linear_graph, capsys,
    ) -> None:
        print_summary(
            linear_graph, None,
            pair_distances=[50, 100, 150],
        )
        out = capsys.readouterr().out
        assert "Estimated insert sizes" in out

    def test_empty_pair_distances(
        self, linear_graph, capsys,
    ) -> None:
        print_summary(
            linear_graph, None, pair_distances=[],
        )
        out = capsys.readouterr().out
        assert "no pairs found" in out

    def test_empty_graph(self, capsys) -> None:
        g: rx.PyGraph = rx.PyGraph()
        print_summary(g, None)
        out = capsys.readouterr().out
        assert "Nodes (segments): 0" in out


# ------------------------------------------------------------------ #
# E2E tests: CLI
# ------------------------------------------------------------------ #

class TestCli:
    """End-to-end CLI tests via CliRunner."""

    def test_basic_run(self, linear_gfa) -> None:
        result = runner.invoke(app, [str(linear_gfa)])
        assert result.exit_code == 0, result.output
        assert "Nodes" in result.output

    def test_no_sample_flag(self, linear_gfa) -> None:
        result = runner.invoke(
            app, [str(linear_gfa), "--no-sample"],
        )
        assert result.exit_code == 0, result.output
        assert "Nodes" in result.output

    def test_custom_samples(self, linear_gfa) -> None:
        result = runner.invoke(
            app, [str(linear_gfa), "-n", "5"],
        )
        assert result.exit_code == 0, result.output

    def test_weight_modes(self, linear_gfa) -> None:
        for mode in ("kmer", "overlap", "unweighted"):
            result = runner.invoke(
                app,
                [str(linear_gfa), "--weight", mode, "-n", "3"],
            )
            assert result.exit_code == 0, result.output

    def test_verbose_flag(self, linear_gfa) -> None:
        result = runner.invoke(
            app, [str(linear_gfa), "--verbose", "-n", "3"],
        )
        assert result.exit_code == 0, result.output

    def test_empty_gfa(self, empty_gfa) -> None:
        result = runner.invoke(
            app, [str(empty_gfa), "--no-sample"],
        )
        assert result.exit_code == 0, result.output
        assert "Nodes (segments): 0" in result.output

    def test_paired_end_without_bam_fails(
        self, linear_gfa,
    ) -> None:
        result = runner.invoke(
            app, [str(linear_gfa), "--paired-end"],
        )
        assert result.exit_code != 0

    def test_nonexistent_file(self) -> None:
        result = runner.invoke(app, ["/nonexistent/file.gfa"])
        assert result.exit_code != 0
