"""Tests for scripts/parse_gfa.py."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import rustworkx as rx
from typer.testing import CliRunner

# Import from the script as a module
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from parse_gfa import (
    AhoCorasick,
    Link,
    MatcherMode,
    PathMatch,
    ReadIndex,
    Segment,
    SeedExtender,
    WeightMode,
    _compute_eligible,
    _overlap_length,
    _parse_tags,
    _path_pair_distances,
    _path_pair_insert_sizes,
    _pick_by_kmer,
    _pick_by_overlap,
    _pick_uniformly,
    _random_simple_path,
    _seg_name_index,
    _template_name,
    app,
    build_aho_corasick,
    build_read_index,
    build_seed_extender,
    build_seg_min_index,
    leaf_nodes,
    longest_simple_path,
    match_reads_to_path,
    parse_gfa,
    path_minimizer_sequence,
    print_summary,
    sample_pair_distances,
    sample_path_lengths,
)


runner = CliRunner()


# ------------------------------------------------------------------ #
# Unit: _template_name
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name, expected", [
    # Slash convention: /1, /2
    ("read/1", "read"),
    ("read/2", "read"),
    # Slash with R: /R1, /R2
    ("read/R1", "read"),
    ("read/R2", "read"),
    # Underscore with R: _R1, _R2
    ("read_R1", "read"),
    ("read_R2", "read"),
    # Dot convention: .1, .2, .R1, .R2
    ("read.1", "read"),
    ("read.2", "read"),
    ("read.R1", "read"),
    ("read.R2", "read"),
    # Illumina CASAVA space suffix
    ("read 1:N:0:ATCG", "read"),
    ("read 2:N:0:ATCG", "read"),
    # The specific failing case from a genome-blender read name
    (
        "MGYG000290000_1:MGYG000290000_1:77418-85405:+/2",
        "MGYG000290000_1:MGYG000290000_1:77418-85405:+",
    ),
    (
        "MGYG000290000_1:MGYG000290000_1:77418-85405:+/1",
        "MGYG000290000_1:MGYG000290000_1:77418-85405:+",
    ),
    # Bare _1/_2 must NOT be stripped: indistinguishable from accession suffix
    ("MGYG000290000_1", "MGYG000290000_1"),
    ("MGYG000290000_2", "MGYG000290000_2"),
    # No recognised suffix — name unchanged
    ("read_name", "read_name"),
    ("SRR123456.1.1", "SRR123456.1"),  # .1 stripped, but inner .1 kept
    # Description after space is stripped before suffix matching
    ("read/2 some description here", "read"),
    ("read_R1 extra", "read"),
])
def test_template_name(name: str, expected: str) -> None:
    """_template_name strips recognised pair suffixes and descriptions."""
    assert _template_name(name) == expected


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
        path = [(0, True), (1, True), (2, True)]
        dists = _path_pair_distances(g, path, [(1, 1)])
        assert dists == [200]

    def test_adjacent_segments(self) -> None:
        g = self._make_chain([100, 200, 300])
        path = [(0, True), (1, True), (2, True)]
        dists = _path_pair_distances(g, path, [(0, 1)])
        assert dists == [300]  # 100 + 200

    def test_spanning_full_path(self) -> None:
        g = self._make_chain([100, 200, 300])
        path = [(0, True), (1, True), (2, True)]
        dists = _path_pair_distances(g, path, [(0, 2)])
        assert dists == [600]  # 100 + 200 + 300

    def test_pair_not_in_path(self) -> None:
        g = self._make_chain([100, 200, 300])
        path = [(0, True), (1, True)]
        dists = _path_pair_distances(g, path, [(0, 2)])
        assert dists == []

    def test_reversed_pair_order(self) -> None:
        g = self._make_chain([100, 200, 300])
        path = [(0, True), (1, True), (2, True)]
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

    def test_debug_flag(self, linear_gfa) -> None:
        result = runner.invoke(
            app, [str(linear_gfa), "--debug", "-n", "3"],
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

    def test_json_output(self, tmp_path, linear_gfa) -> None:
        json_path = tmp_path / "summary.json"
        result = runner.invoke(
            app,
            [str(linear_gfa), "--no-sample", "--json", str(json_path)],
        )
        assert result.exit_code == 0, result.output
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["nodes"] == 3
        assert data["edges"] == 2
        assert "total_assembly_bp" in data
        assert data["gfa"] == str(linear_gfa)

    def test_json_with_path_lengths(
        self, tmp_path, linear_gfa,
    ) -> None:
        json_path = tmp_path / "summary.json"
        runner.invoke(
            app,
            [str(linear_gfa), "-n", "5", "--json", str(json_path)],
        )
        data = json.loads(json_path.read_text())
        assert "path_lengths" in data
        assert data["path_lengths"]["samples"] == 5


# ------------------------------------------------------------------ #
# Unit tests: AhoCorasick search with numpy uint64 text
# ------------------------------------------------------------------ #

class TestAhoCorasickNumpySearch:
    """AC search must work when text is np.ndarray[uint64], including
    for minimizer IDs >= 2**63 where numpy and Python int hashing
    can diverge."""

    def _ac_with(self, *patterns: tuple[int, ...]) -> AhoCorasick:
        ac = AhoCorasick()
        for pid, pat in enumerate(patterns):
            ac.add_pattern(pat, pid)
        ac.build()
        return ac

    def test_small_values_tuple(self) -> None:
        """Baseline: small IDs, tuple text."""
        ac = self._ac_with((10, 20, 30))
        matches = ac.search((5, 10, 20, 30, 40))
        assert any(pat_id == 0 for _, _, pat_id in matches)

    def test_small_values_numpy(self) -> None:
        """Small IDs as np.uint64 array must find the same matches."""
        ac = self._ac_with((10, 20, 30))
        text = np.array([5, 10, 20, 30, 40], dtype=np.uint64)
        matches = ac.search(text)
        assert any(pat_id == 0 for _, _, pat_id in matches)

    def test_large_values_above_int64_max(self) -> None:
        """Minimizer IDs >= 2**63 (common for NT-hash) must be found."""
        large = 2**63 + 99  # exceeds signed int64 range
        ac = self._ac_with((large, large + 1, large + 2))
        text = np.array(
            [large - 1, large, large + 1, large + 2, large + 3],
            dtype=np.uint64,
        )
        matches = ac.search(text)
        assert any(pat_id == 0 for _, _, pat_id in matches), (
            "AC search failed to find pattern with IDs >= 2**63 when "
            "text is np.ndarray[uint64]; likely a hash mismatch between "
            "np.uint64 and Python int for large values"
        )

    def test_reversed_pattern_large_values(self) -> None:
        """Reversed read pattern (revcomp) with large IDs must also match."""
        large = 2**63 + 99
        fwd = (large, large + 1, large + 2)
        rev = fwd[::-1]
        ac = self._ac_with(rev)
        text = np.array(
            [large - 1, large + 2, large + 1, large, large + 3],
            dtype=np.uint64,
        )
        matches = ac.search(text)
        assert any(pat_id == 0 for _, _, pat_id in matches), (
            "Reversed pattern with IDs >= 2**63 not found in numpy text"
        )


# ------------------------------------------------------------------ #
# Unit tests: path_minimizer_sequence + read matching pipeline
# ------------------------------------------------------------------ #

class TestRevcompReadMatching:
    """End-to-end tests for the path sequence → AC search pipeline,
    specifically verifying that reads from reverse-oriented nodes
    are found when their reversed minimizer pattern is indexed."""

    # k=3 so overlap=2.  All IDs are large (>= 2**63) to exercise the
    # hash-safety requirement identified by TestAhoCorasickNumpySearch.
    _BASE = 2**63 + 1000

    def _make_graph_and_index(
        self,
        node_minimizers: list[list[int]],
    ) -> tuple[rx.PyGraph, list[tuple[np.ndarray, np.ndarray] | None]]:
        """Build a linear chain graph with the given per-node minimizers."""
        g: rx.PyGraph = rx.PyGraph()
        seg_min: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        prev = None
        for i, mids in enumerate(node_minimizers):
            name = f"s{i}"
            idx = g.add_node(Segment(name, "A" * 10, 10))
            if prev is not None:
                g.add_edge(prev, idx, Link("+", "+", "*"))
            prev = idx
            fwd = np.array(mids, dtype=np.uint64)
            seg_min[name] = (fwd, fwd[::-1].copy())
        index = build_seg_min_index(g, seg_min)
        return g, index

    def test_forward_read_matches_forward_path(self) -> None:
        """A read whose minimizers appear in-order in the path is found."""
        B = self._BASE
        # Node 0: [B, B+1, B+2, B+3, B+4]  (5 minimizers)
        # Node 1: [B+3, B+4, B+5, B+6, B+7]  (overlap 2 with node 0)
        _, seg_idx = self._make_graph_and_index([
            [B, B+1, B+2, B+3, B+4],
            [B+3, B+4, B+5, B+6, B+7],
        ])
        path: list[tuple[int, bool]] = [(0, True), (1, True)]
        k = 3
        seq = path_minimizer_sequence(path, seg_idx, k)
        # Expected path: [B, B+1, B+2, B+3, B+4, B+5, B+6, B+7]
        assert list(map(int, seq)) == [B, B+1, B+2, B+3, B+4, B+5, B+6, B+7]

        read_mids = (B+1, B+2, B+3, B+4)
        ac = AhoCorasick()
        ac.add_pattern(read_mids, 0)
        ac.build()
        matches = match_reads_to_path(seq, ac)
        assert len(matches) == 1
        assert matches[0].read_id == 0

    def test_reversed_read_matches_reversed_node(self) -> None:
        """A read whose minimizers appear in the path only in reversed form
        is found via the reversed AC pattern."""
        B = self._BASE
        # Node 0 traversed in reverse: stored [B+4, B+3, B+2, B+1, B+0]
        # → path uses reversed = [B+0, B+1, B+2, B+3, B+4]
        _, seg_idx = self._make_graph_and_index([
            [B+4, B+3, B+2, B+1, B+0],  # stored; traversed reversed
        ])
        path: list[tuple[int, bool]] = [(0, False)]
        k = 3
        seq = path_minimizer_sequence(path, seg_idx, k)
        assert list(map(int, seq)) == [B+0, B+1, B+2, B+3, B+4]

        # Read in stored (forward) orientation: [B+4, B+3, B+2, B+1, B+0]
        # Its reversed form [B+0, B+1, B+2, B+3, B+4] IS in the path.
        read_fwd = (B+4, B+3, B+2, B+1, B+0)
        read_rev = read_fwd[::-1]
        ac = AhoCorasick()
        ac.add_pattern(read_fwd, 0)
        ac.add_pattern(read_rev, 0)
        ac.build()
        matches = match_reads_to_path(seq, ac)
        assert len(matches) >= 1
        assert matches[0].read_id == 0

    def test_paired_reads_yield_insert_size(self) -> None:
        """Two paired reads on the same path produce an insert size entry."""
        B = self._BASE
        # Linear path: [B, B+1, B+2, B+3, B+4, B+5, B+6, B+7]
        # R1 (forward) matches [B+0, B+1, B+2]
        # R2 (forward) matches [B+5, B+6, B+7]
        _, seg_idx = self._make_graph_and_index([
            [B, B+1, B+2, B+3, B+4],
            [B+3, B+4, B+5, B+6, B+7],
        ])
        path: list[tuple[int, bool]] = [(0, True), (1, True)]
        k = 3
        seq = path_minimizer_sequence(path, seg_idx, k)

        r1_mids = (B, B+1, B+2)
        r2_mids = (B+5, B+6, B+7)
        read_index = ReadIndex(
            name_to_id={"r/1": 0, "r/2": 1},
            names=["r/1", "r/2"],
            pairs={0: 1, 1: 0},
        )
        min_list = [("r/1", r1_mids), ("r/2", r2_mids)]
        path_minimizer_set = {int(m) for m in seq}
        individually_passing = {
            read_index.name_to_id[n] for n, m in min_list
            if m and all(v in path_minimizer_set for v in m)
        }
        eligible = _compute_eligible(read_index.pairs, individually_passing)
        ac = build_aho_corasick(iter(min_list), read_index, eligible)
        matches = match_reads_to_path(seq, ac)
        assert len(matches) == 2

        insert_sizes = _path_pair_insert_sizes(matches, read_index)
        assert len(insert_sizes) == 1
        _, _, span = insert_sizes[0]
        # Span covers positions 0..8 in minimizer space
        assert span == 8

    def test_union_set_filter_excludes_unrelated_reads(self) -> None:
        """Reads with minimizers absent from path sequences are excluded
        from AC and produce no matches."""
        B = self._BASE
        _, seg_idx = self._make_graph_and_index([[B, B+1, B+2, B+3]])
        path: list[tuple[int, bool]] = [(0, True)]
        k = 3
        seq = path_minimizer_sequence(path, seg_idx, k)
        path_minimizer_set = {int(m) for m in seq}

        # Read whose last minimizer (B+99) is NOT in the path.
        unrelated = ReadIndex(
            name_to_id={"r/1": 0},
            names=["r/1"],
            pairs={},
        )
        min_list = [("r/1", (B, B+1, B+99))]
        individually_passing = {
            unrelated.name_to_id[n] for n, m in min_list
            if m and all(v in path_minimizer_set for v in m)
        }
        eligible = _compute_eligible(unrelated.pairs, individually_passing)
        ac = build_aho_corasick(iter(min_list), unrelated, eligible)
        matches = match_reads_to_path(seq, ac)
        assert len(matches) == 0

    def test_union_set_filter_keeps_matching_reads(self) -> None:
        """Reads whose minimizers are all in the path pass the filter
        and are found by AC search."""
        B = self._BASE
        _, seg_idx = self._make_graph_and_index([[B, B+1, B+2, B+3]])
        path: list[tuple[int, bool]] = [(0, True)]
        k = 3
        seq = path_minimizer_sequence(path, seg_idx, k)
        path_minimizer_set = {int(m) for m in seq}

        matching = ReadIndex(
            name_to_id={"r/1": 0},
            names=["r/1"],
            pairs={},
        )
        min_list = [("r/1", (B+1, B+2, B+3))]
        individually_passing = {
            matching.name_to_id[n] for n, m in min_list
            if m and all(v in path_minimizer_set for v in m)
        }
        eligible = _compute_eligible(matching.pairs, individually_passing)
        ac = build_aho_corasick(iter(min_list), matching, eligible)
        matches = match_reads_to_path(seq, ac)
        assert len(matches) == 1

    def test_pair_filter_excludes_both_when_one_end_fails(self) -> None:
        """When one read of a pair fails the union set filter, both ends
        are excluded so a lone end cannot produce a spurious pair match."""
        B = self._BASE
        _, seg_idx = self._make_graph_and_index([[B, B+1, B+2, B+3, B+4]])
        path: list[tuple[int, bool]] = [(0, True)]
        k = 3
        seq = path_minimizer_sequence(path, seg_idx, k)
        path_minimizer_set = {int(m) for m in seq}

        # r/1 passes the filter; r/2 has B+99 which is absent from the path.
        index = ReadIndex(
            name_to_id={"r/1": 0, "r/2": 1},
            names=["r/1", "r/2"],
            pairs={0: 1, 1: 0},
        )
        min_list = [("r/1", (B+1, B+2, B+3)), ("r/2", (B+2, B+3, B+99))]
        individually_passing = {
            index.name_to_id[n] for n, m in min_list
            if m and all(v in path_minimizer_set for v in m)
        }
        eligible = _compute_eligible(index.pairs, individually_passing)
        ac = build_aho_corasick(iter(min_list), index, eligible)
        matches = match_reads_to_path(seq, ac)
        # r/1 must also be excluded because its pair (r/2) failed.
        assert len(matches) == 0

    def test_pair_filter_keeps_both_when_both_ends_pass(self) -> None:
        """When both reads of a pair pass the union set filter, both ends
        are added and a paired match is produced."""
        B = self._BASE
        _, seg_idx = self._make_graph_and_index([[B, B+1, B+2, B+3, B+4]])
        path: list[tuple[int, bool]] = [(0, True)]
        k = 3
        seq = path_minimizer_sequence(path, seg_idx, k)
        path_minimizer_set = {int(m) for m in seq}

        index = ReadIndex(
            name_to_id={"r/1": 0, "r/2": 1},
            names=["r/1", "r/2"],
            pairs={0: 1, 1: 0},
        )
        min_list = [("r/1", (B, B+1, B+2)), ("r/2", (B+2, B+3, B+4))]
        individually_passing = {
            index.name_to_id[n] for n, m in min_list
            if m and all(v in path_minimizer_set for v in m)
        }
        eligible = _compute_eligible(index.pairs, individually_passing)
        ac = build_aho_corasick(iter(min_list), index, eligible)
        matches = match_reads_to_path(seq, ac)
        assert len(matches) == 2
        insert_sizes = _path_pair_insert_sizes(matches, index)
        assert len(insert_sizes) == 1

    def test_first_node_missing_data_no_overlap_drop(self) -> None:
        """When early path nodes have no segment data, the first contributing
        node must NOT have its overlap prefix trimmed."""
        B = self._BASE
        g: rx.PyGraph = rx.PyGraph()
        # Node 0 has no minimizer data; node 1 has data.
        g.add_node(Segment("s0", "A" * 10, 10))
        g.add_node(Segment("s1", "A" * 10, 10))
        g.add_edge(0, 1, Link("+", "+", "*"))
        seg_min_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {
            "s1": (
                np.array([B, B+1, B+2, B+3, B+4], dtype=np.uint64),
                np.array([B+4, B+3, B+2, B+1, B], dtype=np.uint64),
            ),
        }
        seg_idx = build_seg_min_index(g, seg_min_dict)
        path: list[tuple[int, bool]] = [(0, True), (1, True)]
        # k=3 → overlap=2; s1 is the FIRST contributor so no prefix should
        # be stripped — expect all 5 minimizers, not [B+2, B+3, B+4].
        seq = path_minimizer_sequence(path, seg_idx, k=3)
        assert list(map(int, seq)) == [B, B+1, B+2, B+3, B+4]


class TestSeedExtender:
    """Tests for the SeedExtender read matcher."""

    _BASE = 0x4000_0000_0000_0000

    def _make_extender(
        self,
        reads: dict[str, tuple[int, ...]],
        min_chain_score: float = 1.0,
        max_gap: int = 0,
    ) -> tuple[ReadIndex, SeedExtender]:
        """Build a SeedExtender directly from an in-memory read dict."""
        index = ReadIndex(
            name_to_id={n: i for i, n in enumerate(sorted(reads))},
            names=sorted(reads),
            pairs={},
        )
        se = build_seed_extender(
            iter(sorted(reads.items())),
            index,
            min_chain_score=min_chain_score,
            max_gap=max_gap,
        )
        return index, se

    def test_exact_substring_match(self) -> None:
        """A read whose minimizers appear verbatim in the path is found."""
        B = self._BASE
        path_seq = np.array([B, B+1, B+2, B+3, B+4], dtype=np.uint64)
        _, se = self._make_extender({"r1": (B+1, B+2, B+3)})
        matches = se.search_path(path_seq)
        assert len(matches) == 1
        assert matches[0].read_id == 0
        assert matches[0].path_start == 1
        assert matches[0].path_end == 4

    def test_no_match_when_minimizer_absent(self) -> None:
        """A read with minimizers not in the path produces no match."""
        B = self._BASE
        path_seq = np.array([B, B+1, B+2], dtype=np.uint64)
        _, se = self._make_extender({"r1": (B+99,)})
        matches = se.search_path(path_seq)
        assert len(matches) == 0

    def test_reversed_read_is_found(self) -> None:
        """A read stored reversed (REVCOMP_AWARE) is found via the rev index."""
        B = self._BASE
        # Path in forward direction: [B, B+1, B+2, B+3]
        path_seq = np.array([B, B+1, B+2, B+3], dtype=np.uint64)
        # Read stored reversed: [B+3, B+2, B+1] — reversed = [B+1, B+2, B+3]
        # which IS in the path.
        _, se = self._make_extender({"r1": (B+3, B+2, B+1)})
        matches = se.search_path(path_seq)
        assert len(matches) == 1
        assert matches[0].read_id == 0

    def test_approximate_match_with_gap(self) -> None:
        """With max_gap>0 a read with one skipped minimizer still matches."""
        B = self._BASE
        # Path: [B, B+1, B+2, B+3, B+4]
        # Read: [B, B+2, B+4] — every other minimizer; gap=1 between each
        path_seq = np.array([B, B+1, B+2, B+3, B+4], dtype=np.uint64)
        _, se = self._make_extender(
            {"r1": (B, B+2, B+4)}, min_chain_score=1.0, max_gap=1,
        )
        matches = se.search_path(path_seq)
        assert len(matches) == 1

    def test_below_threshold_no_match(self) -> None:
        """With min_chain_score=1.0 a partial chain is rejected."""
        B = self._BASE
        # Path: [B, B+1, B+2, B+3]
        # Read: [B, B+1, B+99] — last minimizer absent → only 2/3 covered
        path_seq = np.array([B, B+1, B+2, B+3], dtype=np.uint64)
        _, se = self._make_extender(
            {"r1": (B, B+1, B+99)}, min_chain_score=1.0, max_gap=5,
        )
        matches = se.search_path(path_seq)
        assert len(matches) == 0

    def test_partial_match_above_threshold(self) -> None:
        """With min_chain_score<1.0 a partial chain is accepted."""
        B = self._BASE
        path_seq = np.array([B, B+1, B+2, B+3], dtype=np.uint64)
        # Read has 3 minimizers; only 2 appear in path → 2/3 ≈ 0.67
        _, se = self._make_extender(
            {"r1": (B, B+1, B+99)}, min_chain_score=0.6, max_gap=5,
        )
        matches = se.search_path(path_seq)
        assert len(matches) == 1

    def test_paired_reads_yield_insert_size(self) -> None:
        """Two paired reads chained to the same path produce an insert size."""
        B = self._BASE
        path_seq = np.array(
            [B, B+1, B+2, B+3, B+4, B+5, B+6, B+7], dtype=np.uint64
        )
        index = ReadIndex(
            name_to_id={"r/1": 0, "r/2": 1},
            names=["r/1", "r/2"],
            pairs={0: 1, 1: 0},
        )
        se = build_seed_extender(
            iter([("r/1", (B, B+1, B+2)), ("r/2", (B+5, B+6, B+7))]),
            index,
            min_chain_score=1.0,
            max_gap=0,
        )
        matches = se.search_path(path_seq)
        assert len(matches) == 2
        insert_sizes = _path_pair_insert_sizes(matches, index)
        assert len(insert_sizes) == 1
        _, _, span = insert_sizes[0]
        assert span == 8  # positions 0..8 inclusive

    def test_ac_and_seed_extend_agree_on_exact_match(self) -> None:
        """For exact matches both matchers report the same path positions."""
        B = self._BASE
        # Build a single-node graph with 5 minimizers.
        g: rx.PyGraph = rx.PyGraph()
        g.add_node(Segment("s0", "*", 50))
        fwd = np.array([B, B+1, B+2, B+3, B+4], dtype=np.uint64)
        seg_idx = build_seg_min_index(g, {"s0": (fwd, fwd[::-1].copy())})
        path: list[tuple[int, bool]] = [(0, True)]
        path_seq = path_minimizer_sequence(path, seg_idx, k=3)

        read_mids = (B+1, B+2, B+3)
        index = ReadIndex(
            name_to_id={"r1": 0}, names=["r1"], pairs={},
        )
        ac = build_aho_corasick(iter([("r1", read_mids)]), index)
        se = build_seed_extender(
            iter([("r1", read_mids)]), index,
            min_chain_score=1.0, max_gap=0,
        )
        ac_matches = ac.search_path(path_seq)
        se_matches = se.search_path(path_seq)
        assert len(ac_matches) == len(se_matches) == 1
        assert ac_matches[0].path_start == se_matches[0].path_start
        assert ac_matches[0].path_end == se_matches[0].path_end
