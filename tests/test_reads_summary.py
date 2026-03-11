"""Tests for scripts/reads_summary.py."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from reads_summary import (
    _detect_format,
    _read_lengths_fasta,
    _read_lengths_fastq,
    app,
    print_summary,
)


runner = CliRunner()


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture()
def fastq_path(tmp_path) -> Path:
    """Write a small FASTQ file (3 reads of lengths 10, 20, 30)."""
    lines = []
    for i, seq in enumerate(["A" * 10, "C" * 20, "G" * 30]):
        lines += [f"@read{i}", seq, "+", "I" * len(seq)]
    path = tmp_path / "reads.fastq"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture()
def fasta_path(tmp_path) -> Path:
    """Write a small FASTA file (3 records of lengths 15, 25, 35)."""
    text = ""
    for i, seq in enumerate(["A" * 15, "C" * 25, "G" * 35]):
        text += f">seq{i}\n{seq}\n"
    path = tmp_path / "seqs.fasta"
    path.write_text(text)
    return path


@pytest.fixture()
def multiline_fasta(tmp_path) -> Path:
    """Write a FASTA with sequence split across lines."""
    path = tmp_path / "multi.fasta"
    path.write_text(
        ">contig1\n"
        "ACGT\n"
        "ACGT\n"
        ">contig2\n"
        "CCCC\n"
    )
    return path


@pytest.fixture()
def gzipped_fastq(tmp_path, fastq_path) -> Path:
    """Write a gzipped copy of fastq_path."""
    gz_path = tmp_path / "reads.fastq.gz"
    with open(fastq_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        dst.write(src.read())
    return gz_path


@pytest.fixture()
def gzipped_fasta(tmp_path, fasta_path) -> Path:
    """Write a gzipped copy of fasta_path."""
    gz_path = tmp_path / "seqs.fasta.gz"
    with open(fasta_path, "rb") as src, gzip.open(gz_path, "wb") as dst:
        dst.write(src.read())
    return gz_path


# ------------------------------------------------------------------ #
# Unit tests: _detect_format
# ------------------------------------------------------------------ #

class TestDetectFormat:
    """Tests for _detect_format."""

    def test_detects_fastq(self, fastq_path) -> None:
        assert _detect_format(fastq_path) == "fastq"

    def test_detects_fasta(self, fasta_path) -> None:
        assert _detect_format(fasta_path) == "fasta"

    def test_detects_gzipped_fastq(self, gzipped_fastq) -> None:
        assert _detect_format(gzipped_fastq) == "fastq"

    def test_detects_gzipped_fasta(self, gzipped_fasta) -> None:
        assert _detect_format(gzipped_fasta) == "fasta"

    def test_unknown_format_raises(self, tmp_path) -> None:
        bad = tmp_path / "bad.txt"
        bad.write_text("not a valid format\n")
        with pytest.raises(Exception):
            _detect_format(bad)


# ------------------------------------------------------------------ #
# Unit tests: _read_lengths_fastq
# ------------------------------------------------------------------ #

class TestReadLengthsFastq:
    """Tests for _read_lengths_fastq."""

    def test_reads_all_lengths(self, fastq_path) -> None:
        lengths = _read_lengths_fastq(fastq_path, None)
        assert sorted(lengths) == [10, 20, 30]

    def test_max_reads_limits_count(self, fastq_path) -> None:
        lengths = _read_lengths_fastq(fastq_path, 2)
        assert len(lengths) == 2

    def test_gzipped_file(self, gzipped_fastq) -> None:
        lengths = _read_lengths_fastq(gzipped_fastq, None)
        assert sorted(lengths) == [10, 20, 30]

    def test_empty_file(self, tmp_path) -> None:
        empty = tmp_path / "empty.fastq"
        empty.write_text("")
        lengths = _read_lengths_fastq(empty, None)
        assert lengths == []


# ------------------------------------------------------------------ #
# Unit tests: _read_lengths_fasta
# ------------------------------------------------------------------ #

class TestReadLengthsFasta:
    """Tests for _read_lengths_fasta."""

    def test_reads_all_lengths(self, fasta_path) -> None:
        lengths = _read_lengths_fasta(fasta_path, None)
        assert sorted(lengths) == [15, 25, 35]

    def test_max_reads_limits_count(self, fasta_path) -> None:
        lengths = _read_lengths_fasta(fasta_path, 2)
        assert len(lengths) == 2

    def test_multiline_sequence(self, multiline_fasta) -> None:
        lengths = _read_lengths_fasta(multiline_fasta, None)
        assert sorted(lengths) == [4, 8]

    def test_gzipped_file(self, gzipped_fasta) -> None:
        lengths = _read_lengths_fasta(gzipped_fasta, None)
        assert sorted(lengths) == [15, 25, 35]

    def test_empty_file(self, tmp_path) -> None:
        empty = tmp_path / "empty.fasta"
        empty.write_text("")
        lengths = _read_lengths_fasta(empty, None)
        assert lengths == []


# ------------------------------------------------------------------ #
# Unit tests: print_summary
# ------------------------------------------------------------------ #

class TestPrintSummary:
    """Tests for print_summary."""

    def test_empty_lengths(self, capsys) -> None:
        print_summary([])
        out = capsys.readouterr().out
        assert "No reads" in out

    def test_single_read(self, capsys) -> None:
        print_summary([100])
        out = capsys.readouterr().out
        assert "Reads:      1" in out
        assert "Min:        100" in out
        assert "Max:        100" in out
        assert "N50:        100" in out

    def test_stats_present(self, capsys) -> None:
        print_summary([10, 20, 30, 40, 50])
        out = capsys.readouterr().out
        assert "Reads:" in out
        assert "Total bp:" in out
        assert "Mean:" in out
        assert "Median:" in out
        assert "Std dev:" in out
        assert "N50:" in out

    def test_n50_correct(self, capsys) -> None:
        # Lengths [1, 2, 3, 4, 5], total=15, half=7.5
        # Sorted descending: 5, 4, 3, 2, 1
        # cumulative: 5 (< 7.5), 9 (>= 7.5) at length 4 → N50 = 4
        print_summary([1, 2, 3, 4, 5])
        out = capsys.readouterr().out
        assert "N50:        4" in out

    def test_even_count_median(self, capsys) -> None:
        print_summary([10, 20, 30, 40])
        out = capsys.readouterr().out
        assert "Median:" in out


# ------------------------------------------------------------------ #
# E2E tests: CLI
# ------------------------------------------------------------------ #

class TestCli:
    """End-to-end CLI tests via CliRunner."""

    def test_fastq_default_n(self, fastq_path) -> None:
        result = runner.invoke(app, [str(fastq_path)])
        assert result.exit_code == 0, result.output
        assert "reads.fastq" in result.output
        assert "fastq" in result.output

    def test_fasta_default_n(self, fasta_path) -> None:
        result = runner.invoke(app, [str(fasta_path)])
        assert result.exit_code == 0, result.output
        assert "seqs.fasta" in result.output
        assert "fasta" in result.output

    def test_all_flag(self, fastq_path) -> None:
        result = runner.invoke(app, [str(fastq_path), "--all"])
        assert result.exit_code == 0, result.output
        assert "all reads" in result.output
        assert "Reads:      3" in result.output

    def test_n_limits_reads(self, fastq_path) -> None:
        result = runner.invoke(app, [str(fastq_path), "-n", "2"])
        assert result.exit_code == 0, result.output
        assert "first 2" in result.output
        assert "Reads:      2" in result.output

    def test_gzipped_fastq(self, gzipped_fastq) -> None:
        result = runner.invoke(app, [str(gzipped_fastq), "--all"])
        assert result.exit_code == 0, result.output
        assert "Reads:      3" in result.output

    def test_gzipped_fasta(self, gzipped_fasta) -> None:
        result = runner.invoke(app, [str(gzipped_fasta), "--all"])
        assert result.exit_code == 0, result.output
        assert "Reads:      3" in result.output

    def test_nonexistent_file(self) -> None:
        result = runner.invoke(app, ["/nonexistent/reads.fastq"])
        assert result.exit_code != 0

    def test_stats_in_output(self, fasta_path) -> None:
        result = runner.invoke(app, [str(fasta_path), "--all"])
        assert result.exit_code == 0, result.output
        assert "N50:" in result.output
        assert "Mean:" in result.output

    def test_json_output(self, tmp_path, fastq_path) -> None:
        json_path = tmp_path / "stats.json"
        result = runner.invoke(
            app,
            [str(fastq_path), "--all", "--json", str(json_path)],
        )
        assert result.exit_code == 0, result.output
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["reads"] == 3
        assert "mean" in data
        assert "n50" in data
        assert data["format"] == "fastq"

    def test_json_contains_mean(
        self, tmp_path, fastq_path,
    ) -> None:
        json_path = tmp_path / "stats.json"
        runner.invoke(
            app,
            [str(fastq_path), "--all", "--json", str(json_path)],
        )
        data = json.loads(json_path.read_text())
        # mean of [10, 20, 30] = 20.0
        assert data["mean"] == pytest.approx(20.0)
