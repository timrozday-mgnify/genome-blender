"""End-to-end tests for the generate_reads CLI pipeline."""

from __future__ import annotations

from pathlib import Path

import pysam
from typer.testing import CliRunner

from generate_reads import app

runner = CliRunner()

# Fragment params suitable for small test sequences (80-100 bp)
_FRAG_ARGS = ["--fragment-mean", "30", "--fragment-variance", "100"]


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _write_fasta(path: Path, records: list[tuple[str, str]]) -> Path:
    """Write records as a FASTA file."""
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{seq}\n")
    return path


def _make_input(tmp_path: Path) -> tuple[Path, Path]:
    """Create a small FASTA and CSV for CLI testing.

    Returns:
        Tuple of (csv_path, fasta_path).
    """
    seq_a = "ACGTACGTAC" * 10  # 100 bp
    seq_b = "GCTAGCTAGC" * 8   # 80 bp
    fa = _write_fasta(
        tmp_path / "test.fa",
        [("contigA", seq_a), ("contigB", seq_b)],
    )
    csv_file = tmp_path / "input.csv"
    csv_file.write_text(
        "genome_id,fasta_path,abundance\n"
        f"sample1,{fa},1.0\n"
    )
    return csv_file, fa


def _count_fastq_reads(path: Path) -> int:
    """Count reads in a FASTQ file (4 lines per record)."""
    lines = path.read_text().splitlines()
    return len(lines) // 4


# ------------------------------------------------------------------ #
# Single-end, no error model
# ------------------------------------------------------------------ #

class TestSingleEndNoError:
    """SE reads without error model."""

    def test_produces_fastq(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "--input-csv", str(csv_path),
            "--num-reads", "10",
            "--output-prefix", str(out),
            "--single-end",
            "--seed", "42",
            *_FRAG_ARGS,
        ])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "out.fastq").exists()
        assert _count_fastq_reads(tmp_path / "out.fastq") == 10

    def test_produces_bam(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "--input-csv", str(csv_path),
            "--num-reads", "5",
            "--output-prefix", str(out),
            "--single-end",
            "--seed", "42",
            *_FRAG_ARGS,
        ])
        assert result.exit_code == 0, result.output
        bam_path = tmp_path / "out.bam"
        assert bam_path.exists()
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            alns = list(bam)
        assert len(alns) == 5


# ------------------------------------------------------------------ #
# Paired-end, no error model
# ------------------------------------------------------------------ #

class TestPairedEndNoError:
    """PE reads without error model."""

    def test_produces_r1_r2(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "--input-csv", str(csv_path),
            "--num-reads", "20",
            "--output-prefix", str(out),
            "--paired-end",
            "--seed", "42",
            *_FRAG_ARGS,
        ])
        assert result.exit_code == 0, result.output
        r1 = tmp_path / "out_R1.fastq"
        r2 = tmp_path / "out_R2.fastq"
        assert r1.exists()
        assert r2.exists()
        # 20 reads = 10 pairs
        assert _count_fastq_reads(r1) == 10
        assert _count_fastq_reads(r2) == 10

    def test_bam_has_paired_flags(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "--input-csv", str(csv_path),
            "--num-reads", "10",
            "--output-prefix", str(out),
            "--paired-end",
            "--seed", "42",
            *_FRAG_ARGS,
        ])
        assert result.exit_code == 0, result.output
        bam_path = tmp_path / "out.bam"
        with pysam.AlignmentFile(bam_path, "rb") as bam:
            alns = list(bam)
        # 10 reads = 5 pairs = 10 alignments
        assert len(alns) == 10
        for aln in alns:
            assert aln.is_paired


# ------------------------------------------------------------------ #
# Error model
# ------------------------------------------------------------------ #

class TestWithErrorModel:
    """Runs with an error model applied."""

    def test_illumina_se(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "--input-csv", str(csv_path),
            "--num-reads", "10",
            "--output-prefix", str(out),
            "--single-end",
            "--error-model", "illumina",
            "--seed", "42",
            *_FRAG_ARGS,
        ])
        assert result.exit_code == 0, result.output
        assert _count_fastq_reads(tmp_path / "out.fastq") == 10

    def test_illumina_pe(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "--input-csv", str(csv_path),
            "--num-reads", "10",
            "--output-prefix", str(out),
            "--paired-end",
            "--error-model", "illumina",
            "--seed", "42",
            *_FRAG_ARGS,
        ])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "out_R1.fastq").exists()
        assert (tmp_path / "out_R2.fastq").exists()


# ------------------------------------------------------------------ #
# Amplicon mode
# ------------------------------------------------------------------ #

class TestAmpliconMode:
    """CLI with --amplicon flag."""

    def test_amplicon_se(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "--input-csv", str(csv_path),
            "--num-reads", "10",
            "--output-prefix", str(out),
            "--single-end",
            "--amplicon",
            "--seed", "42",
        ])
        assert result.exit_code == 0, result.output
        assert _count_fastq_reads(tmp_path / "out.fastq") == 10

    def test_amplicon_pe(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "--input-csv", str(csv_path),
            "--num-reads", "20",
            "--output-prefix", str(out),
            "--paired-end",
            "--amplicon",
            "--seed", "42",
        ])
        assert result.exit_code == 0, result.output
        r1 = tmp_path / "out_R1.fastq"
        r2 = tmp_path / "out_R2.fastq"
        assert r1.exists()
        assert r2.exists()
        assert _count_fastq_reads(r1) == 10
        assert _count_fastq_reads(r2) == 10

    def test_amplicon_with_error_model(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "--input-csv", str(csv_path),
            "--num-reads", "10",
            "--output-prefix", str(out),
            "--single-end",
            "--amplicon",
            "--error-model", "illumina",
            "--seed", "42",
        ])
        assert result.exit_code == 0, result.output
        assert _count_fastq_reads(tmp_path / "out.fastq") == 10


# ------------------------------------------------------------------ #
# YAML config
# ------------------------------------------------------------------ #

class TestYamlConfig:
    """CLI with --config YAML file."""

    def test_config_file(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        out = tmp_path / "out"
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"input-csv: {csv_path}\n"
            f"num-reads: 10\n"
            f"output-prefix: {out}\n"
            f"fragment-mean: 30\n"
            f"fragment-variance: 100\n"
            f"seed: 42\n"
        )
        result = runner.invoke(app, ["--config", str(cfg)])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "out.fastq").exists()
        assert _count_fastq_reads(tmp_path / "out.fastq") == 10

    def test_cli_overrides_config(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        out = tmp_path / "out"
        cfg = tmp_path / "config.yaml"
        cfg.write_text(
            f"input-csv: {csv_path}\n"
            f"num-reads: 100\n"
            f"output-prefix: {out}\n"
            f"fragment-mean: 30\n"
            f"fragment-variance: 100\n"
            f"seed: 42\n"
        )
        # CLI overrides num-reads from 100 to 5
        result = runner.invoke(app, [
            "--config", str(cfg),
            "--num-reads", "5",
        ])
        assert result.exit_code == 0, result.output
        assert _count_fastq_reads(tmp_path / "out.fastq") == 5


# ------------------------------------------------------------------ #
# Reproducibility
# ------------------------------------------------------------------ #

class TestReproducibility:
    """Same seed produces identical output."""

    def test_same_seed_same_fastq(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)

        def _run(run_dir: Path) -> str:
            run_dir.mkdir(exist_ok=True)
            out = run_dir / "out"
            runner.invoke(app, [
                "--input-csv", str(csv_path),
                "--num-reads", "10",
                "--output-prefix", str(out),
                "--single-end",
                "--seed", "99",
                *_FRAG_ARGS,
            ])
            return (run_dir / "out.fastq").read_text()

        run1 = _run(tmp_path / "run1")
        run2 = _run(tmp_path / "run2")
        assert run1 == run2

    def test_different_seed_different_fastq(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)

        def _run(run_dir: Path, seed: int) -> str:
            run_dir.mkdir(exist_ok=True)
            out = run_dir / "out"
            runner.invoke(app, [
                "--input-csv", str(csv_path),
                "--num-reads", "10",
                "--output-prefix", str(out),
                "--single-end",
                "--seed", str(seed),
                *_FRAG_ARGS,
            ])
            return (run_dir / "out.fastq").read_text()

        run1 = _run(tmp_path / "run1", 42)
        run2 = _run(tmp_path / "run2", 99)
        assert run1 != run2


# ------------------------------------------------------------------ #
# Error cases
# ------------------------------------------------------------------ #

class TestErrorCases:
    """CLI validation errors."""

    def test_missing_input_csv(self, tmp_path) -> None:
        result = runner.invoke(app, [
            "--num-reads", "10",
            "--output-prefix", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

    def test_missing_num_reads(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        result = runner.invoke(app, [
            "--input-csv", str(csv_path),
            "--output-prefix", str(tmp_path / "out"),
        ])
        assert result.exit_code != 0

    def test_missing_output_prefix(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        result = runner.invoke(app, [
            "--input-csv", str(csv_path),
            "--num-reads", "10",
        ])
        assert result.exit_code != 0


# ------------------------------------------------------------------ #
# Verbose flag
# ------------------------------------------------------------------ #

class TestVerboseFlag:
    """CLI --verbose flag."""

    def test_verbose_runs_ok(self, tmp_path) -> None:
        csv_path, _ = _make_input(tmp_path)
        out = tmp_path / "out"
        result = runner.invoke(app, [
            "--input-csv", str(csv_path),
            "--num-reads", "5",
            "--output-prefix", str(out),
            "--single-end",
            "--seed", "42",
            "--verbose",
            *_FRAG_ARGS,
        ])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "out.fastq").exists()
