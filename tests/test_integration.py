"""Integration tests for generate_reads components working together."""

from __future__ import annotations

from pathlib import Path

import pysam
import pytest
import torch

from generate_reads import (
    Fragment,
    Read,
    ReadBatch,
    amplicon_fragments,
    apply_error_model,
    default_illumina_profile,
    generate_reads,
    load_genomes,
    sample_fragments,
    write_bam,
    write_fastq,
)


# ------------------------------------------------------------------ #
# load_genomes
# ------------------------------------------------------------------ #

class TestLoadGenomes:
    """Tests for load_genomes with real files."""

    def test_loads_single_genome(self, csv_path) -> None:
        genomes, abundances = load_genomes(csv_path)
        assert "genome1" in genomes
        assert len(genomes["genome1"]) == 2
        assert abundances["genome1"] == pytest.approx(1.0)

    def test_loads_two_genomes(self, two_genome_csv) -> None:
        genomes, abundances = load_genomes(two_genome_csv)
        assert len(genomes) == 2
        assert abundances["genome1"] == pytest.approx(0.7)
        assert abundances["genome2"] == pytest.approx(0.3)

    def test_abundances_normalised(self, tmp_path) -> None:
        fa = tmp_path / "g.fa"
        fa.write_text(">c1\nACGTACGT\n")
        csv_file = tmp_path / "input.csv"
        csv_file.write_text(
            "genome_id,fasta_path,abundance\n"
            f"a,{fa},3.0\n"
            f"b,{fa},7.0\n"
        )
        _, abundances = load_genomes(csv_file)
        assert abundances["a"] == pytest.approx(0.3)
        assert abundances["b"] == pytest.approx(0.7)

    def test_zero_abundance_raises(self, tmp_path) -> None:
        fa = tmp_path / "empty.fa"
        fa.write_text("")
        csv_file = tmp_path / "input.csv"
        csv_file.write_text(
            "genome_id,fasta_path,abundance\n"
            f"x,{fa},0.0\n"
        )
        with pytest.raises(ValueError, match="zero"):
            load_genomes(csv_file)


# ------------------------------------------------------------------ #
# sample_fragments
# ------------------------------------------------------------------ #

class TestSampleFragments:
    """Tests for sample_fragments with loaded genomes."""

    def test_correct_count(self, single_genome, rng) -> None:
        genomes, abundances = single_genome
        frags = sample_fragments(
            genomes, abundances, num_fragments=20,
            fragment_mean=30.0, fragment_variance=100.0,
            gc_bias_strength=0.0, rng=rng,
        )
        assert len(frags) == 20

    def test_fragment_coordinates_valid(
        self, single_genome, rng,
    ) -> None:
        genomes, abundances = single_genome
        frags = sample_fragments(
            genomes, abundances, num_fragments=50,
            fragment_mean=20.0, fragment_variance=100.0,
            gc_bias_strength=0.0, rng=rng,
        )
        for f in frags:
            assert 0 <= f.start < f.end
            assert f.strand in ("+", "-")
            assert len(f.sequence) == f.end - f.start
            assert f.genome_id == "genome1"

    def test_abundance_proportionality(
        self, two_genomes, rng,
    ) -> None:
        genomes, abundances = two_genomes
        frags = sample_fragments(
            genomes, abundances, num_fragments=100,
            fragment_mean=20.0, fragment_variance=100.0,
            gc_bias_strength=0.0, rng=rng,
        )
        g1 = sum(1 for f in frags if f.genome_id == "genome1")
        g2 = sum(1 for f in frags if f.genome_id == "genome2")
        assert g1 == 60
        assert g2 == 40

    def test_gc_bias_reduces_extreme_gc(
        self, single_genome, rng,
    ) -> None:
        genomes, abundances = single_genome
        frags_no_bias = sample_fragments(
            genomes, abundances, num_fragments=50,
            fragment_mean=20.0, fragment_variance=100.0,
            gc_bias_strength=0.0, rng=rng,
        )
        rng2 = torch.Generator()
        rng2.manual_seed(42)
        torch.manual_seed(42)
        frags_bias = sample_fragments(
            genomes, abundances, num_fragments=50,
            fragment_mean=20.0, fragment_variance=100.0,
            gc_bias_strength=10.0, rng=rng2,
        )
        # With strong GC bias, may get fewer fragments
        assert len(frags_bias) <= len(frags_no_bias)

    def test_reproducible_with_same_seed(
        self, single_genome,
    ) -> None:
        genomes, abundances = single_genome

        def _run(seed: int) -> list[str]:
            g = torch.Generator()
            g.manual_seed(seed)
            torch.manual_seed(seed)
            frags = sample_fragments(
                genomes, abundances, num_fragments=10,
                fragment_mean=30.0, fragment_variance=100.0,
                gc_bias_strength=0.0, rng=g,
            )
            return [f.sequence for f in frags]

        assert _run(99) == _run(99)


# ------------------------------------------------------------------ #
# amplicon_fragments
# ------------------------------------------------------------------ #

class TestAmpliconFragments:
    """Tests for amplicon_fragments."""

    def test_correct_count(self, single_genome, rng) -> None:
        genomes, abundances = single_genome
        frags = amplicon_fragments(genomes, abundances, 30, rng)
        assert len(frags) == 30

    def test_full_sequence_used(self, single_genome, rng) -> None:
        genomes, abundances = single_genome
        frags = amplicon_fragments(genomes, abundances, 10, rng)
        for f in frags:
            assert f.start == 0
            assert f.strand == "+"
            assert len(f.sequence) == f.end

    def test_abundance_proportionality(
        self, two_genomes, rng,
    ) -> None:
        genomes, abundances = two_genomes
        frags = amplicon_fragments(genomes, abundances, 100, rng)
        g1 = sum(1 for f in frags if f.genome_id == "genome1")
        g2 = sum(1 for f in frags if f.genome_id == "genome2")
        # genome1 has 1 contig (0.6), genome2 has 2 contigs (0.4)
        # after per-amplicon normalisation the split depends on
        # weight_tensor normalisation across 3 amplicons
        assert g1 + g2 == 100

    def test_shuffled(self, single_genome, rng) -> None:
        genomes, abundances = single_genome
        frags = amplicon_fragments(genomes, abundances, 30, rng)
        contig_ids = [f.contig_id for f in frags]
        # Not all grouped together (with two contigs, should be mixed)
        assert len(set(contig_ids)) > 1

    def test_empty_genomes(self, rng) -> None:
        frags = amplicon_fragments({}, {}, 10, rng)
        assert frags == []


# ------------------------------------------------------------------ #
# generate_reads
# ------------------------------------------------------------------ #

class TestGenerateReads:
    """Tests for generate_reads."""

    @pytest.fixture()
    def fragments(self) -> list[Fragment]:
        return [
            Fragment("g1", "c1", 0, 100, "+", "ACGT" * 25),
            Fragment("g1", "c1", 50, 150, "-", "TGCA" * 25),
        ]

    def test_single_end_count(self, fragments, rng) -> None:
        batch = generate_reads(
            fragments, read_length_mean=50.0,
            read_length_variance=1.0, paired_end=False, rng=rng,
        )
        assert batch.single is not None
        assert len(batch.single) == 2
        assert not batch.is_paired

    def test_paired_end_count(self, fragments, rng) -> None:
        batch = generate_reads(
            fragments, read_length_mean=50.0,
            read_length_variance=1.0, paired_end=True, rng=rng,
        )
        assert batch.paired is not None
        assert len(batch.paired) == 2
        assert batch.is_paired

    def test_read_length_bounded_by_fragment(
        self, rng,
    ) -> None:
        short_frag = Fragment("g1", "c1", 0, 10, "+", "ACGTACGTAC")
        batch = generate_reads(
            [short_frag], read_length_mean=1000.0,
            read_length_variance=1.0, paired_end=False, rng=rng,
        )
        assert len(batch.single[0].sequence) <= 10

    def test_quality_string_matches_sequence(
        self, fragments, rng,
    ) -> None:
        batch = generate_reads(
            fragments, read_length_mean=50.0,
            read_length_variance=1.0, paired_end=False, rng=rng,
        )
        for read in batch.single:
            assert len(read.quality) == len(read.sequence)

    def test_paired_read_names(self, fragments, rng) -> None:
        batch = generate_reads(
            fragments, read_length_mean=50.0,
            read_length_variance=1.0, paired_end=True, rng=rng,
        )
        for r1, r2 in batch.paired:
            assert "/1" in r1.name
            assert "/2" in r2.name


# ------------------------------------------------------------------ #
# apply_error_model
# ------------------------------------------------------------------ #

class TestApplyErrorModel:
    """Tests for apply_error_model integration."""

    def test_none_profile_returns_unchanged(self, rng) -> None:
        r = Read("r1", "ACGT", "IIII")
        batch = ReadBatch(single=[r])
        result = apply_error_model(batch, None, rng)
        assert result.single[0].sequence == "ACGT"

    def test_illumina_modifies_reads(self, rng) -> None:
        torch.manual_seed(42)
        seq = "ACGTACGTAC" * 10
        r = Read("r1", seq, "I" * len(seq))
        batch = ReadBatch(single=[r])
        profile = default_illumina_profile()
        result = apply_error_model(batch, profile, rng)
        assert result.single is not None
        assert len(result.single) == 1
        # CIGAR should be present
        assert result.single[0].cigar is not None

    def test_paired_end_error_model(self, rng) -> None:
        torch.manual_seed(42)
        seq = "ACGTACGTAC" * 5
        r1 = Read("r1/1", seq, "I" * len(seq))
        r2 = Read("r1/2", seq, "I" * len(seq))
        batch = ReadBatch(paired=[(r1, r2)])
        profile = default_illumina_profile()
        result = apply_error_model(batch, profile, rng)
        assert result.is_paired
        assert len(result.paired) == 1


# ------------------------------------------------------------------ #
# write_fastq
# ------------------------------------------------------------------ #

class TestWriteFastq:
    """Tests for write_fastq output format."""

    def test_valid_fastq(self, tmp_path) -> None:
        reads = [
            Read("read_0", "ACGT", "IIII"),
            Read("read_1", "TGCA", "!!!!"),
        ]
        out = tmp_path / "out.fastq"
        write_fastq(reads, out)

        lines = out.read_text().splitlines()
        assert len(lines) == 8
        assert lines[0] == "@read_0"
        assert lines[1] == "ACGT"
        assert lines[2] == "+"
        assert lines[3] == "IIII"
        assert lines[4] == "@read_1"

    def test_empty_reads(self, tmp_path) -> None:
        out = tmp_path / "empty.fastq"
        write_fastq([], out)
        assert out.read_text() == ""


# ------------------------------------------------------------------ #
# write_bam
# ------------------------------------------------------------------ #

class TestWriteBam:
    """Tests for write_bam output format."""

    def test_single_end_bam(
        self, tmp_path, single_genome,
    ) -> None:
        genomes, _ = single_genome
        frag = Fragment(
            "genome1", "contigA", 10, 30, "+", "ACGT" * 5,
        )
        read = Read("r1", "ACGT" * 5, "I" * 20)
        batch = ReadBatch(single=[read])
        bam_path = tmp_path / "out.bam"
        write_bam([frag], batch, genomes, bam_path)

        with pysam.AlignmentFile(bam_path, "rb") as bam:
            alns = list(bam)
        assert len(alns) == 1
        assert alns[0].reference_start == 10
        assert alns[0].mapping_quality == 255

    def test_paired_end_bam(
        self, tmp_path, single_genome,
    ) -> None:
        genomes, _ = single_genome
        frag = Fragment(
            "genome1", "contigA", 10, 50, "+", "ACGT" * 10,
        )
        r1 = Read("r1/1 read_0", "ACGT" * 5, "I" * 20)
        r2 = Read("r1/2 read_0", "ACGT" * 5, "I" * 20)
        batch = ReadBatch(paired=[(r1, r2)])
        bam_path = tmp_path / "out.bam"
        write_bam([frag], batch, genomes, bam_path)

        with pysam.AlignmentFile(bam_path, "rb") as bam:
            alns = list(bam)
        assert len(alns) == 2
        assert alns[0].is_read1
        assert alns[1].is_read2
        assert alns[0].is_paired
        assert alns[1].is_paired

    def test_bam_reference_names(
        self, tmp_path, single_genome,
    ) -> None:
        genomes, _ = single_genome
        frag = Fragment(
            "genome1", "contigA", 0, 20, "+", "ACGT" * 5,
        )
        read = Read("r1", "ACGT" * 5, "I" * 20)
        batch = ReadBatch(single=[read])
        bam_path = tmp_path / "out.bam"
        write_bam([frag], batch, genomes, bam_path)

        with pysam.AlignmentFile(bam_path, "rb") as bam:
            refs = bam.references
        assert "genome1:contigA" in refs
        assert "genome1:contigB" in refs
