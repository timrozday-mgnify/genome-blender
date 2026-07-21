"""Integration tests for generate_reads components working together."""

from __future__ import annotations

from pathlib import Path

import pysam
import pytest
import torch

from genome_blender import (
    Fragment,
    Read,
    ReadBatch,
    SkiverModelConfig,
    amplicon_fragments,
    apply_error_model,
    generate_reads,
    load_genomes,
    sample_fragments,
    write_bam,
    write_fastq,
)

import sys
import textwrap

# A stub `skiver-generate` that echoes each input read unchanged with a
# full-length M CIGAR and uniform quality.  Lets us test genome-blender's
# subprocess plumbing (FASTA out, FASTQ + cigar in, pairing, name preservation)
# without depending on a real Skiver install.
_STUB_SCRIPT = textwrap.dedent('''
    import sys
    args = sys.argv[1:]
    inp = args[args.index("--input") + 1]
    name, seq = None, []
    recs = []
    for line in open(inp):
        line = line.rstrip("\\n")
        if line.startswith(">"):
            if name is not None:
                recs.append((name, "".join(seq)))
            name, seq = line[1:].split()[0], []
        elif line:
            seq.append(line)
    if name is not None:
        recs.append((name, "".join(seq)))
    out = sys.stdout
    for nm, s in recs:
        cig = f"{len(s)}M" if s else ""
        out.write(f"@{nm} cigar:{cig}\\n{s}\\n+\\n{'I' * len(s)}\\n")
''')


def _stub_model_cfg(tmp_path) -> SkiverModelConfig:
    stub = tmp_path / "stub_skiver_generate.py"
    stub.write_text(_STUB_SCRIPT)
    return SkiverModelConfig(
        preset="hq-illumina",
        generate_cmd=[sys.executable, str(stub)],
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
        # SE headers end in /1 too, so AAP's fastq_suffix_header_check passes.
        for read in batch.single:
            assert read.name.endswith("/1")

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
            # Mirrors AAP's read headers: shared record.id (token before the first
            # space) so mates pair by id, and the line ends in /1,/2 so the suffix
            # check passes.
            assert r1.name.split()[0] == r2.name.split()[0]
            assert r1.name.endswith("/1")
            assert r2.name.endswith("/2")

    def test_names_unique_for_identical_coordinates(
        self, rng,
    ) -> None:
        # Two fragments at the same locus (as amplicon copies are) must
        # still get distinct read names, so merged chunk BAMs have no
        # colliding query_names.
        dup = [
            Fragment("g1", "c1", 0, 100, "+", "ACGT" * 25),
            Fragment("g1", "c1", 0, 100, "+", "ACGT" * 25),
        ]
        batch = generate_reads(
            dup, read_length_mean=50.0, read_length_variance=1.0,
            paired_end=False, rng=rng,
        )
        names = [r.name for r in batch.single]
        assert len(set(names)) == len(names)

    def test_name_prefix_applied(self, fragments, rng) -> None:
        batch = generate_reads(
            fragments, read_length_mean=50.0,
            read_length_variance=1.0, paired_end=False, rng=rng,
            name_prefix="chunk7:",
        )
        for r in batch.single:
            assert r.name.startswith("chunk7:")


# ------------------------------------------------------------------ #
# apply_error_model
# ------------------------------------------------------------------ #

class TestApplyErrorModel:
    """Tests for apply_error_model via the skiver-generate subprocess."""

    def test_none_config_returns_unchanged(self) -> None:
        r = Read("r1", "ACGT", "IIII")
        batch = ReadBatch(single=[r])
        result = apply_error_model(batch, None, seed=0)
        assert result.single is not None
        assert result.single[0].sequence == "ACGT"

    def test_single_end_subprocess(self, tmp_path) -> None:
        seq = "ACGTACGTAC" * 10
        r = Read("read_0", seq, "I" * len(seq))
        batch = ReadBatch(single=[r])
        result = apply_error_model(
            batch, _stub_model_cfg(tmp_path), seed=0,
        )
        assert result.single is not None
        assert len(result.single) == 1
        out = result.single[0]
        # Stub echoes the sequence; plumbing must preserve name and parse CIGAR.
        assert out.name == "read_0"
        assert out.sequence == seq
        assert out.cigar == [(0, len(seq))]

    def test_paired_end_subprocess(self, tmp_path) -> None:
        s1, s2 = "ACGTACGTAC" * 5, "TTTTGGGGCC" * 5
        r1 = Read("r0/1 read_0", s1, "I" * len(s1))
        r2 = Read("r0/2 read_0", s2, "I" * len(s2))
        batch = ReadBatch(paired=[(r1, r2)])
        result = apply_error_model(
            batch, _stub_model_cfg(tmp_path), seed=0,
        )
        assert result.is_paired
        assert result.paired is not None
        assert len(result.paired) == 1
        out1, out2 = result.paired[0]
        # Order-based round-trip keeps mates aligned to their source reads.
        assert out1.name == "r0/1 read_0" and out1.sequence == s1
        assert out2.name == "r0/2 read_0" and out2.sequence == s2
        assert out1.cigar == [(0, len(s1))]


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
        r1 = Read("r1 read_0/1", "ACGT" * 5, "I" * 20)
        r2 = Read("r1 read_0/2", "ACGT" * 5, "I" * 20)
        batch = ReadBatch(paired=[(r1, r2)])
        bam_path = tmp_path / "out.bam"
        write_bam([frag], batch, genomes, bam_path)

        with pysam.AlignmentFile(bam_path, "rb") as bam:
            alns = list(bam)
        assert len(alns) == 2
        assert alns[0].is_read1
        # Both mates carry the same bare query_name (mate flag stripped).
        assert alns[0].query_name == alns[1].query_name == "r1"
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

    def test_minimal_bam_omits_seq_qual(
        self, tmp_path, single_genome,
    ) -> None:
        # minimal=True drops SEQ/QUAL but keeps name, position, ref and CIGAR,
        # so the truth-table pipeline (name + reference only) is unaffected.
        genomes, _ = single_genome
        frag = Fragment(
            "genome1", "contigA", 10, 30, "+", "ACGT" * 5,
        )
        read = Read("r1", "ACGT" * 5, "I" * 20)
        batch = ReadBatch(single=[read])
        bam_path = tmp_path / "minimal.bam"
        write_bam([frag], batch, genomes, bam_path, minimal=True)

        with pysam.AlignmentFile(bam_path, "rb") as bam:
            alns = list(bam)
        assert len(alns) == 1
        assert alns[0].query_sequence is None
        assert alns[0].query_qualities is None
        assert alns[0].query_name == "r1"
        assert alns[0].reference_start == 10
        assert alns[0].reference_name == "genome1:contigA"
        assert alns[0].cigarstring is not None
