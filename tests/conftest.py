"""Shared fixtures for genome-blender tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


# ---------------------------------------------------------------------------
# Small synthetic sequences (deterministic, easy to reason about)
# ---------------------------------------------------------------------------

#: 100 bp, 50% GC
SEQ_A = "ACGTACGTAC" * 10

#: 80 bp, 60% GC
SEQ_B = "GCTAGCTAGC" * 8

#: 120 bp, ~42% GC
SEQ_C = "TTAGAATTAG" * 12


@pytest.fixture()
def seqrecord_a() -> SeqRecord:
    """Return a 100 bp SeqRecord."""
    return SeqRecord(Seq(SEQ_A), id="contigA", description="")


@pytest.fixture()
def seqrecord_b() -> SeqRecord:
    """Return an 80 bp SeqRecord."""
    return SeqRecord(Seq(SEQ_B), id="contigB", description="")


@pytest.fixture()
def seqrecord_c() -> SeqRecord:
    """Return a 120 bp SeqRecord."""
    return SeqRecord(Seq(SEQ_C), id="contigC", description="")


# ---------------------------------------------------------------------------
# Genome dicts (as returned by load_genomes)
# ---------------------------------------------------------------------------

@pytest.fixture()
def single_genome(seqrecord_a, seqrecord_b):
    """Return a single-genome dict with two contigs."""
    genomes = {"genome1": [seqrecord_a, seqrecord_b]}
    abundances = {"genome1": 1.0}
    return genomes, abundances


@pytest.fixture()
def two_genomes(seqrecord_a, seqrecord_b, seqrecord_c):
    """Return a two-genome dict."""
    genomes = {
        "genome1": [seqrecord_a],
        "genome2": [seqrecord_b, seqrecord_c],
    }
    abundances = {"genome1": 0.6, "genome2": 0.4}
    return genomes, abundances


# ---------------------------------------------------------------------------
# File fixtures (FASTA + CSV on disk)
# ---------------------------------------------------------------------------

def _write_fasta(path: Path, records: list[tuple[str, str]]) -> Path:
    """Write records as a FASTA file. Returns path."""
    with open(path, "w") as fh:
        for name, seq in records:
            fh.write(f">{name}\n{seq}\n")
    return path


@pytest.fixture()
def fasta_path(tmp_path) -> Path:
    """Write a two-contig FASTA and return its path."""
    return _write_fasta(
        tmp_path / "test.fa",
        [("contigA", SEQ_A), ("contigB", SEQ_B)],
    )


@pytest.fixture()
def csv_path(tmp_path, fasta_path) -> Path:
    """Write an input CSV pointing to fasta_path."""
    csv_file = tmp_path / "input.csv"
    csv_file.write_text(
        "genome_id,fasta_path,abundance\n"
        f"genome1,{fasta_path},1.0\n"
    )
    return csv_file


@pytest.fixture()
def two_genome_csv(tmp_path) -> Path:
    """Write a CSV with two genomes at different abundances."""
    fa1 = _write_fasta(
        tmp_path / "g1.fa", [("contigA", SEQ_A)],
    )
    fa2 = _write_fasta(
        tmp_path / "g2.fa",
        [("contigB", SEQ_B), ("contigC", SEQ_C)],
    )
    csv_file = tmp_path / "input.csv"
    csv_file.write_text(
        "genome_id,fasta_path,abundance\n"
        f"genome1,{fa1},0.7\n"
        f"genome2,{fa2},0.3\n"
    )
    return csv_file


# ---------------------------------------------------------------------------
# RNG fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def rng() -> torch.Generator:
    """Return a seeded torch Generator."""
    g = torch.Generator()
    g.manual_seed(42)
    torch.manual_seed(42)
    return g
