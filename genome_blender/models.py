"""Data models for the read-generation pipeline.

Dataclasses for fragments, reads, and read batches, plus the error-model
selection enum.  Error application itself is delegated to Skiver (see
:mod:`genome_blender.error_model`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ErrorModel(str, Enum):
    """Sequencing error model selection.

    Each non-``none`` value maps to a Skiver platform preset in
    :mod:`genome_blender.error_model`.
    """

    none = "none"
    illumina = "illumina"
    pacbio = "pacbio"
    nanopore = "nanopore"


@dataclass
class Fragment:
    """A DNA fragment excised from a genome."""

    genome_id: str
    contig_id: str
    start: int  # 0-based
    end: int  # half-open
    strand: str  # '+' or '-'
    sequence: str


@dataclass
class Read:
    """A sequencing read with name, sequence, and quality string."""

    name: str
    sequence: str
    quality: str
    cigar: list[tuple[int, int]] | None = None


@dataclass
class ReadBatch:
    """Container for generated reads (single-end or paired-end).

    Exactly one of ``single`` or ``paired`` is set; the other
    is ``None``.
    """

    single: list[Read] | None = None
    paired: list[tuple[Read, Read]] | None = None

    @property
    def is_paired(self) -> bool:
        """Return True if this batch contains paired-end reads."""
        return self.paired is not None
