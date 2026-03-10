"""Data models for the read-generation pipeline.

Dataclasses for fragments, reads, quality calibration, and error
model profiles, plus CLI-facing enums.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch


class ErrorModel(str, Enum):
    """Sequencing error model selection."""

    none = "none"
    illumina = "illumina"
    pacbio = "pacbio"
    nanopore = "nanopore"


class QualityCalibrationModel(str, Enum):
    """Quality-score-to-error-rate calibration model."""

    phred = "phred"
    log_linear = "log-linear"
    sigmoid = "sigmoid"


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


@dataclass
class QualityCalibration:
    """Base for quality-score-to-error-rate calibration models."""

    name: str

    def __call__(self, q_scores: torch.Tensor) -> torch.Tensor:
        """Convert quality scores to error probabilities."""
        raise NotImplementedError


@dataclass
class PhredCalibration(QualityCalibration):
    """Theoretical Phred: P = 10^(-Q/10)."""

    name: str = "phred"

    def __call__(self, q_scores: torch.Tensor) -> torch.Tensor:
        return 10.0 ** (-q_scores.float() / 10.0)


@dataclass
class LogLinearCalibration(QualityCalibration):
    """DADA2-style: P = clamp(10^(intercept + slope*Q), floor, ceiling)."""

    name: str = "log-linear"
    intercept: float = -0.3
    slope: float = -0.08
    floor: float = 1e-7
    ceiling: float = 0.5

    def __call__(self, q_scores: torch.Tensor) -> torch.Tensor:
        log10_p = self.intercept + self.slope * q_scores.float()
        return (10.0 ** log10_p).clamp(
            min=self.floor, max=self.ceiling,
        )


@dataclass
class SigmoidCalibration(QualityCalibration):
    """Logistic sigmoid: P = floor + (ceiling-floor) * sigma(-k*(Q-mid))."""

    name: str = "sigmoid"
    steepness: float = 0.25
    midpoint: float = 15.0
    floor: float = 1e-6
    ceiling: float = 0.5

    def __call__(self, q_scores: torch.Tensor) -> torch.Tensor:
        x = -self.steepness * (q_scores.float() - self.midpoint)
        return self.floor + (
            (self.ceiling - self.floor) * torch.sigmoid(x)
        )


@dataclass
class ErrorModelProfile:
    """HMM-based sequencing error model parameters."""

    name: str
    num_states: int
    initial_logits: torch.Tensor
    transition_logits: torch.Tensor
    emission_logits: torch.Tensor
    substitution_ratio: float
    insertion_ratio: float
    deletion_ratio: float
