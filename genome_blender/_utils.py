"""Shared low-level utilities for the read-generation pipeline."""

from __future__ import annotations

import math

from Bio.Seq import Seq

_BASES = "ACGT"


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return str(Seq(sequence).reverse_complement())


def gc_fraction(sequence: str) -> float:
    """Compute GC fraction of a sequence."""
    if not sequence:
        return 0.5
    upper = sequence.upper()
    gc = upper.count("G") + upper.count("C")
    return gc / len(sequence)


def nb_params_from_mean_variance(
    mean: float, variance: float,
) -> tuple[str, dict[str, float]]:
    """Convert mean/variance to NegativeBinomial or Poisson params.

    Returns
    -------
    dist_name : 'nb' or 'poisson'
    params : dict with keys appropriate for the distribution

    """
    if variance <= mean:
        return "poisson", {"rate": mean}
    r = mean ** 2 / (variance - mean)
    p = 1.0 - mean / variance
    return "nb", {"total_count": r, "probs": p}


def lognormal_params_from_mean_variance(
    mean: float, variance: float,
) -> tuple[float, float]:
    """Convert desired mean/variance to LogNormal mu_ln, sigma_ln."""
    sigma2_ln = math.log(1 + variance / mean ** 2)
    mu_ln = math.log(mean) - sigma2_ln / 2
    sigma_ln = math.sqrt(sigma2_ln)
    return mu_ln, sigma_ln
