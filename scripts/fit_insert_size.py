#!/usr/bin/env python3
"""Fit a log-normal insert-size model from directly measured fragment lengths.

Two estimates are produced and written to a single JSON file:

minimizer_space
    Fragment lengths are measured as the number of minimizers in each sampled
    path (from a pe_path_sample.py JSONL file).  Because each minimizer spans
    an uncertain number of base pairs, the log-normal parameters are shifted
    from minimizer-space to basepair-space using a Poisson noise correction:

        μ_bp   = μ_m  − log(density)
        σ²_bp  = max(σ²_m − 1/exp(μ_m), ε)

    where 1/exp(μ_m) is the Poisson-placement noise variance in log-space.

bp_space
    Fragment lengths are measured directly from a FASTA file produced by
    reconstruct_sequences.py.  The log-normal parameters are fitted by MLE
    (μ = mean(log(lengths)), σ = std(log(lengths))).
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import pysam
import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

app = typer.Typer(add_completion=False)
log = logging.getLogger(__name__)

_PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class InsertSizeEstimate:
    """Log-normal insert-size estimate derived from observed fragment lengths."""

    mu_log: float
    """Log-scale mean (μ of log X)."""
    sigma_log: float
    """Log-scale standard deviation (σ of log X)."""
    median_bp: float
    """Estimated median insert size in basepairs (= exp(μ_log))."""
    mean_bp: float
    """Estimated mean insert size in basepairs (= exp(μ_log + σ²_log / 2))."""
    n_samples: int
    """Number of fragments used for the estimate."""


@dataclass
class MinimizerSpaceEstimate(InsertSizeEstimate):
    """Insert-size estimate derived from minimizer-space path lengths."""

    density: float
    """Minimizer density used for deconvolution (minimizers per bp)."""
    noise_fraction: float
    """Fraction of log-space variance attributable to Poisson placement noise."""


# ---------------------------------------------------------------------------
# Core estimation functions
# ---------------------------------------------------------------------------


def _fit_lognormal(lengths: list[int]) -> tuple[float, float]:
    """Return (mu_log, sigma_log) MLE for a log-normal fit to *lengths*.

    Uses the unbiased sample standard deviation (ddof=1).
    """
    if len(lengths) < 2:
        raise ValueError(f"Need at least 2 observations; got {len(lengths)}")
    log_lengths = np.log(np.array(lengths, dtype=np.float64))
    mu_log = float(log_lengths.mean())
    sigma_log = float(log_lengths.std(ddof=1))
    return mu_log, sigma_log


def _minimizer_space_estimate(
    paths: list[dict],
    density: float,
    progress: Progress | None = None,
) -> MinimizerSpaceEstimate:
    """Fit insert-size from minimizer-count path lengths with Poisson deconvolution.

    Args:
        paths: Parsed path dicts from pe_path_sample.py JSONL.
        density: Minimizer density in minimizers per bp.
        progress: Optional Rich progress display.

    Returns:
        Insert-size estimate deconvolved to basepair space.
    """
    task = None
    if progress is not None:
        task = progress.add_task("Measuring minimizer-space path lengths …", total=len(paths))

    n_mers: list[int] = []
    for path in paths:
        n_mers.append(len(path["minimizer_ids"]))
        if progress is not None and task is not None:
            progress.advance(task)

    if progress is not None and task is not None:
        progress.update(task, description=f"Measured {len(n_mers):,} path lengths (minimizer space)")

    mu_log_m, sigma_log_m = _fit_lognormal(n_mers)

    # Poisson deconvolution: shift mean by -log(density), remove Poisson noise
    # from variance.  The Poisson noise contribution in log-space is 1/E[n_mers]
    # ≈ 1/exp(mu_log_m).
    mu_log_bp = mu_log_m - math.log(density)
    poisson_noise_var = 1.0 / math.exp(mu_log_m)
    raw_var = sigma_log_m ** 2
    sigma_log_bp = math.sqrt(max(raw_var - poisson_noise_var, 1e-9))
    noise_fraction = min(poisson_noise_var / max(raw_var, 1e-30), 1.0)

    median_bp = math.exp(mu_log_bp)
    mean_bp = math.exp(mu_log_bp + 0.5 * sigma_log_bp ** 2)

    log.info(
        "Minimizer-space estimate: mu_log_m=%.4f sigma_log_m=%.4f → "
        "mu_log_bp=%.4f sigma_log_bp=%.4f  median=%.1f bp  mean=%.1f bp  "
        "noise_fraction=%.3f",
        mu_log_m, sigma_log_m, mu_log_bp, sigma_log_bp, median_bp, mean_bp, noise_fraction,
    )

    return MinimizerSpaceEstimate(
        mu_log=mu_log_bp,
        sigma_log=sigma_log_bp,
        median_bp=median_bp,
        mean_bp=mean_bp,
        n_samples=len(n_mers),
        density=density,
        noise_fraction=noise_fraction,
    )


def _bp_space_estimate(
    fasta_path: Path,
    progress: Progress | None = None,
) -> InsertSizeEstimate:
    """Fit insert-size directly from reconstructed FASTA sequence lengths.

    Args:
        fasta_path: FASTA file produced by reconstruct_sequences.py.
        progress: Optional Rich progress display.

    Returns:
        Insert-size estimate in basepairs.
    """
    task = None
    if progress is not None:
        task = progress.add_task(f"Reading {fasta_path.name} …", total=None)

    lengths: list[int] = []
    with pysam.FastxFile(str(fasta_path)) as fh:
        for record in fh:
            if record.sequence:
                lengths.append(len(record.sequence))
            if progress is not None and task is not None:
                progress.advance(task)

    if progress is not None and task is not None:
        progress.update(
            task,
            description=f"Read {len(lengths):,} sequences from {fasta_path.name}",
            completed=len(lengths),
            total=len(lengths),
        )

    mu_log, sigma_log = _fit_lognormal(lengths)
    median_bp = math.exp(mu_log)
    mean_bp = math.exp(mu_log + 0.5 * sigma_log ** 2)

    log.info(
        "Basepair-space estimate: mu_log=%.4f sigma_log=%.4f  median=%.1f bp  mean=%.1f bp",
        mu_log, sigma_log, median_bp, mean_bp,
    )

    return InsertSizeEstimate(
        mu_log=mu_log,
        sigma_log=sigma_log,
        median_bp=median_bp,
        mean_bp=mean_bp,
        n_samples=len(lengths),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    paths_jsonl: Annotated[Path, typer.Argument(help="pe_path_sample.py JSONL output file.")],
    reconstructed: Annotated[
        Path,
        typer.Option("--reconstructed", help="Reconstructed FASTA from reconstruct_sequences.py."),
    ],
    density: Annotated[
        float,
        typer.Option("--density", help="Minimizer density (minimizers per bp) used for rust-mdbg indexing."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSON file (default: stdout)."),
    ] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging.")] = False,
) -> None:
    """Fit log-normal insert-size models from measured fragment lengths.

    Produces two estimates:

    \b
    minimizer_space — path lengths in minimizer units, deconvolved to bp
                      using a Poisson noise correction.
    bp_space        — sequence lengths from reconstructed FASTA, fitted directly.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    log.info("Loading paths from %s", paths_jsonl)
    paths: list[dict] = []
    with paths_jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                paths.append(json.loads(line))
    log.info("Loaded %d paths", len(paths))

    with Progress(*_PROGRESS_COLUMNS) as progress:
        mer_est = _minimizer_space_estimate(paths, density, progress=progress)
        bp_est = _bp_space_estimate(reconstructed, progress=progress)

    result = {
        "minimizer_space": {
            **asdict(mer_est),
            "n_paths": mer_est.n_samples,
        },
        "bp_space": {
            **asdict(bp_est),
            "n_sequences": bp_est.n_samples,
        },
    }
    # Rename n_samples key to avoid confusion in output
    result["minimizer_space"].pop("n_samples", None)
    result["bp_space"].pop("n_samples", None)

    text = json.dumps(result, indent=2)
    if output is None:
        print(text)
    else:
        output.write_text(text)
        log.info("Written to %s", output)


if __name__ == "__main__":
    app()
