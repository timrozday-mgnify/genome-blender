#!/usr/bin/env python3
"""Estimate insert-size distribution from PE-combo minimizer containment rates.

Given sampled paths (JSONL from ``pe_path_sample.py``) and a rust-mdbg
PE-combo LMDB index, fits a log-normal insert-size model using the same
Bayesian containment-based approach as ``asf_sample.py``.

Outputs a JSON file with both minimizer-space parameters (top-level
``mu_log`` / ``sigma_log``) and bp-space deconvolved estimates (nested
``bp_space`` dict).  The top-level keys are compatible with the
``--insert-size-json`` argument of ``reconstruct_sequences.py``.

Units note
----------
``pe_path_sample.py`` paths have ``distances`` in minimizer-count space
(always 1 for adjacent minimizers), identical to ``asf_sample.py``
``PathResult`` objects.  The bin edges (``--insert-size-bins``) are treated
as minimizer-count thresholds in the containment kernel; ``--read-length``
(bp) is scaled to minimizer units using ``--density`` before being passed to
the kernel.  This exactly mirrors ``asf_sample._run_insert_size_estimation``.

Bp-space deconvolution
-----------------------
When ``--inference nuts`` is used, the Poisson-LogNormal variance
decomposition is applied to posterior samples via
``asf_sample._deconvolve_to_bp_space``.

When ``--inference map`` is used there are no posterior samples, so the
deconvolution is applied analytically to the MAP point estimates:
    μ_bp   = μ_m  − log(density)
    σ²_bp  = max(σ²_m − 1/exp(μ_m), ε)
"""
from __future__ import annotations

import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

# Import helpers and inference functions from sibling script.
sys.path.insert(0, str(Path(__file__).parent))
from asf_sample import (  # noqa: E402
    PathResult,
    _DEFAULT_INSERT_BINS,
    _build_path_bin_sketches,
    _deconvolve_to_bp_space,
    _open_pe_combo_lmdb,
    estimate_fragment_length,
    estimate_fragment_length_map,
)

app = typer.Typer(add_completion=False)
log = logging.getLogger(__name__)

_PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
)

_DEFAULT_PE_COMBO_DENSITY: float = 0.05
_DEFAULT_MIN_PATH_HASHES: int = 50
_DEFAULT_READ_LENGTH: float = 150.0
_DEFAULT_INSERT_SIZE_PATHS: int = 500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_paths(paths_jsonl: Path, progress: Progress | None = None) -> list[PathResult]:
    """Load paths from a ``pe_path_sample.py`` JSONL file as ``PathResult`` objects."""
    task = None
    if progress is not None:
        task = progress.add_task(f"Loading paths from {paths_jsonl.name} …", total=None)
    results: list[PathResult] = []
    with paths_jsonl.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            results.append(PathResult(
                minimizer_ids=obj["minimizer_ids"],
                distances=obj["distances"],
                support=obj["support"],
            ))
            if progress is not None and task is not None:
                progress.advance(task)
    if progress is not None and task is not None:
        progress.update(
            task,
            description=f"Loaded {len(results):,} paths from {paths_jsonl.name}",
            completed=len(results),
            total=len(results),
        )
    log.info("Loaded %d paths from %s", len(results), paths_jsonl)
    return results


def _map_bp_space(mu_log: float, sigma_log: float, density: float) -> dict[str, object]:
    """Analytically deconvolve MAP point estimates to bp space.

    Used when NUTS posterior samples are unavailable.  Applies the
    Poisson-LogNormal variance decomposition directly on the point estimates.
    """
    mu_bp = mu_log - math.log(density)
    sigma_noise_sq = 1.0 / math.exp(mu_log)
    sigma_bp_sq = max(sigma_log ** 2 - sigma_noise_sq, 1e-9)
    sigma_bp = math.sqrt(sigma_bp_sq)
    noise_fraction = min(sigma_noise_sq / max(sigma_log ** 2, 1e-30), 1.0)
    median_bp = math.exp(mu_bp)
    mean_bp = math.exp(mu_bp + sigma_bp_sq / 2.0)
    return {
        "mu_bp": mu_bp,
        "sigma_bp": sigma_bp,
        "median_bp": median_bp,
        "mean_bp": mean_bp,
        "noise_fraction": noise_fraction,
        "density": density,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    prefix: Annotated[Path, typer.Argument(
        help="rust-mdbg output prefix (used to locate PE-combo LMDB files).",
    )],
    paths_jsonl: Annotated[Path, typer.Argument(
        help="JSONL file of sampled paths from pe_path_sample.py.",
    )],
    density: Annotated[float, typer.Option(
        "--density",
        help="Minimizer density (minimizers per bp); must match rust-mdbg indexing.",
    )],
    output: Annotated[Path | None, typer.Option(
        "--output", "-o",
        help="Output JSON file (default: stdout).",
    )] = None,
    inference: Annotated[Literal["map", "nuts"], typer.Option(
        "--inference",
        help="Inference backend.  'map': fast MAP point estimate via SVI.  "
             "'nuts': full Bayesian posterior via NUTS (slower, enables bp-space "
             "credible intervals).",
    )] = "map",
    insert_size_bins: Annotated[str, typer.Option(
        "--insert-size-bins",
        help="Comma-separated bin edges in basepairs.  Scaled to minimizer units "
             "internally using --density.  Should cover the expected PE insert size range.",
    )] = _DEFAULT_INSERT_BINS,
    pe_combo_density: Annotated[float, typer.Option(
        "--pe-combo-density",
        help="PE combo thinning density; must match rust-mdbg --pe-combo-density.",
    )] = _DEFAULT_PE_COMBO_DENSITY,
    min_path_hashes: Annotated[int, typer.Option(
        "--min-path-hashes",
        help="Minimum combo hashes per path-bin to include in inference.",
        min=1,
    )] = _DEFAULT_MIN_PATH_HASHES,
    read_length: Annotated[float, typer.Option(
        "--read-length",
        help="Estimated read length in bp; converted to minimizer units internally.",
    )] = _DEFAULT_READ_LENGTH,
    combo_max_distance: Annotated[int | None, typer.Option(
        "--combo-max-distance",
        help="The --combo-max-distance value used in rust-mdbg (optional).",
    )] = None,
    insert_size_paths: Annotated[int, typer.Option(
        "--insert-size-paths",
        help="Maximum number of paths used for estimation (random subsample if more).",
        min=1,
    )] = _DEFAULT_INSERT_SIZE_PATHS,
    num_steps: Annotated[int, typer.Option(
        "--num-steps",
        help="SVI optimisation steps for MAP inference.  Increase if the "
             "prior is far from the true insert size.",
        min=1,
    )] = 4000,
    seed: Annotated[int, typer.Option(
        "--seed",
        help="Random seed for path subsampling.",
    )] = 42,
    verbose: Annotated[bool, typer.Option(
        "--verbose", "-v",
        help="Enable debug logging.",
    )] = False,
) -> None:
    """Estimate insert size via PE-combo minimizer containment.

    Fits a log-normal insert-size model using PE-combo containment rates
    computed from sampled paths.  Outputs minimizer-space parameters at the
    top level (compatible with reconstruct_sequences.py --insert-size-json)
    and bp-space deconvolved estimates under a nested 'bp_space' key.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    rng = random.Random(seed)

    with Progress(*_PROGRESS_COLUMNS) as progress:

        # --- Load paths ----------------------------------------------------------
        all_paths = _load_paths(paths_jsonl, progress=progress)
        if not all_paths:
            typer.echo("ERROR: no paths loaded from JSONL.", err=True)
            raise typer.Exit(1)

        # Subsample if necessary.
        if len(all_paths) > insert_size_paths:
            use_paths = rng.sample(all_paths, insert_size_paths)
            log.info(
                "Subsampled %d paths from %d for estimation",
                insert_size_paths, len(all_paths),
            )
        else:
            use_paths = all_paths

        # --- Build path-bin sketches ---------------------------------------------
        # bin_distances are user-supplied in bp; scale to minimizer-count units
        # because pe_path_sample.py distances are always 1 (minimizer steps).
        bin_distances_bp = [float(x) for x in insert_size_bins.split(",")]
        bin_distances_m = [b * density for b in bin_distances_bp]
        log.info(
            "Bin edges: %s bp  →  %s minimizers",
            [int(b) for b in bin_distances_bp],
            [round(b, 2) for b in bin_distances_m],
        )
        sketch_task = progress.add_task(
            f"Building combo sketches ({len(use_paths):,} paths, "
            f"{len(bin_distances_m) - 1} bins) …",
            total=None,
        )
        path_bin_sketches = _build_path_bin_sketches(use_paths, bin_distances_m, pe_combo_density)
        progress.update(sketch_task, description="Built path-bin sketches", completed=1, total=1)
        log.info(
            "Built sketches for %d paths, %d bins, pe_combo_density=%.4f",
            len(use_paths), len(bin_distances_m) - 1, pe_combo_density,
        )

        # --- Open PE combo LMDB --------------------------------------------------
        open_task = progress.add_task("Opening PE combo LMDB …", total=None)
        pe_shards = _open_pe_combo_lmdb(prefix)
        progress.update(
            open_task,
            description=f"Opened {len(pe_shards)} PE combo shard(s)",
            completed=1, total=1,
        )
        log.info("PE combo n_shards=%d", len(pe_shards))

        # Convert read length to minimizer units (kernel consistency).
        read_length_mers = read_length * density
        log.info(
            "read_length=%.1f bp → %.2f minimizers (density=%.4f)",
            read_length, read_length_mers, density,
        )

        # --- Run inference -------------------------------------------------------
        is_dict: dict[str, object]
        raw_samples: dict | None = None

        if inference == "map":
            inf_task = progress.add_task("Running MAP inference …", total=None)
            result_map = estimate_fragment_length_map(
                pe_shards, path_bin_sketches, bin_distances_m,
                min_path_hashes_per_bin=min_path_hashes,
                num_steps=num_steps,
                read_length_bp=read_length_mers,
                combo_max_distance=combo_max_distance,
            )
            progress.update(inf_task, description="MAP inference complete", completed=1, total=1)
            log.info(
                "MAP: median=%.0f  mean=%.0f  mu_log=%.3f  sigma_log=%.3f  "
                "rho=%.4f  norm=%.4f  bins_used=%d  signal_reliable=%s",
                result_map.median, result_map.mean,
                result_map.mu_log, result_map.sigma_log,
                result_map.rho, result_map.norm,
                result_map.n_bins_used, result_map.signal_reliable,
            )
            is_dict = {
                "method": "map",
                "mu_log": result_map.mu_log,
                "sigma_log": result_map.sigma_log,
                "rho": result_map.rho,
                "norm": result_map.norm,
                "median_bp": result_map.median,
                "mean_bp": result_map.mean,
                "n_bins_used": result_map.n_bins_used,
                "signal_reliable": result_map.signal_reliable,
                "loss_final": result_map.loss_final,
            }
        else:
            inf_task = progress.add_task("Running NUTS inference …", total=None)
            result_nuts = estimate_fragment_length(
                pe_shards, path_bin_sketches, bin_distances_m,
                min_path_hashes_per_bin=min_path_hashes,
                inference=inference,
                read_length_bp=read_length_mers,
                combo_max_distance=combo_max_distance,
            )
            progress.update(inf_task, description="NUTS inference complete", completed=1, total=1)
            raw_samples = result_nuts.raw_samples
            log.info(
                "NUTS: median=%.0f  mean=%.0f  mu_log=%.3f [%.3f, %.3f]  "
                "sigma_log=%.3f  rho=%.4f  bins_used=%d  signal_reliable=%s",
                result_nuts.median, result_nuts.mean,
                result_nuts.mu_log, *result_nuts.mu_log_ci,
                result_nuts.sigma_log, result_nuts.rho,
                result_nuts.n_bins_used, result_nuts.signal_reliable,
            )
            is_dict = {
                "method": result_nuts.inference,
                "mu_log": result_nuts.mu_log,
                "sigma_log": result_nuts.sigma_log,
                "mu_log_ci": list(result_nuts.mu_log_ci),
                "sigma_log_ci": list(result_nuts.sigma_log_ci),
                "rho": result_nuts.rho,
                "norm": result_nuts.norm,
                "median_bp": result_nuts.median,
                "mean_bp": result_nuts.mean,
                "n_bins_used": result_nuts.n_bins_used,
                "signal_reliable": result_nuts.signal_reliable,
            }

        # --- Bp-space deconvolution ----------------------------------------------
        bp_task = progress.add_task("Computing bp-space deconvolution …", total=None)
        if raw_samples is not None:
            # NUTS: full posterior deconvolution with credible intervals.
            bp_est = _deconvolve_to_bp_space(raw_samples, density)
            is_dict["bp_space"] = {
                "mu_bp": bp_est.mu_bp,
                "sigma_bp": bp_est.sigma_bp,
                "mu_bp_ci": list(bp_est.mu_bp_ci),
                "sigma_bp_ci": list(bp_est.sigma_bp_ci),
                "median_bp": bp_est.median_bp,
                "mean_bp": bp_est.mean_bp,
                "median_bp_ci": list(bp_est.median_bp_ci),
                "noise_fraction": bp_est.noise_fraction,
                "density": bp_est.density,
            }
            log.info(
                "Bp-space (NUTS): median=%.1f bp (95%% CI: %.1f–%.1f)  "
                "sigma_bp=%.3f  noise_fraction=%.1f%%",
                bp_est.median_bp,
                bp_est.median_bp_ci[0], bp_est.median_bp_ci[1],
                bp_est.sigma_bp, bp_est.noise_fraction * 100,
            )
        else:
            # MAP: analytic point estimate deconvolution (no credible intervals).
            mu_log = float(is_dict["mu_log"])  # type: ignore[arg-type]
            sigma_log = float(is_dict["sigma_log"])  # type: ignore[arg-type]
            is_dict["bp_space"] = _map_bp_space(mu_log, sigma_log, density)
            log.info(
                "Bp-space (MAP analytic): median=%.1f bp  sigma_bp=%.3f  "
                "noise_fraction=%.1f%%",
                float(is_dict["bp_space"]["median_bp"]),  # type: ignore[index]
                float(is_dict["bp_space"]["sigma_bp"]),   # type: ignore[index]
                float(is_dict["bp_space"]["noise_fraction"]) * 100,  # type: ignore[index]
            )
        progress.update(bp_task, description="Bp-space deconvolution complete", completed=1, total=1)

        # --- Close LMDB ----------------------------------------------------------
        for env, _db in pe_shards:
            env.close()

    # --- Write output ------------------------------------------------------------
    text = json.dumps(is_dict, indent=2) + "\n"
    if output is None:
        print(text, end="")
    else:
        output.write_text(text)
        log.info("Written to %s", output)


if __name__ == "__main__":
    app()
