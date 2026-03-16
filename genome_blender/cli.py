"""CLI entry point and pipeline orchestration.

Provide the ``main`` Typer command and the chunked pipeline runner.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

from rich.console import Group
from rich.live import Live
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

import click.core
import pysam
import torch
import typer
import yaml

from genome_blender._progress import (
    progress_task,
    set_inner_progress,
)
from genome_blender.error_model import (
    apply_error_model,
    build_quality_calibration,
    default_illumina_profile,
    default_nanopore_profile,
    default_pacbio_profile,
)
from genome_blender.fragments import (
    amplicon_fragments,
    sample_fragments,
)
from genome_blender.genomes import load_genomes
from genome_blender.io import (
    build_bam_header,
    write_bam_chunk,
    write_fastq,
)
from genome_blender.models import (
    ErrorModel,
    ErrorModelProfile,
    QualityCalibration,
    QualityCalibrationModel,
    ReadBatch,
)
from genome_blender.reads import generate_reads

logger = logging.getLogger(__name__)

app = typer.Typer()

# Mapping from YAML key names to enum classes needing conversion
_ENUM_PARAMS: dict[str, type[Enum]] = {
    "error_model": ErrorModel,
    "quality_calibration_model": QualityCalibrationModel,
}


def _load_yaml_config(
    config_path: Path,
) -> dict[str, object]:
    """Load a YAML config file and normalise keys to snake_case.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dict mapping parameter names (snake_case) to values.

    Raises:
        typer.BadParameter: If the file doesn't contain a YAML
            mapping.
    """
    with open(config_path) as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise typer.BadParameter(
            f"Config file must contain a YAML mapping, "
            f"got {type(data).__name__}"
        )
    return {k.replace("-", "_"): v for k, v in data.items()}


def _apply_yaml_config(
    ctx: typer.Context,
    config: dict[str, object],
) -> None:
    """Override default-valued CLI params with YAML config values.

    Only parameters whose source is DEFAULT (i.e. not explicitly
    provided on the command line) are overridden.
    """
    for key, value in config.items():
        source = ctx.get_parameter_source(key)
        if source is not click.core.ParameterSource.COMMANDLINE:
            if key in _ENUM_PARAMS and isinstance(value, str):
                value = _ENUM_PARAMS[key](value)
            elif key == "input_csv" and isinstance(value, str):
                value = Path(value)
            ctx.params[key] = value


@app.command()
def main(
    ctx: typer.Context,
    config: Annotated[Path | None, typer.Option(
        help="YAML config file "
        "(CLI options override config values)",
    )] = None,
    verbose: Annotated[bool, typer.Option(
        "--verbose/--no-verbose",
        help="Enable verbose (DEBUG) logging",
    )] = False,
    no_ansi: Annotated[bool, typer.Option(
        "--no-ansi",
        help="Disable ANSI escape codes "
        "(progress bars, colours)",
    )] = False,
    input_csv: Annotated[Path | None, typer.Option(
        help="CSV with columns: "
        "genome_id, fasta_path, abundance",
    )] = None,
    num_reads: Annotated[int | None, typer.Option(
        help="Total number of reads to generate",
    )] = None,
    output_prefix: Annotated[str | None, typer.Option(
        help="Output file prefix",
    )] = None,
    fragment_mean: Annotated[float, typer.Option(
        help="Mean fragment length",
    )] = 300.0,
    fragment_variance: Annotated[float, typer.Option(
        help="Variance of fragment length "
        "(Negative Binomial)",
    )] = 300.0,
    read_length_mean: Annotated[float, typer.Option(
        help="Mean read length (LogNormal)",
    )] = 150.0,
    read_length_variance: Annotated[float, typer.Option(
        help="Variance of read length (LogNormal)",
    )] = 10.0,
    gc_bias_strength: Annotated[float, typer.Option(
        help="GC bias strength; 0 = no bias",
    )] = 0.0,
    paired_end: Annotated[bool, typer.Option(
        "--paired-end/--single-end",
        help="Generate paired-end or single-end reads "
        "(default: single-end)",
    )] = False,
    seed: Annotated[int | None, typer.Option(
        help="Random seed for reproducibility",
    )] = None,
    error_model: Annotated[ErrorModel, typer.Option(
        help="Sequencing error model profile",
        case_sensitive=False,
    )] = ErrorModel.none,
    quality_calibration_model: Annotated[
        QualityCalibrationModel, typer.Option(
            help="Quality-score-to-error-rate "
            "calibration model",
            case_sensitive=False,
        )
    ] = QualityCalibrationModel.phred,
    qcal_variability: Annotated[float, typer.Option(
        help="Per-run noise multiplier for calibration "
        "parameters; 0 = no noise",
    )] = 0.0,
    qcal_intercept: Annotated[float, typer.Option(
        help="Log-linear model intercept (log10 scale)",
    )] = -0.3,
    qcal_slope: Annotated[float, typer.Option(
        help="Log-linear model slope",
    )] = -0.08,
    qcal_floor: Annotated[float, typer.Option(
        help="Minimum error probability "
        "(log-linear and sigmoid)",
    )] = 1e-7,
    qcal_ceiling: Annotated[float, typer.Option(
        help="Maximum error probability "
        "(log-linear and sigmoid)",
    )] = 0.5,
    qcal_steepness: Annotated[float, typer.Option(
        help="Sigmoid model steepness",
    )] = 0.25,
    qcal_midpoint: Annotated[float, typer.Option(
        help="Sigmoid model midpoint "
        "(Q-score at inflection)",
    )] = 15.0,
    error_rate_scale: Annotated[float, typer.Option(
        help="Multiplier applied to error probabilities "
        "after quality calibration; "
        "<1 reduces errors, >1 increases them",
    )] = 1.0,
    long_read: Annotated[bool, typer.Option(
        "--long-read/--no-long-read",
        help="Sequence entire fragments "
        "(read length params ignored)",
    )] = False,
    amplicon: Annotated[bool, typer.Option(
        "--amplicon/--no-amplicon",
        help="Treat input sequences as amplicons "
        "(no shearing); replicate proportionally "
        "to abundance",
    )] = False,
    chunk_size: Annotated[int, typer.Option(
        help="Number of fragments to process per chunk "
        "to limit memory usage",
    )] = 100_000,
    compress: Annotated[bool, typer.Option(
        "--compress/--no-compress",
        help="Write gzip-compressed FASTQ output "
        "(.fastq.gz).  Enabled by default; use "
        "--no-compress for plain-text FASTQ.",
    )] = True,
) -> None:
    """Generate simulated WGS reads from reference genomes."""
    # Apply YAML config: values from the file fill in anything
    # not explicitly provided on the command line.
    if config is not None:
        _apply_yaml_config(
            ctx, _load_yaml_config(config),
        )
        p = ctx.params
        no_ansi = p.get("no_ansi", False)
        input_csv = (
            Path(p["input_csv"])
            if p.get("input_csv") else None
        )
        num_reads = p.get("num_reads")
        output_prefix = p.get("output_prefix")
        fragment_mean = p["fragment_mean"]
        fragment_variance = p["fragment_variance"]
        read_length_mean = p["read_length_mean"]
        read_length_variance = p["read_length_variance"]
        gc_bias_strength = p["gc_bias_strength"]
        paired_end = p["paired_end"]
        seed = p.get("seed")
        error_model = ErrorModel(p["error_model"])
        quality_calibration_model = (
            QualityCalibrationModel(
                p["quality_calibration_model"]
            )
        )
        qcal_variability = p["qcal_variability"]
        qcal_intercept = p["qcal_intercept"]
        qcal_slope = p["qcal_slope"]
        qcal_floor = p["qcal_floor"]
        qcal_ceiling = p["qcal_ceiling"]
        qcal_steepness = p["qcal_steepness"]
        qcal_midpoint = p["qcal_midpoint"]
        error_rate_scale = p.get("error_rate_scale", 1.0)
        long_read = p.get("long_read", False)
        amplicon = p["amplicon"]
        chunk_size = p.get("chunk_size", 1_000_000)

    # Validate required parameters
    if input_csv is None:
        raise typer.BadParameter(
            "--input-csv is required (via CLI or config)"
        )
    if num_reads is None:
        raise typer.BadParameter(
            "--num-reads is required (via CLI or config)"
        )
    if output_prefix is None:
        raise typer.BadParameter(
            "--output-prefix is required (via CLI or config)"
        )
    if long_read and paired_end:
        raise typer.BadParameter(
            "--long-read and --paired-end are "
            "mutually exclusive"
        )

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    logger.debug("Verbose logging enabled")

    _log_parameters(
        input_csv=input_csv,
        num_reads=num_reads,
        output_prefix=output_prefix,
        fragment_mean=fragment_mean,
        fragment_variance=fragment_variance,
        read_length_mean=read_length_mean,
        read_length_variance=read_length_variance,
        gc_bias_strength=gc_bias_strength,
        paired_end=paired_end,
        long_read=long_read,
        amplicon=amplicon,
        chunk_size=chunk_size,
        error_model=error_model,
        quality_calibration_model=quality_calibration_model,
        seed=seed,
    )

    # Set up RNG
    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)
        torch.manual_seed(seed)
        logger.info("Random seed: %d", seed)
    else:
        rng.seed()
        logger.info("Using random seed")

    # Resolve error model profile
    profile = _resolve_error_profile(error_model)

    # Build quality calibration model
    calibration = build_quality_calibration(
        model_name=quality_calibration_model.value,
        variability=qcal_variability,
        rng=rng,
        intercept=qcal_intercept,
        slope=qcal_slope,
        floor=qcal_floor,
        ceiling=qcal_ceiling,
        steepness=qcal_steepness,
        midpoint=qcal_midpoint,
    )

    # Determine whether to show progress bars
    try:
        is_tty = sys.stderr.isatty()
    except ValueError:
        is_tty = False
    disable_progress = no_ansi or not is_tty

    progress_columns = (
        SpinnerColumn(),
        TextColumn(
            "[progress.description]{task.description}",
        ),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[cyan]{task.elapsed:.1f}s"),
    )
    outer_progress = Progress(*progress_columns)
    inner_progress = Progress(*progress_columns)

    pipeline_kwargs = dict(
        input_csv=input_csv,
        num_reads=num_reads,
        output_prefix=output_prefix,
        fragment_mean=fragment_mean,
        fragment_variance=fragment_variance,
        read_length_mean=read_length_mean,
        read_length_variance=read_length_variance,
        gc_bias_strength=gc_bias_strength,
        paired_end=paired_end,
        long_read=long_read,
        amplicon=amplicon,
        chunk_size=chunk_size,
        rng=rng,
        profile=profile,
        calibration=calibration,
        error_rate_scale=error_rate_scale,
        compress=compress,
    )

    if disable_progress:
        set_inner_progress(None)
        _run_pipeline(
            **pipeline_kwargs,
            outer_progress=None,
            inner_progress=None,
        )
    else:
        saved_level = logging.root.level
        logging.root.setLevel(logging.WARNING)
        with Live(
            Group(outer_progress, inner_progress),
            refresh_per_second=10,
        ):
            set_inner_progress(inner_progress)
            _run_pipeline(
                **pipeline_kwargs,
                outer_progress=outer_progress,
                inner_progress=inner_progress,
            )
            set_inner_progress(None)
        logging.root.setLevel(saved_level)

    logger.info("Done.")


def _log_parameters(
    *,
    input_csv: Path,
    num_reads: int,
    output_prefix: str,
    fragment_mean: float,
    fragment_variance: float,
    read_length_mean: float,
    read_length_variance: float,
    gc_bias_strength: float,
    paired_end: bool,
    long_read: bool,
    amplicon: bool,
    chunk_size: int,
    error_model: ErrorModel,
    quality_calibration_model: QualityCalibrationModel,
    seed: int | None,
) -> None:
    """Log all resolved parameters at DEBUG level."""
    logger.debug("Parameters:")
    logger.debug("  input_csv       = %s", input_csv)
    logger.debug("  num_reads       = %d", num_reads)
    logger.debug("  output_prefix   = %s", output_prefix)
    logger.debug(
        "  fragment_mean   = %.1f", fragment_mean,
    )
    logger.debug(
        "  fragment_var    = %.1f", fragment_variance,
    )
    logger.debug(
        "  read_len_mean   = %.1f", read_length_mean,
    )
    logger.debug(
        "  read_len_var    = %.1f", read_length_variance,
    )
    logger.debug(
        "  gc_bias         = %.2f", gc_bias_strength,
    )
    logger.debug("  paired_end      = %s", paired_end)
    logger.debug("  long_read       = %s", long_read)
    logger.debug("  amplicon        = %s", amplicon)
    logger.debug("  chunk_size      = %d", chunk_size)
    logger.debug(
        "  error_model     = %s", error_model.value,
    )
    logger.debug(
        "  quality_cal     = %s",
        quality_calibration_model.value,
    )
    logger.debug("  seed            = %s", seed)


def _resolve_error_profile(
    error_model: ErrorModel,
) -> ErrorModelProfile | None:
    """Return the error model profile, or None for 'none'."""
    profile_map = {
        "illumina": default_illumina_profile,
        "pacbio": default_pacbio_profile,
        "nanopore": default_nanopore_profile,
    }
    error_model_str = error_model.value
    profile = (
        profile_map[error_model_str]()
        if error_model_str != "none" else None
    )
    if profile is not None:
        logger.debug(
            "Error profile: %d HMM states, "
            "sub=%.0f%% ins=%.0f%% del=%.0f%%",
            profile.num_states,
            profile.substitution_ratio * 100,
            profile.insertion_ratio * 100,
            profile.deletion_ratio * 100,
        )
    return profile


def _run_pipeline(
    *,
    input_csv: Path,
    num_reads: int,
    output_prefix: str,
    fragment_mean: float,
    fragment_variance: float,
    read_length_mean: float,
    read_length_variance: float,
    gc_bias_strength: float,
    paired_end: bool,
    long_read: bool,
    amplicon: bool,
    chunk_size: int,
    rng: torch.Generator,
    profile: ErrorModelProfile | None,
    calibration: QualityCalibration | None,
    error_rate_scale: float,
    compress: bool,
    outer_progress: Progress | None,
    inner_progress: Progress | None,
) -> None:
    """Run the read-generation pipeline.

    Extracted from ``main()`` so the Live display context can
    be set up before this function is called.
    """
    genomes, abundances = load_genomes(input_csv)
    logger.info("Loaded %d genomes", len(genomes))
    for gid, abd in abundances.items():
        total_bp = sum(
            len(r.seq) for r in genomes[gid]
        )
        logger.debug(
            "  %s: abundance=%.4f, %d contigs, %d bp total",
            gid, abd, len(genomes[gid]), total_bp,
        )

    num_fragments = (
        num_reads // 2 if paired_end else num_reads
    )

    chunk_starts = list(
        range(0, num_fragments, chunk_size),
    )
    chunk_counts = [
        min(chunk_size, num_fragments - start)
        for start in chunk_starts
    ]
    num_chunks = len(chunk_counts)
    logger.info(
        "Processing %d fragments in %d chunk(s) of up to %d",
        num_fragments, num_chunks, chunk_size,
    )

    output_dir = Path(output_prefix).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    bam_path = Path(f"{output_prefix}.bam")
    fq_ext = ".fastq.gz" if compress else ".fastq"
    if paired_end:
        r1_path = Path(f"{output_prefix}_R1{fq_ext}")
        r2_path = Path(f"{output_prefix}_R2{fq_ext}")
    else:
        out_path = Path(f"{output_prefix}{fq_ext}")

    header, ref_name_to_idx = build_bam_header(genomes)

    chunk_task = None
    if outer_progress is not None:
        chunk_task = outer_progress.add_task(
            "Chunks", total=num_chunks,
        )

    # Set up fragment generator (amplicon vs shearing)
    if amplicon:
        def make_fragments(n: int) -> list:
            return amplicon_fragments(
                genomes=genomes,
                abundances=abundances,
                num_fragments=n,
                rng=rng,
            )
    else:
        def make_fragments(n: int) -> list:
            return sample_fragments(
                genomes=genomes,
                abundances=abundances,
                num_fragments=n,
                fragment_mean=fragment_mean,
                fragment_variance=fragment_variance,
                gc_bias_strength=gc_bias_strength,
                rng=rng,
            )

    # Set up FASTQ writer (SE/PE mode)
    if paired_end:
        def write_fastqs(
            batch: ReadBatch, append: bool,
        ) -> None:
            assert batch.paired is not None
            r1_reads = [p[0] for p in batch.paired]
            r2_reads = [p[1] for p in batch.paired]
            write_fastq(r1_reads, r1_path, append=append)
            write_fastq(r2_reads, r2_path, append=append)
    else:
        def write_fastqs(
            batch: ReadBatch, append: bool,
        ) -> None:
            assert batch.single is not None
            write_fastq(
                batch.single, out_path, append=append,
            )

    with pysam.AlignmentFile(
        bam_path, "wb", header=header,
    ) as bam:
        for chunk_idx, (chunk_start, chunk_n) in enumerate(
            zip(chunk_starts, chunk_counts),
        ):
            if inner_progress is not None:
                for tid in list(inner_progress.task_ids):
                    inner_progress.remove_task(tid)

            logger.info(
                "Chunk %d/%d: %d fragments (offset %d)",
                chunk_idx + 1, num_chunks, chunk_n,
                chunk_start,
            )

            fragments = make_fragments(chunk_n)
            logger.info(
                "Generated %d fragments", len(fragments),
            )

            read_batch = generate_reads(
                fragments=fragments,
                read_length_mean=read_length_mean,
                read_length_variance=read_length_variance,
                paired_end=paired_end,
                rng=rng,
                read_index_offset=chunk_start,
                long_read=long_read,
            )
            read_batch = apply_error_model(
                read_batch, profile, rng, calibration,
                error_rate_scale,
            )

            write_fastqs(
                read_batch, append=chunk_idx > 0,
            )

            write_bam_chunk(
                bam, header, ref_name_to_idx,
                fragments, read_batch,
            )

            if (
                outer_progress is not None
                and chunk_task is not None
            ):
                outer_progress.advance(chunk_task)

    logger.info("Wrote ground-truth BAM to %s", bam_path)
