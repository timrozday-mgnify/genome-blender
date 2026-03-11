#!/usr/bin/env python3
"""Report read-length summary statistics from FASTA/FASTQ files.

Usage::

    python scripts/reads_summary.py reads.fastq
    python scripts/reads_summary.py reads.fq.gz -n 10000
    python scripts/reads_summary.py genome.fasta --all
"""

from __future__ import annotations

import gzip
import math
from collections.abc import Iterator
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)

app = typer.Typer()

_PROGRESS_COLUMNS = (
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("[cyan]{task.elapsed:.1f}s"),
)


def _open_file(path: Path) -> Iterator[str]:
    """Yield lines from a plain or gzipped text file.

    Args:
        path: Path to the file (gzipped if the name ends
            with ``.gz``).

    Yields:
        Stripped lines from the file.
    """
    opener = (
        gzip.open if path.name.endswith(".gz") else open
    )
    with opener(path, "rt") as fh:
        for line in fh:
            yield line.rstrip("\n")


def _detect_format(path: Path) -> str:
    """Detect whether a file is FASTA or FASTQ.

    Peeks at the first non-empty line and checks the leading
    character (``>`` for FASTA, ``@`` for FASTQ).

    Args:
        path: Path to the reads file.

    Returns:
        ``"fasta"`` or ``"fastq"``.

    Raises:
        typer.BadParameter: If the format cannot be determined.
    """
    for line in _open_file(path):
        if not line:
            continue
        if line.startswith(">"):
            return "fasta"
        if line.startswith("@"):
            return "fastq"
        break
    raise typer.BadParameter(
        f"Cannot determine format of {path}: "
        "first record does not start with '>' or '@'"
    )


def _read_lengths_fastq(
    path: Path,
    max_reads: int | None,
) -> list[int]:
    """Return read lengths from a FASTQ file.

    FASTQ records are groups of four lines: header, sequence,
    separator, quality.

    Args:
        path: Path to the FASTQ file.
        max_reads: Maximum number of reads to parse, or
            ``None`` for all.

    Returns:
        List of sequence lengths.
    """
    lengths: list[int] = []
    total = max_reads if max_reads is not None else None
    lines = _open_file(path)

    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Reading FASTQ", total=total,
        )
        for header in lines:
            if not header.startswith("@"):
                continue
            seq = next(lines, "")
            next(lines, "")  # separator (+)
            next(lines, "")  # quality
            lengths.append(len(seq))
            progress.advance(task)
            if (
                max_reads is not None
                and len(lengths) >= max_reads
            ):
                break

    return lengths


def _read_lengths_fasta(
    path: Path,
    max_reads: int | None,
) -> list[int]:
    """Return read/sequence lengths from a FASTA file.

    FASTA records start with ``>`` and the sequence may span
    multiple lines until the next ``>`` or end of file.

    Args:
        path: Path to the FASTA file.
        max_reads: Maximum number of sequences to parse,
            or ``None`` for all.

    Returns:
        List of sequence lengths.
    """
    lengths: list[int] = []
    total = max_reads if max_reads is not None else None
    current_len = 0
    in_record = False

    with Progress(*_PROGRESS_COLUMNS) as progress:
        task = progress.add_task(
            "Reading FASTA", total=total,
        )
        for line in _open_file(path):
            if line.startswith(">"):
                if in_record:
                    lengths.append(current_len)
                    progress.advance(task)
                    if (
                        max_reads is not None
                        and len(lengths) >= max_reads
                    ):
                        return lengths
                current_len = 0
                in_record = True
            elif in_record:
                current_len += len(line)

        if in_record:
            lengths.append(current_len)
            progress.advance(task)

    return lengths


def print_summary(lengths: list[int]) -> None:
    """Print read-length summary statistics to stdout.

    Args:
        lengths: List of read/sequence lengths.
    """
    if not lengths:
        print("No reads found.")
        return

    n = len(lengths)
    total_bp = sum(lengths)
    mean = total_bp / n
    variance = sum((x - mean) ** 2 for x in lengths) / n
    sorted_lens = sorted(lengths)
    median = (
        sorted_lens[n // 2]
        if n % 2 == 1
        else (sorted_lens[n // 2 - 1] + sorted_lens[n // 2]) / 2
    )

    # N50: smallest length L such that reads of length >= L
    # cover at least half the total bases
    cumulative = 0
    n50 = sorted_lens[-1]
    for length in reversed(sorted_lens):
        cumulative += length
        if cumulative >= total_bp / 2:
            n50 = length
            break

    print(f"Reads:      {n:,}")
    print(f"Total bp:   {total_bp:,}")
    print(f"Min:        {sorted_lens[0]:,}")
    print(f"Max:        {sorted_lens[-1]:,}")
    print(f"Mean:       {mean:,.1f}")
    print(f"Median:     {median:,.1f}")
    print(f"Std dev:    {math.sqrt(variance):,.1f}")
    print(f"N50:        {n50:,}")


@app.command()
def main(
    reads: Annotated[Path, typer.Argument(
        help="Path to a FASTA or FASTQ file "
        "(optionally gzipped)",
        exists=True,
        dir_okay=False,
    )],
    n: Annotated[int, typer.Option(
        "-n", "--num-reads",
        help="Number of reads to sample from the start "
        "of the file",
    )] = 10_000,
    all_reads: Annotated[bool, typer.Option(
        "--all",
        help="Read the entire file instead of the "
        "first n reads",
    )] = False,
) -> None:
    """Report read-length statistics from a reads file."""
    fmt = _detect_format(reads)
    max_reads = None if all_reads else n

    if fmt == "fastq":
        lengths = _read_lengths_fastq(reads, max_reads)
    else:
        lengths = _read_lengths_fasta(reads, max_reads)

    label = "all" if all_reads else f"first {n:,}"
    print(f"\n{reads.name} ({fmt}, {label} reads):")
    print_summary(lengths)


if __name__ == "__main__":
    app()
