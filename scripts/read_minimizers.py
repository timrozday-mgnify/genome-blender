#!/usr/bin/env python3
"""Read and decode rust-mdbg per-read minimizer files.

Supports two file formats (auto-detected by magic bytes after LZ4
decompression):

* **Binary** (``RMBG\x01`` magic) — packed little-endian integers; produced
  by the ``--dump-read-minimizers`` flag in the patched rust-mdbg build.
* **TSV** (legacy) — LZ4-compressed text ``read_id\\tids\\tpositions`` as
  written by the upstream rust-mdbg build.

Reads all ``{prefix}.*.read_minimizers`` files and the accompanying
``{prefix}.minimizer_table`` (plain-text TSV) produced by rust-mdbg
``--dump-read-minimizers``, and yields structured records.

Usage::

    python scripts/read_minimizers.py rust_mdbg_out
    python scripts/read_minimizers.py rust_mdbg_out --decode
    python scripts/read_minimizers.py rust_mdbg_out --json out.jsonl
    python scripts/read_minimizers.py rust_mdbg_out --summary
    python scripts/read_minimizers.py rust_mdbg_out --summary -n 5000
    python scripts/read_minimizers.py rust_mdbg_out --summary --json summary.json
    python scripts/read_minimizers.py rust_mdbg_out --convert
"""

from __future__ import annotations

import glob
import io
import json
import math
import struct
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import lz4.frame
import typer

app = typer.Typer()

# ---------------------------------------------------------------------------
# Binary format constants
# ---------------------------------------------------------------------------

_RM_MAGIC = b"RMBG"
_RM_VERSION = 1
_RM_HDR = _RM_MAGIC + bytes([_RM_VERSION])
_RM_HDR_LEN = len(_RM_HDR)  # 5


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ReadRecord:
    """Per-read minimizer record.

    Attributes:
        read_id: The read name from the input FASTQ/FASTA.
        minimizer_ids: Ordered list of u64 NT-hash IDs selected
            from this read.
        positions: 0-based position of each minimizer in the raw
            (pre-HPC) read sequence.
        lmers: l-mer strings corresponding to each minimizer ID,
            populated only when a lookup table is provided.
    """

    read_id: str
    minimizer_ids: list[int]
    positions: list[int]
    lmers: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Minimizer table
# ---------------------------------------------------------------------------


def load_minimizer_table(table_path: Path) -> dict[int, str]:
    """Load a minimizer_table file into a hash-to-lmer dict.

    Args:
        table_path: Path to ``{prefix}.minimizer_table``.

    Returns:
        Mapping of NT-hash (u64) to l-mer string.
    """
    lookup: dict[int, str] = {}
    with open(table_path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                lookup[int(parts[0])] = parts[1]
    return lookup


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------


def _iter_tsv_file(
    raw: bytes,
) -> Iterator[tuple[str, list[int], list[int]]]:
    """Yield ``(read_id, minimizer_ids, positions)`` from LZ4-TSV bytes.

    Args:
        raw: Decompressed content of a TSV read_minimizers file.

    Yields:
        Tuples of ``(read_id, minimizer_ids, positions)``.
    """
    for line in io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        read_id = parts[0]
        ids = [int(x) for x in parts[1].split(",") if x]
        pos = [int(x) for x in parts[2].split(",") if x]
        yield read_id, ids, pos


def _iter_binary_file(
    raw: bytes,
) -> Iterator[tuple[str, list[int], list[int]]]:
    """Yield ``(read_id, minimizer_ids, positions)`` from binary bytes.

    The binary format stores each record as::

        u32 LE  name_len
        bytes   read name (UTF-8, name_len bytes)
        u32 LE  n  (number of minimizers)
        u64 LE  minimizer_ids[0..n]
        u64 LE  positions[0..n]

    The file starts with a 5-byte header (``RMBG\\x01``) which must be
    skipped before calling this function.

    Args:
        raw: Decompressed content of a binary read_minimizers file,
            including the 5-byte header.

    Yields:
        Tuples of ``(read_id, minimizer_ids, positions)``.
    """
    mv = memoryview(raw)
    offset = _RM_HDR_LEN
    total = len(raw)
    while offset < total:
        (name_len,) = struct.unpack_from("<I", mv, offset)
        offset += 4
        name = bytes(mv[offset: offset + name_len]).decode()
        offset += name_len
        (n,) = struct.unpack_from("<I", mv, offset)
        offset += 4
        ids: list[int] = list(struct.unpack_from(f"<{n}Q", mv, offset))
        offset += n * 8
        pos: list[int] = list(struct.unpack_from(f"<{n}Q", mv, offset))
        offset += n * 8
        yield name, ids, pos


def _iter_read_minimizers_file(
    path: Path,
) -> Iterator[tuple[str, list[int], list[int]]]:
    """Yield ``(read_id, minimizer_ids, positions)`` from one file.

    Auto-detects three formats:
    1. Uncompressed binary (starts with ``RMBG`` magic directly).
    2. LZ4-compressed binary (decompresses to ``RMBG`` magic).
    3. LZ4-compressed TSV (legacy format).

    Args:
        path: Path to a ``*.read_minimizers`` file.

    Yields:
        Tuples of ``(read_id, minimizer_ids, positions)``.
    """
    data = path.read_bytes()
    if data[:4] == _RM_MAGIC:
        yield from _iter_binary_file(data)
        return
    raw = lz4.frame.decompress(data)
    if raw[:4] == _RM_MAGIC:
        yield from _iter_binary_file(raw)
    else:
        yield from _iter_tsv_file(raw)


# ---------------------------------------------------------------------------
# Binary writer
# ---------------------------------------------------------------------------


def write_binary_read_minimizers(
    path: Path,
    records: Iterable[tuple[str, list[int], list[int]]],
) -> None:
    """Write read minimizer records to *path* in packed binary format.

    The output is LZ4-compressed.  Records are streamed so the full dataset
    is never held in memory simultaneously.

    Args:
        path: Destination file path (will be overwritten if it exists).
        records: Iterable of ``(read_id, minimizer_ids, positions)`` tuples.
    """
    compressor = lz4.frame.LZ4FrameCompressor()
    with open(path, "wb") as fh:
        fh.write(compressor.begin())
        fh.write(compressor.compress(_RM_HDR))
        for name, ids, pos in records:
            name_bytes = name.encode()
            n = len(ids)
            chunk = bytearray()
            chunk += struct.pack("<I", len(name_bytes))
            chunk += name_bytes
            chunk += struct.pack("<I", n)
            if n:
                chunk += struct.pack(f"<{n}Q", *ids)
                chunk += struct.pack(f"<{n}Q", *pos)
            fh.write(compressor.compress(bytes(chunk)))
        fh.write(compressor.flush())


# ---------------------------------------------------------------------------
# Public iterator
# ---------------------------------------------------------------------------


def iter_records(
    prefix: Path,
    lookup: dict[int, str] | None = None,
) -> Iterator[ReadRecord]:
    """Yield ReadRecord objects for all reads across all thread files.

    Args:
        prefix: Output prefix used with rust-mdbg (e.g.
            ``Path("rust_mdbg_out")``).
        lookup: Optional hash-to-lmer mapping from
            :func:`load_minimizer_table`.  When provided,
            ``ReadRecord.lmers`` is populated.

    Yields:
        One :class:`ReadRecord` per input read.
    """
    pattern = str(prefix) + ".*.read_minimizers"
    files = sorted(glob.glob(pattern))
    if not files:
        typer.echo(f"No .read_minimizers files found for prefix: {prefix}", err=True)
        return
    for fpath in files:
        for read_id, ids, pos in _iter_read_minimizers_file(Path(fpath)):
            lmers = (
                [lookup.get(h, "?") for h in ids]
                if lookup is not None
                else []
            )
            yield ReadRecord(
                read_id=read_id,
                minimizer_ids=ids,
                positions=pos,
                lmers=lmers,
            )


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _compute_minimizer_stats(
    counts: list[int],
) -> dict[str, object]:
    """Compute summary statistics over per-read minimizer counts.

    Args:
        counts: Number of minimizers for each sampled read.

    Returns:
        Dict with keys ``reads``, ``total_minimizers``, ``min``,
        ``max``, ``mean``, ``median``, ``std_dev``, ``n50``.
    """
    n = len(counts)
    total = sum(counts)
    mean = total / n
    variance = sum((x - mean) ** 2 for x in counts) / n
    sorted_counts = sorted(counts)
    median: float = (
        sorted_counts[n // 2]
        if n % 2 == 1
        else (sorted_counts[n // 2 - 1] + sorted_counts[n // 2]) / 2
    )
    cumulative = 0
    n50 = sorted_counts[-1]
    for v in reversed(sorted_counts):
        cumulative += v
        if cumulative >= total / 2:
            n50 = v
            break
    return {
        "reads": n,
        "total_minimizers": total,
        "min": sorted_counts[0],
        "max": sorted_counts[-1],
        "mean": mean,
        "median": median,
        "std_dev": math.sqrt(variance),
        "n50": n50,
    }


def _print_minimizer_stats(stats: dict[str, object], label: str) -> None:
    """Print minimizer-count summary statistics to stdout.

    Args:
        stats: Output of :func:`_compute_minimizer_stats`.
        label: Description line printed before the stats block.
    """
    print(f"\n{label}")
    print(f"Reads:             {stats['reads']:,}")
    print(f"Total minimizers:  {stats['total_minimizers']:,}")
    print(f"Min:               {stats['min']:,}")
    print(f"Max:               {stats['max']:,}")
    print(f"Mean:              {stats['mean']:,.1f}")
    print(f"Median:            {stats['median']:,.1f}")
    print(f"Std dev:           {stats['std_dev']:,.1f}")
    print(f"N50:               {stats['n50']:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    prefix: Annotated[Path, typer.Argument(
        help="rust-mdbg output prefix (e.g. rust_mdbg_out)",
    )],
    summary: Annotated[bool, typer.Option(
        "--summary",
        help="Report summary statistics on the number of minimizers "
             "per read instead of printing individual records",
    )] = False,
    decode: Annotated[bool, typer.Option(
        "--decode",
        help="Decode minimizer IDs to l-mer strings using the "
             "accompanying minimizer_table file",
    )] = False,
    json_out: Annotated[Path | None, typer.Option(
        "--json",
        help="In normal mode: write records as newline-delimited JSON. "
             "In --summary mode: write the stats dict as JSON",
        dir_okay=False,
    )] = None,
    n: Annotated[int, typer.Option(
        "-n", "--num-reads",
        help="Stop after this many reads "
             "(default 1000 in --summary mode, 0 = all otherwise)",
    )] = -1,
    convert: Annotated[bool, typer.Option(
        "--convert",
        help="Convert all {prefix}.*.read_minimizers files from "
             "legacy LZ4-TSV format to packed binary format in-place, "
             "then exit.  Already-binary files are skipped.",
    )] = False,
) -> None:
    """Read and decode rust-mdbg per-read minimizer files."""
    if convert:
        _do_convert(prefix)
        return

    # resolve default for -n
    max_reads: int = (1_000 if n == -1 and summary else n)

    lookup: dict[int, str] | None = None
    if decode:
        table_path = Path(str(prefix) + ".minimizer_table")
        if not table_path.exists():
            typer.echo(
                f"minimizer_table not found: {table_path}", err=True,
            )
            raise typer.Exit(1)
        lookup = load_minimizer_table(table_path)
        typer.echo(
            f"Loaded {len(lookup):,} minimizers from {table_path}",
            err=True,
        )

    counts: list[int] = []
    out_fh = open(json_out, "w") if (json_out and not summary) else None
    total_reads = 0
    total_minimizers = 0

    try:
        for record in iter_records(prefix, lookup):
            total_reads += 1
            n_min = len(record.minimizer_ids)
            total_minimizers += n_min

            if summary:
                counts.append(n_min)
            elif out_fh is not None:
                obj: dict[str, object] = {
                    "read_id": record.read_id,
                    "minimizer_ids": record.minimizer_ids,
                    "positions": record.positions,
                }
                if decode and record.lmers:
                    obj["lmers"] = record.lmers
                out_fh.write(json.dumps(obj) + "\n")
            else:
                if decode and record.lmers:
                    seq_preview = " ".join(record.lmers[:6])
                    if len(record.lmers) > 6:
                        seq_preview += " ..."
                else:
                    seq_preview = ",".join(
                        str(x) for x in record.minimizer_ids[:6]
                    )
                    if len(record.minimizer_ids) > 6:
                        seq_preview += " ..."
                print(f"{record.read_id}\t{n_min} minimizers\t{seq_preview}")

            if max_reads and total_reads >= max_reads:
                break
    finally:
        if out_fh is not None:
            out_fh.close()

    if summary:
        if not counts:
            typer.echo("No reads found.", err=True)
            return
        label_n = (
            "all" if not max_reads else f"first {max_reads:,}"
        )
        label = f"{prefix.name} ({label_n} reads) — minimizers per read:"
        stats = _compute_minimizer_stats(counts)
        _print_minimizer_stats(stats, label)
        if json_out:
            stats["prefix"] = str(prefix)
            stats["reads_sampled"] = max_reads or None
            json_out.write_text(json.dumps(stats, indent=2) + "\n")
            typer.echo(f"Summary written to {json_out}", err=True)
    else:
        typer.echo(
            f"\nTotal reads: {total_reads:,}  "
            f"Total minimizers: {total_minimizers:,}",
            err=True,
        )
        if json_out:
            typer.echo(f"Records written to {json_out}", err=True)


def _do_convert(prefix: Path) -> None:
    """Convert legacy LZ4-TSV read_minimizers files to binary in-place.

    Args:
        prefix: rust-mdbg output prefix.
    """
    pattern = str(prefix) + ".*.read_minimizers"
    files = sorted(glob.glob(pattern))
    if not files:
        typer.echo(
            f"No .read_minimizers files found for prefix: {prefix}",
            err=True,
        )
        raise typer.Exit(1)
    for fpath in files:
        p = Path(fpath)
        raw = lz4.frame.decompress(p.read_bytes())
        if raw[:4] == _RM_MAGIC:
            typer.echo(f"Already binary, skipping: {fpath}", err=True)
            continue
        records = list(_iter_tsv_file(raw))
        write_binary_read_minimizers(p, iter(records))
        typer.echo(f"Converted: {fpath} ({len(records):,} reads)", err=True)


if __name__ == "__main__":
    app()
