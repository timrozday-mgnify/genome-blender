#!/usr/bin/env python3
"""Plot PE combo count distributions for each distance bin from debug output.

Reads the JSONL debug file written by ``parse_gfa.py --debug-path-bins`` and
the PE combo LMDB written by ``--pe-combo-lmdb-out``.  For each distance bin,
looks up the PE sketch count for every path hash and plots a histogram of the
count distribution.

Usage::

    python scripts/plot_path_bin_pe_counts.py \\
        --debug-bins output/rust_mdbg_out.path_bins.jsonl \\
        --pe-lmdb  output/rust_mdbg_out.pe_index.lmdb \\
        --output   output/path_bin_pe_counts.pdf
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Annotated

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import typer

matplotlib.use("Agg")

app = typer.Typer(add_completion=False)


def _load_debug_bins(
    path: Path,
) -> tuple[dict, list[dict]]:
    """Load metadata and per-bin records from a JSONL debug file.

    Args:
        path: Path to the ``.jsonl`` file written by
            ``parse_gfa.py --debug-path-bins``.

    Returns:
        A ``(meta, bins)`` tuple where *meta* is the metadata dict and
        *bins* is a list of bin dicts (one per distance bin).

    Raises:
        ValueError: If the first line is not a ``{"type":"meta"}`` record.
    """
    meta: dict = {}
    bins: list[dict] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if lineno == 1:
                if record.get("type") != "meta":
                    raise ValueError(
                        f"Expected meta record on line 1, got: {record.get('type')!r}"
                    )
                meta = record
            elif record.get("type") == "bin":
                bins.append(record)
    return meta, bins


def _read_pe_lmdb(lmdb_path: Path) -> dict[int, int]:
    """Load all entries from a PE combo LMDB into a dict.

    Keys are uint64 combo hashes; values are uint32 occurrence counts.

    Args:
        lmdb_path: Path to the LMDB directory (``*.pe_index.lmdb``).

    Returns:
        Mapping of hash → count for every entry in the ``"combo"`` database.

    Raises:
        ImportError: If the ``lmdb`` package is not installed.
    """
    try:
        import lmdb
    except ImportError as exc:
        raise ImportError(
            "lmdb package required; install with: pip install lmdb"
        ) from exc

    counts: dict[int, int] = {}
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, max_dbs=1)
    db = env.open_db(b"combo")
    with env.begin(db=db) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            h = struct.unpack("<Q", key)[0]
            c = struct.unpack("<I", value)[0]
            counts[h] = c
    env.close()
    return counts


def _decode_hashes(hex_str: str) -> np.ndarray:
    """Decode hex-packed uint64 hashes from a bin record.

    Args:
        hex_str: Hex string produced by ``np.ndarray.tobytes().hex()``.

    Returns:
        1-D ``np.ndarray`` of dtype ``uint64``.
    """
    raw = bytes.fromhex(hex_str)
    return np.frombuffer(raw, dtype=np.uint64)


@app.command()
def main(
    debug_bins: Annotated[Path, typer.Option(
        "--debug-bins",
        help="JSONL debug file from parse_gfa.py --debug-path-bins",
        exists=True,
        file_okay=True,
        dir_okay=False,
    )],
    pe_lmdb: Annotated[Path, typer.Option(
        "--pe-lmdb",
        help="PE combo LMDB directory from parse_gfa.py --pe-combo-lmdb-out",
        exists=True,
        file_okay=False,
        dir_okay=True,
    )],
    output: Annotated[Path, typer.Option(
        "--output",
        help="Output figure path (.pdf or .png)",
    )] = Path("path_bin_pe_counts.pdf"),
    bins: Annotated[int, typer.Option(
        "--bins",
        help="Number of histogram bins",
        min=1,
    )] = 30,
    log_scale: Annotated[bool, typer.Option(
        "--log-scale / --no-log-scale",
        help="Use log scale on the x-axis (count axis)",
    )] = True,
) -> None:
    """Plot PE combo count histograms for each distance bin.

    For each distance bin in the debug file, looks up the PE sketch count for
    every path hash and plots a histogram of the count distribution.  Bins with
    no path hashes are shown as empty subplots.

    A zero count means the hash was not observed in any read pair; non-zero
    counts reflect how many times that combination minimizer pattern was seen
    across all paired-end reads.
    """
    typer.echo(f"Loading debug bins from {debug_bins} …", err=True)
    meta, bin_records = _load_debug_bins(debug_bins)

    n_bins = meta.get("n_bins", len(bin_records))
    combo_k = meta.get("combo_k", "?")
    combo_density = meta.get("combo_density", "?")
    n_paths = meta.get("n_paths", "?")
    bin_distances = meta.get("bin_distances", [])

    typer.echo(
        f"  combo_k={combo_k}  density={combo_density}  "
        f"n_paths={n_paths}  n_bins={n_bins}",
        err=True,
    )

    typer.echo(f"Loading PE LMDB from {pe_lmdb} …", err=True)
    pe_counts = _read_pe_lmdb(pe_lmdb)
    typer.echo(f"  {len(pe_counts):,} PE hashes loaded", err=True)

    if not bin_records:
        typer.echo("No bin records found in debug file; nothing to plot.", err=True)
        raise typer.Exit(1)

    n_cols = min(4, len(bin_records))
    n_rows = (len(bin_records) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    for ax_idx, (record, ax) in enumerate(
        zip(bin_records, axes.flat)
    ):
        lo = record["lo"]
        hi = record["hi"]
        n_total = record["n_hashes_total"]
        n_sampled = record["n_hashes_sampled"]
        hashes = _decode_hashes(record["hashes"])

        # Look up PE counts for each hash (0 if absent).
        counts_arr = np.array(
            [pe_counts.get(int(h), 0) for h in hashes], dtype=np.int64
        )

        n_present = int((counts_arr > 0).sum())
        containment = n_present / len(hashes) if hashes.size else float("nan")

        if hashes.size == 0:
            ax.text(
                0.5, 0.5, "no hashes",
                ha="center", va="center", transform=ax.transAxes,
            )
        else:
            plot_data = counts_arr[counts_arr > 0] if log_scale else counts_arr
            if log_scale and plot_data.size > 0:
                ax.hist(
                    np.log10(plot_data.astype(np.float64)),
                    bins=bins,
                    color="steelblue",
                    edgecolor="white",
                    linewidth=0.3,
                )
                ax.set_xlabel("log₁₀(PE count)")
            else:
                ax.hist(
                    counts_arr,
                    bins=bins,
                    color="steelblue",
                    edgecolor="white",
                    linewidth=0.3,
                )
                ax.set_xlabel("PE count")

        hi_str = f"{hi:.0f}" if hi != float("inf") else "∞"
        ax.set_title(
            f"bin {ax_idx}: [{lo:.0f}, {hi_str}) bp\n"
            f"n={n_total:,}  sampled={n_sampled:,}  "
            f"present={n_present:,}  containment={containment:.3f}",
            fontsize=8,
        )
        ax.set_ylabel("count")

    # Hide unused subplots.
    for ax in axes.flat[len(bin_records):]:
        ax.set_visible(False)

    dist_str = (
        " → ".join(f"{d:.0f}" for d in bin_distances)
        if bin_distances else "?"
    )
    fig.suptitle(
        f"PE combo counts per path distance bin\n"
        f"combo_k={combo_k}  density={combo_density}  "
        f"n_paths={n_paths}\n"
        f"bin edges: {dist_str} bp",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    typer.echo(f"Saved figure to {output}", err=True)


if __name__ == "__main__":
    app()
