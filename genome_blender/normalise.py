"""Normalise FASTA headers to unique, whitespace-free reference names.

Downstream BAM headers use one ``@SQ`` line per contig keyed on the
name before the first space; duplicate or whitespace-bearing names
corrupt the header. This strips each header to its first token and
disambiguates collisions with a ``_2``, ``_3``, ... suffix, writing a
TSV mapping the original full header to the new name.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from Bio import SeqIO

logger = logging.getLogger(__name__)

app = typer.Typer()


def normalise_fasta(
    input_fasta: Path,
    output_fasta: Path,
    mapping_path: Path,
) -> int:
    """Rewrite FASTA headers to unique first-token names.

    Args:
        input_fasta: Source FASTA.
        output_fasta: Destination FASTA with normalised headers.
        mapping_path: TSV written as ``old_header<TAB>new_name``,
            where ``old_header`` is the full original header line
            (text after the space included).

    Returns:
        Number of records written.
    """
    seen: dict[str, int] = {}
    count = 0

    with (
        open(mapping_path, "w") as mapping,
        open(output_fasta, "w") as out,
    ):
        mapping.write("old_header\tnew_name\n")
        for record in SeqIO.parse(input_fasta, "fasta"):
            base = record.id  # first whitespace-delimited token
            n = seen.get(base, 0) + 1
            seen[base] = n
            new_name = base if n == 1 else f"{base}_{n}"

            mapping.write(f"{record.description}\t{new_name}\n")
            record.id = new_name
            record.name = new_name
            record.description = ""
            SeqIO.write(record, out, "fasta")
            count += 1

    dupes = sum(1 for v in seen.values() if v > 1)
    logger.info(
        "Normalised %d records (%d name(s) disambiguated) "
        "from %s -> %s (mapping: %s)",
        count, dupes, input_fasta, output_fasta, mapping_path,
    )
    return count


@app.command()
def main(
    input_fasta: Annotated[Path, typer.Argument(
        help="Input FASTA file",
    )],
    output_fasta: Annotated[Path, typer.Option(
        "-o", "--output",
        help="Output FASTA with normalised headers",
    )],
    mapping: Annotated[Path, typer.Option(
        "-m", "--mapping",
        help="Output TSV mapping old header -> new name",
    )],
) -> None:
    """Normalise FASTA headers so reference names are unique."""
    logging.basicConfig(level=logging.INFO)
    normalise_fasta(input_fasta, output_fasta, mapping)


if __name__ == "__main__":
    app()
