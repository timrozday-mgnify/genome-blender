"""Genome loading from CSV input tables."""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from Bio import SeqIO

from genome_blender._progress import progress_task
from genome_blender._utils import gc_fraction

logger = logging.getLogger(__name__)


def load_genomes(
    csv_path: Path,
) -> tuple[dict[str, list], dict[str, float]]:
    """Parse input CSV and load FASTA files.

    Args:
        csv_path: CSV with columns ``genome_id``,
            ``fasta_path``, ``abundance``.

    Returns:
        Tuple of (genomes dict mapping genome_id to list of
        Bio.SeqRecord, abundances dict mapping genome_id to
        normalised abundance).

    Raises:
        ValueError: If total abundance is zero.
    """
    genomes: dict[str, list] = {}
    raw_abundances: dict[str, float] = {}

    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    logger.debug("Found %d entries in %s", len(rows), csv_path)

    with progress_task(len(rows), "Loading genomes") as step:
        for row in rows:
            genome_id = row["genome_id"]
            fasta_path = Path(row["fasta_path"])
            abundance = float(row["abundance"])

            logger.debug(
                "Parsing FASTA %s for genome %s",
                fasta_path, genome_id,
            )
            records = list(SeqIO.parse(fasta_path, "fasta"))
            if not records:
                logger.warning(
                    "No sequences found in %s", fasta_path,
                )
                step()
                continue

            genomes[genome_id] = records
            raw_abundances[genome_id] = abundance
            logger.info(
                "Loaded %s: %d contigs from %s",
                genome_id, len(records), fasta_path,
            )
            for rec in records:
                logger.debug(
                    "  contig %s: %d bp, GC=%.1f%%",
                    rec.id, len(rec.seq),
                    gc_fraction(str(rec.seq)) * 100,
                )
            step()

    total = sum(raw_abundances.values())
    if total == 0:
        raise ValueError(
            "Total abundance is zero; check input CSV"
        )
    abundances = {
        gid: a / total for gid, a in raw_abundances.items()
    }

    return genomes, abundances
