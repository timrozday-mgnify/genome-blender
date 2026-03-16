"""FASTQ and BAM I/O for the read-generation pipeline."""

from __future__ import annotations

import gzip
import logging
from pathlib import Path

import pysam

from genome_blender._progress import progress_task
from genome_blender._utils import reverse_complement
from genome_blender.models import Fragment, Read, ReadBatch

logger = logging.getLogger(__name__)


def write_fastq(
    reads: list[Read],
    output_path: Path,
    append: bool = False,
) -> None:
    """Write reads to a FASTQ file with Phred+33 encoding.

    Gzip compression is applied automatically when *output_path* ends
    with ``.gz``; each call opens (or appends to) an independent gzip
    frame, producing a valid concatenated gzip stream.

    Args:
        reads: Reads to write.
        output_path: Output FASTQ file path.  Use a ``.gz`` suffix for
            gzip-compressed output.
        append: If True, append to the existing file.
    """
    compressed = output_path.suffix == ".gz"
    text_mode = "at" if append else "wt"
    plain_mode = "a" if append else "w"
    logger.debug(
        "%s %d reads to %s",
        "Appending" if append else "Writing",
        len(reads), output_path,
    )
    opener = gzip.open if compressed else open
    open_mode = text_mode if compressed else plain_mode
    with opener(output_path, open_mode) as fh:
        with progress_task(
            len(reads), f"Writing {output_path.name}",
        ) as step:
            for read in reads:
                fh.write(f"@{read.name}\n")
                fh.write(f"{read.sequence}\n")
                fh.write("+\n")
                fh.write(f"{read.quality}\n")
                step()

    logger.info(
        "%s %d reads to %s",
        "Appended" if append else "Wrote",
        len(reads), output_path,
    )


# ------------------------------------------------------------------
# BAM output
# ------------------------------------------------------------------

def build_bam_header(
    genomes: dict[str, list],
) -> tuple[pysam.AlignmentHeader, dict[str, int]]:
    """Build a BAM header and reference name index from genomes.

    Args:
        genomes: Mapping of genome_id to list of
            Bio.SeqRecord.

    Returns:
        Tuple of (AlignmentHeader, dict mapping ref_name
        to ref_id).
    """
    ref_names: list[str] = []
    ref_lengths: list[int] = []
    ref_name_to_idx: dict[str, int] = {}

    for genome_id, records in genomes.items():
        for record in records:
            name = f"{genome_id}:{record.id}"
            ref_name_to_idx[name] = len(ref_names)
            ref_names.append(name)
            ref_lengths.append(len(record.seq))

    header = pysam.AlignmentHeader.from_dict({
        "HD": {"VN": "1.6", "SO": "unsorted"},
        "SQ": [
            {"SN": name, "LN": length}
            for name, length in zip(ref_names, ref_lengths)
        ],
    })
    logger.debug(
        "BAM header: %d reference sequences", len(ref_names),
    )
    return header, ref_name_to_idx


def _ref_consumed(cigar: list[tuple[int, int]]) -> int:
    """Return the number of reference bases consumed by a CIGAR."""
    return sum(
        length for op, length in cigar if op in (0, 2)
    )


def _bam_fields_for_read(
    read: Read,
    frag: Fragment,
    is_reverse: bool,
) -> tuple[str, str, list[tuple[int, int]], int]:
    """Compute BAM-ready sequence, quality, CIGAR, and ref_start.

    SAM stores query_sequence on the forward strand.  When the
    read maps to the reverse strand, sequence and quality must
    be reversed and complemented, and the CIGAR must be reversed.

    Returns:
        Tuple of (query_sequence, quality, cigar,
        reference_start).
    """
    cigar = (
        read.cigar if read.cigar is not None
        else [(0, len(read.sequence))]
    )
    seq = read.sequence
    qual = read.quality
    ref_start = frag.start
    if is_reverse:
        seq = reverse_complement(seq)
        qual = qual[::-1]
        cigar = list(reversed(cigar))
        ref_start = frag.end - _ref_consumed(cigar)
    return seq, qual, cigar, ref_start


def write_bam_chunk(
    bam: pysam.AlignmentFile,
    header: pysam.AlignmentHeader,
    ref_name_to_idx: dict[str, int],
    fragments: list[Fragment],
    read_batch: ReadBatch,
) -> None:
    """Write a chunk of ground-truth alignments to an open BAM.

    Args:
        bam: Open BAM file for writing.
        header: BAM header.
        ref_name_to_idx: Reference name to index mapping.
        fragments: Source fragments.
        read_batch: Generated reads.
    """
    def write_pe(
        i: int, frag: Fragment, ref_id: int,
    ) -> None:
        assert read_batch.paired is not None
        r1, r2 = read_batch.paired[i]
        qname = r1.name.split("/")[0]
        tlen = frag.end - frag.start

        r1_is_reverse = frag.strand == "-"
        r2_is_reverse = not r1_is_reverse

        r1_seq, r1_qual, r1_cigar, r1_start = (
            _bam_fields_for_read(r1, frag, r1_is_reverse)
        )
        r2_seq, r2_qual, r2_cigar, r2_start = (
            _bam_fields_for_read(r2, frag, r2_is_reverse)
        )

        if r1_start <= r2_start:
            r1_tlen, r2_tlen = tlen, -tlen
        else:
            r1_tlen, r2_tlen = -tlen, tlen

        a1 = pysam.AlignedSegment(header)
        a1.query_name = qname
        a1.query_sequence = r1_seq
        a1.flag = 0
        a1.is_paired = True
        a1.is_proper_pair = True
        a1.is_read1 = True
        a1.is_reverse = r1_is_reverse
        a1.mate_is_reverse = r2_is_reverse
        a1.reference_id = ref_id
        a1.reference_start = r1_start
        a1.cigar = r1_cigar
        a1.mapping_quality = 255
        a1.query_qualities = (
            pysam.qualitystring_to_array(r1_qual)
        )
        a1.next_reference_id = ref_id
        a1.next_reference_start = r2_start
        a1.template_length = r1_tlen

        a2 = pysam.AlignedSegment(header)
        a2.query_name = qname
        a2.query_sequence = r2_seq
        a2.flag = 0
        a2.is_paired = True
        a2.is_proper_pair = True
        a2.is_read2 = True
        a2.is_reverse = r2_is_reverse
        a2.mate_is_reverse = r1_is_reverse
        a2.reference_id = ref_id
        a2.reference_start = r2_start
        a2.cigar = r2_cigar
        a2.mapping_quality = 255
        a2.query_qualities = (
            pysam.qualitystring_to_array(r2_qual)
        )
        a2.next_reference_id = ref_id
        a2.next_reference_start = r1_start
        a2.template_length = r2_tlen

        bam.write(a1)
        bam.write(a2)

    def write_se(
        i: int, frag: Fragment, ref_id: int,
    ) -> None:
        assert read_batch.single is not None
        read = read_batch.single[i]
        is_reverse = frag.strand == "-"
        seq, qual, cigar, ref_start = (
            _bam_fields_for_read(read, frag, is_reverse)
        )
        a = pysam.AlignedSegment(header)
        a.query_name = read.name
        a.query_sequence = seq
        a.flag = 0
        a.is_reverse = is_reverse
        a.reference_id = ref_id
        a.reference_start = ref_start
        a.cigar = cigar
        a.mapping_quality = 255
        a.query_qualities = (
            pysam.qualitystring_to_array(qual)
        )
        bam.write(a)

    write_alignment = (
        write_pe if read_batch.is_paired else write_se
    )

    with progress_task(
        len(fragments), "Writing BAM",
    ) as step:
        for i, frag in enumerate(fragments):
            ref_name = (
                f"{frag.genome_id}:{frag.contig_id}"
            )
            ref_id = ref_name_to_idx[ref_name]
            write_alignment(i, frag, ref_id)
            step()


def write_bam(
    fragments: list[Fragment],
    read_batch: ReadBatch,
    genomes: dict[str, list],
    output_path: Path,
) -> None:
    """Write ground-truth alignments to a BAM file.

    Convenience wrapper that builds the header and writes all
    fragments in a single pass.

    Args:
        fragments: Source fragments.
        read_batch: Generated reads.
        genomes: Mapping of genome_id to list of SeqRecord.
        output_path: Output BAM file path.
    """
    header, ref_name_to_idx = build_bam_header(genomes)
    with pysam.AlignmentFile(
        output_path, "wb", header=header,
    ) as bam:
        write_bam_chunk(
            bam, header, ref_name_to_idx,
            fragments, read_batch,
        )
    logger.info("Wrote ground-truth BAM to %s", output_path)
