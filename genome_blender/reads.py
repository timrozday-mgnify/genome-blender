"""Read generation from DNA fragments.

Generate single-end, paired-end, or long reads from a list of
fragments.
"""

from __future__ import annotations

import logging

import torch
from pyro.distributions import LogNormal

from genome_blender._progress import progress_task
from genome_blender._utils import (
    lognormal_params_from_mean_variance,
    reverse_complement,
)
from genome_blender.models import Fragment, Read, ReadBatch

logger = logging.getLogger(__name__)


def _generate_long_read(
    frag: Fragment, global_idx: int, read_len: int,
) -> Read:
    """Generate a long read spanning the entire fragment."""
    base_name = (
        f"{frag.genome_id}:{frag.contig_id}:"
        f"{frag.start}-{frag.end}:{frag.strand}"
    )
    seq = frag.sequence
    qual = "I" * len(seq)  # Q40 Phred+33
    return Read(
        name=f"{base_name} read_{global_idx}",
        sequence=seq,
        quality=qual,
    )


def _generate_se_read(
    frag: Fragment, global_idx: int, read_len: int,
) -> Read:
    """Generate a single-end read from a fragment."""
    base_name = (
        f"{frag.genome_id}:{frag.contig_id}:"
        f"{frag.start}-{frag.end}:{frag.strand}"
    )
    seq = frag.sequence[:read_len]
    qual = "I" * len(seq)  # Q40 Phred+33
    return Read(
        name=f"{base_name} read_{global_idx}",
        sequence=seq,
        quality=qual,
    )


def _generate_pe_read(
    frag: Fragment, global_idx: int, read_len: int,
) -> tuple[Read, Read]:
    """Generate a paired-end read pair from a fragment."""
    base_name = (
        f"{frag.genome_id}:{frag.contig_id}:"
        f"{frag.start}-{frag.end}:{frag.strand}"
    )
    frag_len = len(frag.sequence)
    r1_len = min(read_len, frag_len)
    r2_len = min(read_len, frag_len)

    r1_seq = frag.sequence[:r1_len]
    r2_seq = reverse_complement(frag.sequence[-r2_len:])

    return (
        Read(
            name=f"{base_name}/1 read_{global_idx}",
            sequence=r1_seq,
            quality="I" * len(r1_seq),
        ),
        Read(
            name=f"{base_name}/2 read_{global_idx}",
            sequence=r2_seq,
            quality="I" * len(r2_seq),
        ),
    )


def generate_reads(
    fragments: list[Fragment],
    read_length_mean: float,
    read_length_variance: float,
    paired_end: bool,
    rng: torch.Generator,
    read_index_offset: int = 0,
    long_read: bool = False,
) -> ReadBatch:
    """Generate reads from fragments.

    Args:
        fragments: Source fragments.
        read_length_mean: Mean read length.
        read_length_variance: Variance of read length
            distribution.
        paired_end: If True, generate paired-end reads.
        rng: Seeded random number generator.
        read_index_offset: Starting index for read naming.
        long_read: If True, each read spans the entire
            fragment (read length params ignored).

    Returns:
        A ``ReadBatch`` containing the generated reads.
    """
    # rng is accepted for API consistency; torch uses global
    # RNG state seeded in main() via torch.manual_seed()
    _ = rng

    # Select read generator and set up length sampling
    if long_read:
        mode = "long"
        generate_read = _generate_long_read
        batch_key = "single"
        read_len_dist = None
    elif paired_end:
        mode = "paired-end"
        generate_read = _generate_pe_read
        batch_key = "paired"
    else:
        mode = "single-end"
        generate_read = _generate_se_read
        batch_key = "single"

    if not long_read:
        if read_length_variance == 0:
            read_len_dist = None
            logger.debug(
                "Read length distribution: fixed at %d",
                int(read_length_mean),
            )
        else:
            mu_ln, sigma_ln = (
                lognormal_params_from_mean_variance(
                    read_length_mean, read_length_variance,
                )
            )
            read_len_dist = LogNormal(
                loc=mu_ln, scale=sigma_ln,
            )
            logger.debug(
                "Read length distribution: "
                "LogNormal(mu=%.4f, sigma=%.4f)",
                mu_ln, sigma_ln,
            )

    reads: list = []

    with progress_task(
        len(fragments), f"Generating {mode} reads",
    ) as step:
        for i, frag in enumerate(fragments):
            global_idx = read_index_offset + i
            frag_len = len(frag.sequence)
            if long_read:
                read_len = frag_len
            elif read_len_dist is None:
                read_len = min(
                    max(1, int(read_length_mean)), frag_len,
                )
            else:
                sampled_len = int(
                    read_len_dist.sample()
                    .clamp(min=1).item()
                )
                read_len = min(sampled_len, frag_len)
            reads.append(
                generate_read(frag, global_idx, read_len),
            )
            step()

    logger.debug("Generated %d %s reads", len(reads), mode)
    return ReadBatch(**{batch_key: reads})
