"""DNA fragment generation from reference genomes.

Sample fragments by shearing (with optional GC bias) or treat
input sequences as amplicons.
"""

from __future__ import annotations

import logging
import math

import torch
from pyro.distributions import NegativeBinomial, Poisson

from genome_blender._progress import progress_task
from genome_blender._utils import (
    gc_fraction,
    nb_params_from_mean_variance,
    reverse_complement,
)
from genome_blender.models import Fragment

logger = logging.getLogger(__name__)


def sample_fragments(
    genomes: dict[str, list],
    abundances: dict[str, float],
    num_fragments: int,
    fragment_mean: float,
    fragment_variance: float,
    gc_bias_strength: float,
    rng: torch.Generator,
) -> list[Fragment]:
    """Sample DNA fragments from genomes with optional GC bias.

    Args:
        genomes: Mapping of genome_id to list of SeqRecord.
        abundances: Mapping of genome_id to normalised abundance.
        num_fragments: Number of fragments to generate.
        fragment_mean: Mean fragment length.
        fragment_variance: Variance of fragment length
            distribution.
        gc_bias_strength: Strength of GC bias filtering.
            0 = no bias.
        rng: Seeded random number generator.

    Returns:
        List of sampled ``Fragment`` objects.
    """
    # Allocate fragments per genome proportional to abundance
    genome_ids = list(abundances.keys())
    abundance_values = torch.tensor(
        [abundances[gid] for gid in genome_ids],
        dtype=torch.float64,
    )
    counts_float = abundance_values * num_fragments
    counts = counts_float.floor().long()
    remainder = num_fragments - counts.sum().item()
    fractional = counts_float - counts.float()
    if remainder > 0:
        _, top_idx = fractional.topk(int(remainder))
        for idx in top_idx:
            counts[idx] += 1

    # Set up fragment length distribution
    frag_dist = _build_fragment_dist(
        fragment_mean, fragment_variance,
    )

    for gid in genome_ids:
        logger.debug(
            "  %s: target %d fragments (abundance=%.4f)",
            gid,
            counts[genome_ids.index(gid)].item(),
            abundances[gid],
        )

    fragments: list[Fragment] = []

    with progress_task(
        num_fragments, "Sampling fragments",
    ) as step:
        for genome_idx, genome_id in enumerate(genome_ids):
            n_frags = counts[genome_idx].item()
            if n_frags == 0:
                continue

            records = genomes[genome_id]
            contig_lengths = torch.tensor(
                [len(r.seq) for r in records],
                dtype=torch.float64,
            )
            contig_weights = (
                contig_lengths / contig_lengths.sum()
            )

            accepted = 0
            rejected = 0
            max_attempts = n_frags * 20
            total_attempts = 0
            # Batch size: oversample to absorb rejections
            _batch = max(min(n_frags * 2, 4096), 64)
            _fixed_len = (
                max(1, int(fragment_mean))
                if frag_dist is None
                else None
            )

            while (
                accepted < n_frags
                and total_attempts < max_attempts
            ):
                batch = min(
                    _batch, max_attempts - total_attempts,
                )
                contig_idxs = torch.multinomial(
                    contig_weights, batch,
                    replacement=True, generator=rng,
                )
                if _fixed_len is not None:
                    frag_lens = torch.full(
                        (batch,), _fixed_len,
                        dtype=torch.long,
                    )
                else:
                    frag_lens = (
                        frag_dist.sample((batch,))
                        .clamp(min=1).long()
                    )
                start_rands = torch.rand(
                    batch, generator=rng,
                )
                strand_rands = torch.rand(
                    batch, generator=rng,
                )
                gc_rands = (
                    torch.rand(batch, generator=rng)
                    if gc_bias_strength > 0
                    else None
                )

                for i in range(batch):
                    total_attempts += 1
                    if total_attempts > max_attempts:
                        break

                    ci = int(contig_idxs[i].item())
                    record = records[ci]
                    contig_len = len(record.seq)
                    frag_len = int(frag_lens[i].item())

                    if frag_len > contig_len:
                        rejected += 1
                        continue

                    max_start = contig_len - frag_len
                    start = min(
                        int(
                            start_rands[i].item()
                            * (max_start + 1)
                        ),
                        max_start,
                    )
                    end = start + frag_len
                    strand = (
                        "+"
                        if strand_rands[i].item() < 0.5
                        else "-"
                    )

                    seq_str = str(record.seq[start:end])
                    if strand == "-":
                        seq_str = reverse_complement(seq_str)

                    if gc_bias_strength > 0:
                        gc = gc_fraction(seq_str)
                        p_keep = math.exp(
                            -gc_bias_strength
                            * (gc - 0.5) ** 2
                        )
                        assert gc_rands is not None
                        if gc_rands[i].item() > p_keep:
                            rejected += 1
                            continue

                    fragments.append(Fragment(
                        genome_id=genome_id,
                        contig_id=record.id,
                        start=start,
                        end=end,
                        strand=strand,
                        sequence=seq_str,
                    ))
                    accepted += 1
                    step()
                    if accepted >= n_frags:
                        break

            logger.debug(
                "  %s: accepted %d/%d fragments "
                "(%d rejected, %d attempts)",
                genome_id, accepted, n_frags,
                rejected, total_attempts,
            )

            if accepted < n_frags:
                logger.warning(
                    "Only generated %d/%d fragments for %s "
                    "(GC bias may be too strong or contigs "
                    "too short)",
                    accepted, n_frags, genome_id,
                )

    return fragments


def amplicon_fragments(
    genomes: dict[str, list],
    abundances: dict[str, float],
    num_fragments: int,
    rng: torch.Generator,
) -> list[Fragment]:
    """Create fragments from input sequences treated as amplicons.

    Each sequence record is used directly as a fragment (no
    shearing).  Fragments are replicated proportionally to genome
    abundance to reach the requested total, modelling PCR
    amplification.

    Args:
        genomes: Mapping of genome_id to list of SeqRecord.
        abundances: Mapping of genome_id to normalised abundance.
        num_fragments: Total number of fragments to produce.
        rng: Seeded random number generator (for shuffling).

    Returns:
        Shuffled list of ``Fragment`` objects.
    """
    amplicons: list[tuple[str, object]] = []
    weights: list[float] = []
    for genome_id, records in genomes.items():
        for record in records:
            amplicons.append((genome_id, record))
            weights.append(abundances[genome_id])

    if not amplicons:
        return []

    weight_tensor = torch.tensor(
        weights, dtype=torch.float64,
    )
    weight_tensor = weight_tensor / weight_tensor.sum()

    # Largest-remainder allocation
    counts_float = weight_tensor * num_fragments
    counts = counts_float.floor().long()
    remainder = num_fragments - counts.sum().item()
    if remainder > 0:
        fractional = counts_float - counts.float()
        _, top_idx = fractional.topk(int(remainder))
        for idx in top_idx:
            counts[idx] += 1

    fragments: list[Fragment] = []
    logger.debug("Amplicon allocation:")
    with progress_task(
        num_fragments, "Building amplicon fragments",
    ) as step:
        for i, (genome_id, record) in enumerate(amplicons):
            n = counts[i].item()
            if n == 0:
                continue
            logger.debug(
                "  %s:%s (%d bp) -> %d copies",
                genome_id, record.id, len(record.seq), n,
            )
            seq_str = str(record.seq)
            for _ in range(n):
                fragments.append(Fragment(
                    genome_id=genome_id,
                    contig_id=record.id,
                    start=0,
                    end=len(record.seq),
                    strand="+",
                    sequence=seq_str,
                ))
                step()

    # Shuffle so reads aren't grouped by amplicon
    n = len(fragments)
    indices = torch.randperm(n, generator=rng).tolist()
    fragments = [fragments[i] for i in indices]

    return fragments


def _build_fragment_dist(
    fragment_mean: float, fragment_variance: float,
) -> NegativeBinomial | Poisson | None:
    """Build the fragment length distribution.

    Returns ``None`` when variance is zero (fixed length).
    """
    if fragment_variance == 0:
        logger.debug(
            "Fragment length distribution: fixed at %d",
            int(fragment_mean),
        )
        return None

    dist_name, dist_params = nb_params_from_mean_variance(
        fragment_mean, fragment_variance,
    )
    if dist_name == "nb":
        logger.debug(
            "Fragment length distribution: NegativeBinomial"
            "(r=%.2f, p=%.4f)",
            dist_params["total_count"],
            dist_params["probs"],
        )
        return NegativeBinomial(
            total_count=dist_params["total_count"],
            probs=dist_params["probs"],
        )

    logger.debug(
        "Fragment length distribution: Poisson(rate=%.2f)",
        dist_params["rate"],
    )
    return Poisson(rate=dist_params["rate"])
