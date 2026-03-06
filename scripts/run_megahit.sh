#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/megahit"
# megahit requires the output dir not to exist
rm -rf "${OUTPUT_DIR}"

# Assembly parameters
K_LIST="21,31,41,51"    # comma-separated odd k-mer sizes (range 15-255)
BUBBLE_LEVEL=0           # bubble merging intensity (0-2); 0 = disabled
PRUNE_LEVEL=0            # low-depth pruning strength (0-3); 0 = disabled
MAX_TIP_LEN=0            # remove tips shorter than this; 0 = keep all
CLEANING_ROUNDS=0        # graph cleaning iterations; 0 = skip cleaning
DISCONNECT_RATIO=0       # disable unitig removal by depth ratio
LOW_LOCAL_RATIO=0        # disable neighbourhood depth filtering
MIN_COUNT=1              # minimum k-mer multiplicity; 1 = keep singletons

# Assemble
docker run --rm \
    -v "${INPUT_DIR}:/input:ro" \
    -v "${INPUT_DIR}:/output" \
    --entrypoint megahit \
    quay.io/biocontainers/megahit:1.2.9--haf24da9_8 \
    -1 /input/sim_reads_R1.fastq \
    -2 /input/sim_reads_R2.fastq \
    -o /output/megahit \
    --k-list "${K_LIST}" \
    --bubble-level "${BUBBLE_LEVEL}" \
    --prune-level "${PRUNE_LEVEL}" \
    --max-tip-len "${MAX_TIP_LEN}" \
    --cleaning-rounds "${CLEANING_ROUNDS}" \
    --disconnect-ratio "${DISCONNECT_RATIO}" \
    --low-local-ratio "${LOW_LOCAL_RATIO}" \
    --min-count "${MIN_COUNT}" \
    --no-mercy \
    --no-local \
    --keep-tmp-files \
    -t 4

# Find the largest k used in intermediate contigs and convert to FASTG
docker run --rm \
    -v "${OUTPUT_DIR}:/output" \
    --entrypoint bash \
    quay.io/biocontainers/megahit:1.2.9--haf24da9_8 \
    -c '
        k=$(ls /output/intermediate_contigs/k*.contigs.fa \
            | sed "s/.*k\([0-9]*\).*/\1/" | sort -n | tail -1)
        megahit_core contig2fastg "${k}" \
            "/output/intermediate_contigs/k${k}.contigs.fa" \
            > "/output/assembly_graph.k${k}.fastg"
        echo "Wrote assembly_graph.k${k}.fastg"
    '

# Convert FASTG to GFA
FASTG=$(ls "${OUTPUT_DIR}"/assembly_graph.k*.fastg)
K_FASTG=$(basename "${FASTG}" .fastg)
docker run --rm \
    -v "${OUTPUT_DIR}:/output" \
    quay.io/biocontainers/gfatools:0.5.5--h577a1d6_0 \
    gfatools view -S "/output/${K_FASTG}.fastg" \
    > "${OUTPUT_DIR}/${K_FASTG}.gfa"
echo "Wrote ${K_FASTG}.gfa"
