#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/megahit"
# megahit requires the output dir not to exist
rm -rf "${OUTPUT_DIR}"

# Assemble
docker run --rm \
    -v "${INPUT_DIR}:/input:ro" \
    -v "${INPUT_DIR}:/output" \
    --entrypoint megahit \
    quay.io/biocontainers/megahit:1.2.9--haf24da9_8 \
    -1 /input/sim_reads_R1.fastq \
    -2 /input/sim_reads_R2.fastq \
    -o /output/megahit \
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
