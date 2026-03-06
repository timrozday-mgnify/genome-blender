#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/metaMDBG"
mkdir -p "${OUTPUT_DIR}"

FASTQ_FILES=("${INPUT_DIR}"/*.fastq)
INPUT_ARGS=""
for f in "${FASTQ_FILES[@]}"; do
    INPUT_ARGS="${INPUT_ARGS} /input/$(basename "${f}")"
done

docker run --rm \
    -v "${INPUT_DIR}:/input:ro" \
    -v "${OUTPUT_DIR}:/output" \
    meta-mdbg asm \
    --out-dir /output \
    --in-ont ${INPUT_ARGS} \
    --threads 4

# Extract assembly graph at the lowest available k
docker run --rm \
    -v "${OUTPUT_DIR}:/output" \
    meta-mdbg gfa \
    --assembly-dir /output \
    --k 0 \
    --contigpath \
    --readpath \
    --threads 4
