#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/bifrost"
mkdir -p "${OUTPUT_DIR}"

FASTQ_FILES=("${INPUT_DIR}"/*.fastq)
SEQ_ARGS=""
for f in "${FASTQ_FILES[@]}"; do
    SEQ_ARGS="${SEQ_ARGS} -s /input/$(basename "${f}")"
done

docker run --rm \
    -v "${INPUT_DIR}:/input:ro" \
    -v "${OUTPUT_DIR}:/output" \
    bifrost build \
    ${SEQ_ARGS} \
    -k 31 \
    -o /output/bifrost_out \
    -i -d \
    -t 4
