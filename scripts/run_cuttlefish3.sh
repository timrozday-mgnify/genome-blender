#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/cuttlefish3"
mkdir -p "${OUTPUT_DIR}" "${OUTPUT_DIR}/tmp"

FASTQ_FILES=("${INPUT_DIR}"/*.fastq)
SEQ_ARGS=""
for f in "${FASTQ_FILES[@]}"; do
    SEQ_ARGS="${SEQ_ARGS} --seq /input/$(basename "${f}")"
done

docker run --rm \
    -v "${INPUT_DIR}:/input:ro" \
    -v "${OUTPUT_DIR}:/output" \
    -v cuttlefish3_tmp:/tmp/cf3 \
    cuttlefish3 build \
    --read \
    ${SEQ_ARGS} \
    -k 31 \
    -o /output/cuttlefish3_out \
    -w /tmp/cf3 \
    -c 1
