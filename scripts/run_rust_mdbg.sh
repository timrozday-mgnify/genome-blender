#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/rust-mdbg"
mkdir -p "${OUTPUT_DIR}"

FASTQ_FILES=("${INPUT_DIR}"/*.fastq)

docker run --rm \
    -v "${INPUT_DIR}:/input:ro" \
    -v "${OUTPUT_DIR}:/output" \
    --entrypoint rust-mdbg \
    rust-mdbg \
    "${FASTQ_FILES[@]/#${INPUT_DIR}//input}" \
    -k 7 \
    --density 0.0008 \
    -l 10 \
    --minabund 2 \
    --prefix /output/rust_mdbg_out

docker run --rm \
    -v "${OUTPUT_DIR}:/output" \
    --entrypoint to_basespace \
    rust-mdbg \
    --gfa /output/rust_mdbg_out.unitigs.gfa \
    --sequences /output/rust_mdbg_out
