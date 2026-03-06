#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/rust-mdbg"
mkdir -p "${OUTPUT_DIR}"

# rust-mdbg accepts a single reads file; concatenate all FASTQ inputs
COMBINED="${OUTPUT_DIR}/combined_reads.fastq"
cat "${INPUT_DIR}"/*.fastq > "${COMBINED}"

docker run --rm \
    -v "${OUTPUT_DIR}:/output" \
    --entrypoint rust-mdbg \
    rust-mdbg \
    /output/combined_reads.fastq \
    -k 4 \
    --density 0.1 \
    -l 8 \
    --minabund 2 \
    --prefix /output/rust_mdbg_out

docker run --rm \
    -v "${OUTPUT_DIR}:/output" \
    --entrypoint to_basespace \
    rust-mdbg \
    --gfa /output/rust_mdbg_out.unitigs.gfa \
    --sequences /output/rust_mdbg_out
