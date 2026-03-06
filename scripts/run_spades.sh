#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/spades"
mkdir -p "${OUTPUT_DIR}"

docker run --rm \
    -v "${INPUT_DIR}:/input:ro" \
    -v "${OUTPUT_DIR}:/output" \
    --entrypoint spades.py \
    quay.io/biocontainers/spades:4.2.0--h8d6e82b_2 \
    --meta \
    --phred-offset 33 \
    -1 /input/sim_reads_R1.fastq \
    -2 /input/sim_reads_R2.fastq \
    -o /output \
    -t 4
