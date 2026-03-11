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
    -k 7 \
    --density 0.08 \
    -l 12 \
    --minabund 2 \
    --prefix /output/rust_mdbg_out \
    --debug \
2>&1 | tee ${OUTPUT_DIR}/output.txt


/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python $(dirname $0)/parse_gfa.py --paired-end -n 10000 ${OUTPUT_DIR}/rust_mdbg_out.gfa

# echo "rust_mdbg_out.gfa head: $( cat ${OUTPUT_DIR}/rust_mdbg_out.gfa | sed -nE 's/^.*LN:i:([0-9]+).*$/\1/p' | sort -nr | head -n 5 )"

# docker run --rm \
#     -v "${OUTPUT_DIR}:/output" \
#     --entrypoint to_basespace \
#     rust-mdbg \
#     --gfa /output/rust_mdbg_out.unitigs.gfa \
#     --sequences /output/rust_mdbg_out
# 
# echo "rust_mdbg_out.unitigs.gfa.complete.gfa wc stats: $(wc ${OUTPUT_DIR}/rust_mdbg_out.unitigs.gfa.complete.gfa)"
