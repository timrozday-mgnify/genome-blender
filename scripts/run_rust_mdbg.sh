#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/rust_mdbg_example && pwd)"
OUTPUT_DIR="${INPUT_DIR}/rust-mdbg"
mkdir -p "${OUTPUT_DIR}"

# rust-mdbg accepts a single reads file; concatenate all FASTQ inputs
COMBINED="${OUTPUT_DIR}/combined_reads.fa.gz"
cat "${INPUT_DIR}"/*.fa.gz > "${COMBINED}"

docker run --rm \
    -v "${OUTPUT_DIR}:/output" \
    --entrypoint rust-mdbg \
    rust-mdbg \
    /output/combined_reads.fa.gz \
    -k 7 \
    --density 0.0008 \
    -l 10 \
    --minabund 2 \
    --prefix /output/rust_mdbg_out \
    --debug \
2>&1 | tee ${OUTPUT_DIR}/output.txt

echo "rust_mdbg_out.gfa head: $( cat ${OUTPUT_DIR}/rust_mdbg_out.gfa | sed -nE 's/^.*LN:i:([0-9]+).*$/\1/p' | sort -nr | head -n 5 )"

docker run --rm \
    -v "${OUTPUT_DIR}:/output" \
    --entrypoint to_basespace \
    rust-mdbg \
    --gfa /output/rust_mdbg_out.unitigs.gfa \
    --sequences /output/rust_mdbg_out

echo "rust_mdbg_out.unitigs.gfa.complete.gfa wc stats: $(wc ${OUTPUT_DIR}/rust_mdbg_out.unitigs.gfa.complete.gfa)"
