#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/rust-mdbg"
mkdir -p "${OUTPUT_DIR}"

# rust-mdbg accepts a single reads file; concatenate all FASTQ inputs
COMBINED="${OUTPUT_DIR}/combined_reads.fastq"
cat "${INPUT_DIR}"/*.fastq > "${COMBINED}"

# Estimate mean read length from the first 1000 reads, then derive density.
# The minimizer length l and density are related by:
#   read_length * density * 0.75 = l  =>  density = l / (read_length * 0.75)
L=12
READS_STATS_JSON="${OUTPUT_DIR}/reads_stats.json"
/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python \
    "$(dirname "$0")/reads_summary.py" "${COMBINED}" -n 1000 \
    --json "${READS_STATS_JSON}"
MEAN_READ_LEN=$(
    /Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python3 -c \
        "import json; d=json.load(open('${READS_STATS_JSON}')); print(int(d['mean'] + 0.5))"
)
DENSITY=$(
    /Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python3 -c \
        "print(f'{${L} / (${MEAN_READ_LEN} * 0.75):.4f}')"
)
echo "Estimated mean read length: ${MEAN_READ_LEN} bp"
echo "Using density: ${DENSITY}  (l=${L} / (${MEAN_READ_LEN} * 0.75))"

docker run --rm \
    -v "${OUTPUT_DIR}:/output" \
    --entrypoint rust-mdbg \
    rust-mdbg \
    /output/combined_reads.fastq \
    -k 7 \
    --density "${DENSITY}" \
    -l ${L} \
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
