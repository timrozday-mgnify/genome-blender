#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/rust-mdbg"
mkdir -p "${OUTPUT_DIR}"

# rust-mdbg accepts a single reads file; concatenate all FASTQ inputs
COMBINED="${OUTPUT_DIR}/combined_reads.fastq"
cat "${INPUT_DIR}"/*.fastq > "${COMBINED}"

# rust-mdbg parameters
K=7   # k-mer size (minimizer k-mer)
L=12  # l-mer length (minimizer length)
MINABUND=2

# rust-mdbg output paths
PREFIX="${OUTPUT_DIR}/rust_mdbg_out"
GFA="${PREFIX}.gfa"
MINIMIZER_TABLE="${PREFIX}.minimizer_table"

# Estimate mean read length from the first 1000 reads, then derive density.
# The minimizer length l and density are related by:
#   read_length * density * 0.75 = l  =>  density = l / (read_length * 0.75)
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

# Run rust-mdbg directly (local build from timrozday-mgnify/rust-mdbg, mg-summary branch)
# --dump-read-minimizers writes {PREFIX}.{thread}.read_minimizers (LZ4-compressed TSV)
# and {PREFIX}.minimizer_table (plain-text TSV: hash <-> l-mer lookup)
"$(dirname "$0")/../bin/rust-mdbg" \
    "${COMBINED}" \
    -k "${K}" \
    --density "${DENSITY}" \
    -l "${L}" \
    --minabund "${MINABUND}" \
    --prefix "${PREFIX}" \
    --dump-read-minimizers \
2>&1 | tee "${OUTPUT_DIR}/output.txt"

UNIQUE_MINIMIZERS=$(grep -vc '^#' "${MINIMIZER_TABLE}" 2>/dev/null || echo 0)
echo "Unique minimizers in minimizer_table: ${UNIQUE_MINIMIZERS}"

/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python "$(dirname "$0")/parse_gfa.py" \
    -n 10000 \
    --read-minimizers "${PREFIX}" \
    --minimizer-table "${MINIMIZER_TABLE}" \
    --insert-sizes-out "${PREFIX}.insert_sizes.tsv" \
    --json "${PREFIX}.graph_summary.json" \
    "${GFA}"

/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python "$(dirname "$0")/read_minimizers.py" \
    --summary \
    --json "${PREFIX}.minimizer_summary.json" \
    "${PREFIX}"

# echo "rust_mdbg_out.gfa head: $( cat ${OUTPUT_DIR}/rust_mdbg_out.gfa | sed -nE 's/^.*LN:i:([0-9]+).*$/\1/p' | sort -nr | head -n 5 )"

# docker run --rm \
#     -v "${OUTPUT_DIR}:/output" \
#     --entrypoint to_basespace \
#     rust-mdbg \
#     --gfa /output/rust_mdbg_out.unitigs.gfa \
#     --sequences /output/rust_mdbg_out
# 
# echo "rust_mdbg_out.unitigs.gfa.complete.gfa wc stats: $(wc ${OUTPUT_DIR}/rust_mdbg_out.unitigs.gfa.complete.gfa)"
