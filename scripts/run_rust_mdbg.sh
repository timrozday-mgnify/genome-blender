#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/multi_genome_full/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/rust-mdbg"
mkdir -p "${OUTPUT_DIR}"

# Whether the input is paired-end (R1 + R2 files).
# Set to 1 to pass R1 and R2 separately via --reads2 (rust-mdbg assigns
# interleaved indices: 1=R1, 2=R2, 3=R1, … enabling arithmetic pairing
# in parse_gfa.py --interleaved-pairs).
# Set to 0 to use a single FASTQ file as single-end input.
PAIRED_END=1

# Detect R1 and R2 files (for paired-end) or collect all FASTQ files (single-end).
if [[ "${PAIRED_END}" -eq 1 ]]; then
    R1=""; for f in "${INPUT_DIR}"/*_R1.fastq.gz "${INPUT_DIR}"/*_R1.fastq; do
        [[ -f "$f" ]] && R1="$f" && break; done
    R2=""; for f in "${INPUT_DIR}"/*_R2.fastq.gz "${INPUT_DIR}"/*_R2.fastq; do
        [[ -f "$f" ]] && R2="$f" && break; done
    if [[ -z "${R1}" || -z "${R2}" ]]; then
        echo "ERROR: could not find R1/R2 FASTQ files in ${INPUT_DIR}" >&2
        exit 1
    fi
    echo "Paired-end reads: R1=${R1}  R2=${R2}"
    READS_FILE="${R1}"
    READS2_FLAG=(--reads2 "${R2}")
else
    # Single-end: prefer .fastq.gz, fall back to .fastq
    SE_GZ=( "${INPUT_DIR}"/*.fastq.gz )
    SE_PLAIN=( "${INPUT_DIR}"/*.fastq )
    if [[ -f "${SE_GZ[0]:-}" ]]; then
        READS_FILE="${SE_GZ[0]}"
    else
        READS_FILE="${SE_PLAIN[0]}"
    fi
    READS2_FLAG=()
fi

# rust-mdbg parameters
K=4   # k-mer size (minimizer k-mer)
L=12  # l-mer length (minimizer length)
MINABUND=2

# rust-mdbg output paths
PREFIX="${OUTPUT_DIR}/rust_mdbg_out"
GFA="${PREFIX}.gfa"
MINIMIZER_TABLE="${PREFIX}.minimizer_table"

# Estimate mean read length from the first 1000 reads, then derive density.
# The minimizer length l and density are related by:
#   read_length * density * 1.25 = l  =>  density = l / (read_length * 1.25)
READS_STATS_JSON="${OUTPUT_DIR}/reads_stats.json"
/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python \
    "$(dirname "$0")/reads_summary.py" "${READS_FILE}" -n 1000 \
    --json "${READS_STATS_JSON}"
MEAN_READ_LEN=$(
    /Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python3 -c \
        "import json; d=json.load(open('${READS_STATS_JSON}')); print(int(d['mean'] + 0.5))"
)
DENSITY=$(
    /Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python3 -c \
        "print(f'{${L} / (${MEAN_READ_LEN} * 1.25):.4f}')"
)
echo "Estimated mean read length: ${MEAN_READ_LEN} bp"
echo "Using density: ${DENSITY}  (l=${L} / (${MEAN_READ_LEN} * 1.25))"

# Run rust-mdbg directly (local build from timrozday-mgnify/rust-mdbg, mg-summary branch)
# --dump-read-minimizers writes the reads LMDB index ({PREFIX}.index.lmdb)
# and {PREFIX}.minimizer_table (plain-text TSV: hash <-> l-mer lookup).
# When --reads2 is given, R1 gets odd indices (1, 3, 5, …) and R2 gets even (2, 4, 6, …).
"$(dirname "$0")/../bin/rust-mdbg" \
    "${READS_FILE}" \
    "${READS2_FLAG[@]}" \
    -k "${K}" \
    --density "${DENSITY}" \
    -l "${L}" \
    --minabund "${MINABUND}" \
    --prefix "${PREFIX}" \
    --dump-read-minimizers \
2>&1 | tee "${OUTPUT_DIR}/output.txt"

UNIQUE_MINIMIZERS=$(grep -vc '^#' "${MINIMIZER_TABLE}" 2>/dev/null || echo 0)
echo "Unique minimizers in minimizer_table: ${UNIQUE_MINIMIZERS}"

# Build parse_gfa.py paired-end flags.
# --interleaved-pairs: arithmetic pairing (0↔1, 2↔3, …) derived from the
#   interleaved 1-based read indices written by rust-mdbg v2 --reads2 mode.
_PAIRED_FLAGS=()
if [[ "${PAIRED_END}" -eq 1 ]]; then
    _PAIRED_FLAGS+=(--interleaved-pairs)
fi

/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python "$(dirname "$0")/parse_gfa.py" \
    -n 100000 \
    --top-paths 5 \
    --sample-component-proportion 0.5 \
    --matcher pseudo-match \
    --read-minimizers "${PREFIX}" \
    --minimizer-table "${MINIMIZER_TABLE}" \
    --insert-sizes-out "${PREFIX}.insert_sizes.tsv" \
    --json "${PREFIX}.graph_summary.json" \
    "${_PAIRED_FLAGS[@]}" \
    "${GFA}"

#    --read-mappings-out "${PREFIX}.read_mappings.tsv" \
#    --paths-out "${PREFIX}.paths.tsv" \

