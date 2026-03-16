#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/multi_genome_full/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/rust-mdbg"
mkdir -p "${OUTPUT_DIR}"

# Whether the input is paired-end (R1 + R2 files).  Set to 1 to interleave;
# set to 0 to concatenate all FASTQ files as single-end.
PAIRED_END=1

# rust-mdbg accepts a single reads file.
# For paired-end input the R1 and R2 files must be interleaved so that
# sequential read indices (v2 format) follow the pattern 1=R1, 2=R2, 3=R1, …
# which allows --interleaved-pairs arithmetic pairing in parse_gfa.py.
COMBINED="${OUTPUT_DIR}/combined_reads.fastq"

_open() { case "$1" in *.gz) zcat "$1";; *) cat "$1";; esac; }
_write() { case "${COMBINED}" in *.gz) gzip -c;; *) cat;; esac > "${COMBINED}"; }

if [[ "${PAIRED_END}" -eq 1 ]]; then
    # Detect gzipped output from genome-blender (default: .fastq.gz)
    R1=""; for f in "${INPUT_DIR}"/*_R1.fastq.gz "${INPUT_DIR}"/*_R1.fastq; do
        [[ -f "$f" ]] && R1="$f" && break; done
    R2=""; for f in "${INPUT_DIR}"/*_R2.fastq.gz "${INPUT_DIR}"/*_R2.fastq; do
        [[ -f "$f" ]] && R2="$f" && break; done
    if [[ -z "${R1}" || -z "${R2}" ]]; then
        echo "ERROR: could not find R1/R2 FASTQ files in ${INPUT_DIR}" >&2
        exit 1
    fi
    # Name the combined file to match the input extension
    case "${R1}" in *.gz) COMBINED="${OUTPUT_DIR}/combined_reads.fastq.gz";;
                    *)    COMBINED="${OUTPUT_DIR}/combined_reads.fastq";; esac
    echo "Interleaving paired-end reads: ${R1}  +  ${R2}"
    # Interleave: collapse each file to one tab-separated line per record,
    # then alternate records from R1 and R2, then expand back to 4 lines each.
    paste <(paste - - - - < <(_open "${R1}")) \
          <(paste - - - - < <(_open "${R2}")) \
    | awk -F'\t' '{print $1; print $2; print $3; print $4;
                   print $5; print $6; print $7; print $8}' \
    | _write
else
    # Single-end: prefer .fastq.gz, fall back to .fastq
    SE_GZ=( "${INPUT_DIR}"/*.fastq.gz )
    SE_PLAIN=( "${INPUT_DIR}"/*.fastq )
    if [[ -f "${SE_GZ[0]}" ]]; then
        COMBINED="${OUTPUT_DIR}/combined_reads.fastq.gz"
        cat "${SE_GZ[@]}" | _write
    else
        COMBINED="${OUTPUT_DIR}/combined_reads.fastq"
        cat "${SE_PLAIN[@]}" | _write
    fi
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
    "$(dirname "$0")/reads_summary.py" "${COMBINED}" -n 1000 \
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

# Build parse_gfa.py paired-end flags.
# --interleaved-pairs: use arithmetic pairing (1↔2, 3↔4, …) for v2 binary
#   files where names are sequential integer indices; requires interleaved input.
# --reads: source read names from the FASTQ so human-readable names appear in
#   output alongside the integer v2 indices.
_PAIRED_FLAGS=()
if [[ "${PAIRED_END}" -eq 1 ]]; then
    _PAIRED_FLAGS+=(--interleaved-pairs --reads "${COMBINED}")
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
