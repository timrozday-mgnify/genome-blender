#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/multi_genome_deep/output && pwd)"
RUST_MDBG_DIR="${INPUT_DIR}/rust-mdbg"
ASF_SAMPLE_DIR="${RUST_MDBG_DIR}/asf_sample"
OUTPUT_DIR="${ASF_SAMPLE_DIR}"
mkdir -p "${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# rust-mdbg parameters (must match the values used to build the index)
# ---------------------------------------------------------------------------

# rust-mdbg output prefix (same as in run_asf_sample.sh).
PREFIX="${RUST_MDBG_DIR}/rust_mdbg_out"

# ---------------------------------------------------------------------------
# Input files
# ---------------------------------------------------------------------------

# Sampled minimizer paths from asf_sample.py.
PATHS_JSONL="${ASF_SAMPLE_DIR}/paths.jsonl"

# Insert-size distribution JSON from asf_sample.py.
INSERT_SIZE_JSON="${ASF_SAMPLE_DIR}/insert_size.json"

# Detect R1 and R2 FASTQ files.
R1=""; for f in "${INPUT_DIR}"/*_R1.fastq.gz "${INPUT_DIR}"/*_R1.fastq; do
    [[ -f "$f" ]] && R1="$f" && break; done
R2=""; for f in "${INPUT_DIR}"/*_R2.fastq.gz "${INPUT_DIR}"/*_R2.fastq; do
    [[ -f "$f" ]] && R2="$f" && break; done
if [[ -z "${R1}" || -z "${R2}" ]]; then
    echo "ERROR: could not find R1/R2 FASTQ files in ${INPUT_DIR}" >&2
    exit 1
fi
echo "Paired-end reads: R1=${R1}  R2=${R2}"

# ---------------------------------------------------------------------------
# reconstruct_sequences parameters
# ---------------------------------------------------------------------------

# Sequence selection mode: best | random | common | consensus
# 'best'      -- highest-scoring read alignment
# 'random'    -- sample one alignment (optionally score-weighted)
# 'common'    -- most frequent exact sequence per gap span
# 'consensus' -- MSA consensus via abpoa or spoa (must be on PATH)
MODE=best

# Minimum chained minimizer anchors for a read alignment to be considered.
MIN_ANCHORS=3

# Minimum chain length for partial reads used only at the first and last path gap.
# Set lower than MIN_ANCHORS to recover reads that only partially overlap a path end,
# which reduces N-padding at the ends of reconstructed sequences.
TERMINAL_MIN_ANCHORS=1

# Maximum reads examined per minimizer lookup (caps runtime for high-coverage positions).
MAX_READS_PER_MINIMIZER=200

# Minimum fraction of path minimizers that must have read coverage; paths below
# this threshold are omitted from the output FASTA.
MIN_COV_FRACTION=0.5

# Scoring components to include in read alignment ranking.
# Set to 1 to enable, 0 to disable each component.
WEIGHT_COVERAGE=1   # --weight-coverage / --no-weight-coverage
WEIGHT_INSERT=1     # --weight-insert   / --no-weight-insert
WEIGHT_MATE=1       # --weight-mate     / --no-weight-mate

# Random seed for reproducibility (affects 'random' mode and tie-breaking).
SEED=42

# Gap-fill character inserted for path spans with no read coverage.
GAP_FILL_CHAR=N

# Minimizer density (minimizers per basepair) used to estimate gap-fill length
# when no read covers a span.  Must match the value used for rust-mdbg indexing.
DENSITY=0.01

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_FA="${OUTPUT_DIR}/reconstructed.fa"

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

echo "Running reconstruct_sequences.py ..."
/Users/timrozday/miniforge3/envs/genome_blender_dev/bin/python \
    "$(dirname "$0")/reconstruct_sequences.py" \
    "${PATHS_JSONL}" \
    --prefix "${PREFIX}" \
    --reads "${R1}" \
    --reads "${R2}" \
    --insert-size-json "${INSERT_SIZE_JSON}" \
    --mode "${MODE}" \
    --min-anchors "${MIN_ANCHORS}" \
    --terminal-min-anchors "${TERMINAL_MIN_ANCHORS}" \
    --max-reads-per-minimizer "${MAX_READS_PER_MINIMIZER}" \
    --min-coverage-fraction "${MIN_COV_FRACTION}" \
    $([[ "${WEIGHT_COVERAGE}" -eq 1 ]] && echo "--weight-coverage" || echo "--no-weight-coverage") \
    $([[ "${WEIGHT_INSERT}" -eq 1 ]] && echo "--weight-insert" || echo "--no-weight-insert") \
    $([[ "${WEIGHT_MATE}" -eq 1 ]] && echo "--weight-mate" || echo "--no-weight-mate") \
    --seed "${SEED}" \
    --gap-fill-char "${GAP_FILL_CHAR}" \
    --density "${DENSITY}" \
    --output "${OUTPUT_FA}" \
2>&1 | tee "${OUTPUT_DIR}/reconstruct_sequences.log"

echo "Output written to: ${OUTPUT_FA}"
