#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="$(cd ../genome-blender_run/single_short_shallow/output && pwd)"
OUTPUT_DIR="${INPUT_DIR}/metaMDBG"
mkdir -p "${OUTPUT_DIR}"

FASTQ_FILES=("${INPUT_DIR}"/*.fastq)
INPUT_ARGS=""
for f in "${FASTQ_FILES[@]}"; do
    INPUT_ARGS="${INPUT_ARGS} /input/$(basename "${f}")"
done

# Run assembly inside a named volume, then copy results out.
# metaMDBG has internal file-copy operations that fail on
# macOS bind-mounts due to filesystem permission differences.
docker volume create metamdbg_work >/dev/null

# Copy input reads into the volume
docker run --rm \
    -v "${INPUT_DIR}:/input:ro" \
    -v metamdbg_work:/work \
    --entrypoint bash \
    meta-mdbg -c 'cp /input/*.fastq /work/'

# Assembly parameters
KMER_SIZE=11          # minimizer length (default: 15; lower for short reads)
DENSITY=0.01          # fraction of k-mers used for assembly (default: 0.005)
MIN_ABUNDANCE=2       # min k-min-mer abundance; 0 = rescue mode
MIN_OVERLAP=150       # min read overlap in bp (default: 1000; match read length)

# Assemble
docker run --rm \
    -v metamdbg_work:/work \
    meta-mdbg asm \
    --out-dir /work/output \
    --in-ont $(for f in "${FASTQ_FILES[@]}"; do echo -n " /work/$(basename "${f}")"; done) \
    --kmer-size "${KMER_SIZE}" \
    --density-assembly "${DENSITY}" \
    --min-abundance "${MIN_ABUNDANCE}" \
    --min-read-overlap "${MIN_OVERLAP}" \
    --skip-correction \
    --threads 4

# Extract assembly graph (--k 0 lists available k values)
docker run --rm \
    -v metamdbg_work:/work \
    meta-mdbg gfa \
    --assembly-dir /work/output \
    --k 0 \
    --threads 4

# Copy results to host
docker run --rm \
    -v metamdbg_work:/work \
    -v "${OUTPUT_DIR}:/out" \
    --entrypoint bash \
    meta-mdbg -c 'cp -r /work/output/* /out/'

docker volume rm metamdbg_work >/dev/null
