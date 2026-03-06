# genome-blender

Simulate whole genome sequencing data from reference genomes for benchmarking bioinformatics pipelines.

## Motivation

Benchmarking variant callers, assemblers, and other genomics tools requires sequencing data where the ground truth is known. Real sequencing data lacks this -- you cannot know every true variant or the exact origin of each read. genome-blender generates synthetic but realistic WGS data from reference genomes, tracking every introduced variant and the true genomic origin of every read.

## Features

### Genome modification

Introduce controlled biological variation into reference genomes:

- **Point mutations (SNPs)** -- single base substitutions at configurable rates
- **Small indels** -- insertions and deletions (< 50 bp)
- **Structural variants** -- large deletions, duplications, inversions, and translocations
- **Horizontal gene transfer (HGT)** -- transfer segments between genomes
- **Genome rearrangements** -- large-scale reordering and restructuring of contigs

All modifications are recorded so that downstream analysis can be evaluated against ground truth.

### Fragment generation

Simulate library preparation by generating DNA fragments from genomes:

- Configurable fragment size distributions
- Optional **GC bias** modelling to reproduce the non-uniform coverage observed in real PCR-amplified libraries

### Read simulation

Generate sequencing reads from fragments:

- **Short single-end reads** (e.g. Illumina SE)
- **Short paired-end reads** (e.g. Illumina PE) with configurable insert sizes
- **Long reads** (e.g. PacBio, Oxford Nanopore) with realistic length distributions

### Sequencing error models

HMM-based error models produce correlated quality score patterns along reads (inspired by pbsim2). Errors (substitutions, insertions, deletions) are then sampled from those quality scores. Built-in profiles:

- **Illumina** -- high quality (Q30--Q37), predominantly substitution errors (~80%)
- **PacBio** -- moderate quality (Q10--Q20), indel-dominated errors (~85%)
- **Oxford Nanopore** -- lower quality (Q7--Q15), indel-dominated errors (~80%)

Select a profile with `--error-model illumina|pacbio|nanopore` or omit for no errors.

### Output

- FASTQ files with quality scores reflecting the applied error model
- Ground-truth BAM recording the true genomic origin of each read with accurate CIGAR strings

## Usage

Prepare an input CSV with columns `genome_id`, `fasta_path`, `abundance`:

```csv
genome_id,fasta_path,abundance
ecoli,/path/to/ecoli.fasta,0.7
staph,/path/to/staph.fasta,0.3
```

### Command-line options

Generate reads by passing options directly:

```bash
# Single-end, no error model
python generate_reads.py --input-csv genomes.csv --num-reads 1000 --output-prefix out --seed 42

# Paired-end with Illumina error model
python generate_reads.py --input-csv genomes.csv --num-reads 1000 --output-prefix out \
    --paired-end --error-model illumina --seed 42

# Long reads with Nanopore error model
python generate_reads.py --input-csv genomes.csv --num-reads 500 --output-prefix out \
    --read-length-mean 5000 --read-length-variance 4000000 \
    --fragment-mean 8000 --fragment-variance 4000000 \
    --error-model nanopore --seed 42
```

### YAML configuration file

Instead of specifying every option on the command line, you can provide a YAML config file with `--config`. Keys use the same names as CLI options (hyphens or underscores both work). Any option passed on the command line takes precedence over the config file value.

```bash
# Run entirely from a config file
python generate_reads.py --config my_config.yaml

# Config file with CLI overrides
python generate_reads.py --config my_config.yaml --num-reads 5000 --seed 99
```

An example config file is provided at [`example_config.yaml`](example_config.yaml):

```yaml
# --- Required --------------------------------------------------------
input_csv: genomes.csv
num_reads: 10000
output_prefix: output/sim_reads

# --- Fragment generation ---------------------------------------------
fragment_mean: 300.0
fragment_variance: 300.0

# --- Read generation -------------------------------------------------
read_length_mean: 150.0
read_length_variance: 10.0
paired_end: true

# --- GC bias ---------------------------------------------------------
gc_bias_strength: 0.0  # 0 = no bias

# --- Amplicon mode ---------------------------------------------------
amplicon: false  # set true to skip shearing and replicate input sequences

# --- Error model -----------------------------------------------------
# Options: none, illumina, pacbio, nanopore
error_model: illumina

# --- Quality calibration ---------------------------------------------
# Options: phred, log-linear, sigmoid
quality_calibration_model: phred
qcal_variability: 0.0
qcal_intercept: -0.3
qcal_slope: -0.08
qcal_floor: 1.0e-7
qcal_ceiling: 0.5
qcal_steepness: 0.25
qcal_midpoint: 15.0

# --- Reproducibility -------------------------------------------------
seed: 42
```

The three required parameters (`input_csv`, `num_reads`, `output_prefix`) must be provided either in the config file or on the command line.

## Parameter estimation tools

The `containers/` directory provides Dockerfiles for external tools being trialled for estimating realistic simulation parameters (e.g. fragment size distributions, error profiles, community composition) from existing metagenome sequencing data. These are not part of the core genome-blender pipeline but support the goal of generating data that closely mirrors real experiments.

| Tool | Description |
|------|-------------|
| [cuttlefish3](https://github.com/COMBINE-lab/cuttlefish/tree/cuttlefish3) | Compacted de Bruijn graph construction |
| [rust-mdbg](https://github.com/ekimb/rust-mdbg) | Minimizer-space de Bruijn graph assembler |
| [metaMDBG](https://github.com/GaetanBenoitDev/metaMDBG) | Metagenome assembler based on minimizer-space de Bruijn graphs |
| [Bifrost](https://github.com/pmelsted/bifrost) | Parallel construction and indexing of compacted de Bruijn graphs |
| [SPAdes](https://github.com/ablab/spades) | Genome assembler (pre-built image from BioContainers) |

Build a container image with:

```bash
docker buildx build -t <tool-name> containers/<tool>/
```

### Run scripts

The `scripts/` directory contains ready-to-run shell scripts that execute each tool via Docker against simulated FASTQ files. All scripts read paired-end input from `../genome-blender_run/single_short_shallow/output/` and write output to a tool-specific subdirectory alongside the input.

#### `run_cuttlefish3.sh` — Compacted de Bruijn graph (unitigs)

Runs [cuttlefish3](https://github.com/COMBINE-lab/cuttlefish/tree/cuttlefish3) to build a compacted de Bruijn graph from reads, outputting unitigs as FASTA.

| Argument | Value | Description |
|----------|-------|-------------|
| `--read` | — | Treat input as sequencing reads (not reference) |
| `-k` | `31` | K-mer length (must be odd, max 63) |
| `-o` | output path | Output file prefix |
| `-w` | `/tmp/cf3` | Working directory (uses a Docker volume to avoid macOS filesystem issues) |
| `-c` | `1` | Minimum (k+1)-mer frequency cutoff |

#### `run_bifrost.sh` — Compacted coloured de Bruijn graph (GFA)

Runs [Bifrost](https://github.com/pmelsted/bifrost) to build a compacted de Bruijn graph, outputting GFA directly.

| Argument | Value | Description |
|----------|-------|-------------|
| `-s` | each FASTQ | Input read files (k-mers occurring once are filtered) |
| `-k` | `31` | K-mer length |
| `-o` | output path | Output file prefix |
| `-i` | — | Clip short tips |
| `-d` | — | Delete small isolated contigs |
| `-t` | `4` | Number of threads |

#### `run_megahit.sh` — Succinct de Bruijn graph assembly (FASTG → GFA)

Runs [MEGAHIT](https://github.com/voutcn/megahit) for metagenome assembly, then converts the intermediate contigs to FASTG using `megahit_core contig2fastg`, and finally converts FASTG to GFA using [gfatools](https://github.com/lh3/gfatools). All graph cleaning is disabled to preserve the raw graph structure.

| Variable | Flag | Value | Description |
|----------|------|-------|-------------|
| `K_LIST` | `--k-list` | `21,31,41,51` | Comma-separated k-mer sizes to iterate over |
| `BUBBLE_LEVEL` | `--bubble-level` | `0` | Bubble merging intensity (0 = disabled) |
| `PRUNE_LEVEL` | `--prune-level` | `0` | Low-depth pruning strength (0 = disabled) |
| `MAX_TIP_LEN` | `--max-tip-len` | `0` | Tip removal length threshold (0 = keep all) |
| `CLEANING_ROUNDS` | `--cleaning-rounds` | `0` | Graph cleaning iterations (0 = skip) |
| `DISCONNECT_RATIO` | `--disconnect-ratio` | `0` | Depth-ratio unitig disconnection (0 = disabled) |
| `LOW_LOCAL_RATIO` | `--low-local-ratio` | `0` | Neighbourhood depth filtering (0 = disabled) |
| `MIN_COUNT` | `--min-count` | `1` | Minimum k-mer multiplicity (1 = keep singletons) |
| — | `--no-mercy` | — | Do not add mercy k-mers |
| — | `--no-local` | — | Disable local assembly |
| — | `--keep-tmp-files` | — | Preserve intermediate contigs for FASTG conversion |

Note: MEGAHIT always compacts unitigs — there is no flag to output the uncompacted de Bruijn graph.

#### `run_spades.sh` — Metagenomic assembly (GFA)

Runs [SPAdes](https://github.com/ablab/spades) in metagenomic mode. SPAdes natively outputs `assembly_graph.gfa` alongside contigs.

| Argument | Value | Description |
|----------|-------|-------------|
| `--meta` | — | Metagenomic assembly mode |
| `-1`, `-2` | R1, R2 FASTQs | Paired-end input files |
| `-o` | output path | Output directory |
| `-t` | `4` | Number of threads |

#### `run_metaMDBG.sh` — Minimizer-space de Bruijn graph assembly

Runs [metaMDBG](https://github.com/GaetanBenoitDev/metaMDBG) for metagenome assembly using minimizer-space de Bruijn graphs. Designed for long reads; parameters are adjusted for short-read input. Uses a Docker volume for working data due to macOS bind-mount permission issues.

| Variable | Flag | Value | Description |
|----------|------|-------|-------------|
| `KMER_SIZE` | `--kmer-size` | `11` | Minimizer length (default 15; reduced for short reads) |
| `DENSITY` | `--density-assembly` | `0.01` | Fraction of k-mers used for assembly (default 0.005) |
| `MIN_ABUNDANCE` | `--min-abundance` | `2` | Minimum k-min-mer abundance (0 = rescue mode) |
| `MIN_OVERLAP` | `--min-read-overlap` | `150` | Minimum read overlap in bp (default 1000; set to match read length) |
| — | `--skip-correction` | — | Skip long-read error correction (not useful for Illumina) |

The `gfa` subcommand is run after assembly with `--k 0` to list available k values for graph extraction.

#### `run_rust_mdbg.sh` — Minimizer-space de Bruijn graph (GFA)

Runs [rust-mdbg](https://github.com/ekimb/rust-mdbg) to build a minimizer-space de Bruijn graph, then converts to base-space with `to_basespace`. Designed for long reads; all FASTQ files are concatenated into a single input since rust-mdbg accepts only one reads file.

| Argument | Value | Description |
|----------|-------|-------------|
| `-k` | `4` | K-min-mer order |
| `--density` | `0.1` | Minimizer density (δ) |
| `-l` | `8` | Minimizer length |
| `--minabund` | `2` | Minimum k-min-mer abundance |
| `--prefix` | output path | Output file prefix |

## Comparison with other simulators

Several existing tools address overlapping aspects of read simulation. genome-blender aims to combine metagenome-aware abundance modelling, configurable error profiles, and full ground-truth tracking in a single Python toolkit.

| Feature | genome-blender | [NEAT](https://github.com/ncsa/NEAT) | [CAMISIM](https://github.com/CAMI-challenge/CAMISIM) | [ART](https://www.niehs.nih.gov/research/resources/software/biostatistics/art) | [InSilicoSeq](https://github.com/HadrienG/InSilicoSeq) |
|---------|---------------|------|---------|-----|-------------|
| Language | Python | Python | Nextflow + Python | C++ | Python |
| Metagenome mixing | Yes (CSV abundances) | No (single genome) | Yes (community profiles) | No (single genome) | Yes (abundances) |
| Paired-end reads | Yes (FR orientation) | Yes (FR orientation) | Yes (via ART/wgsim) | Yes (FR orientation) | Yes |
| Error models | HMM-based (Illumina, PacBio, ONT) | Empirical (learned from BAM) | Delegates to ART/wgsim/NanoSim | Empirical profiles | KDE from real data |
| Quality calibration | Phred / log-linear / sigmoid | Learned from data | Via backend simulator | Empirical | KDE |
| GC bias | Configurable accept/reject | No | No | No | Yes |
| Fragment model | NegativeBinomial / Poisson / fixed | Empirical (learned from BAM) | Configurable mean/sd | Configurable mean/sd | Empirical |
| Ground-truth BAM | Yes | Yes (with golden VCF) | No | Yes (SAM output) | No |
| Amplicon mode | Yes | No | No | No | No |
| Long read support | Yes (ONT, PacBio profiles) | No | Yes (via NanoSim) | No | No |
| Variant simulation | Planned | Yes (SNPs, indels, SVs) | Yes (via sgEvolver) | No | No |
| Config format | YAML + CLI | YAML | INI / Nextflow | CLI only | CLI |

### Paired-end implementation

All tools that generate paired-end reads from scratch use the same FR (forward-reverse) orientation, matching the Illumina sequencing chemistry:

- **R1** is sequenced from the 5' end of the forward strand of the fragment
- **R2** is the reverse complement of the 3' end of the fragment

In genome-blender: `r1_seq = fragment[:read_len]`, `r2_seq = revcomp(fragment[-read_len:])`. NEAT applies the same logic: it extracts the forward segment for R1 and applies `reverse_complement()` to the reverse segment for R2. CAMISIM delegates paired-end generation entirely to ART (`art_illumina -p`) or wgsim, which implement the same FR convention internally.

## Requirements

- Python 3.10+
- PyTorch
- Pyro-PPL
- pysam
- Biopython
- Typer
- PyYAML

## License

TBD
