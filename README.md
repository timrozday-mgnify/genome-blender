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
