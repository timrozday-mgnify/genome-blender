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

Apply platform-appropriate error profiles:

- Position-dependent substitution errors for short reads
- Indel-dominated error profiles for long reads
- Realistic quality scores in Phred+33 encoding

### Output

- FASTQ files with quality scores reflecting the applied error model
- Ground-truth files recording the true origin of each read and all introduced variants

## Requirements

- Python 3.10+
- numpy
- pysam
- Biopython

## License

TBD
