# genome-blender

A toolkit for generating synthetic whole genome sequencing (WGS) data from reference genomes, intended for benchmarking bioinformatics pipelines.

## Project overview

genome-blender takes reference genome(s) as input and produces realistic simulated sequencing data (FASTQ) alongside a ground-truth BAM recording the true genomic origin of each read. The pipeline has several stages:

1. **Genome modification** -- optionally apply biological variation to input genomes (mutations, rearrangements, HGT)
2. **Fragment generation** -- shear genomes into DNA fragments with configurable size distributions and optional GC bias
3. **Read generation** -- sample reads from fragments (short single-end, short paired-end, or long reads)
4. **Error simulation** -- apply platform-specific sequencing error models and produce FASTQ with realistic quality scores

A ground-truth record of all introduced variants and the true origin of each read should be maintained for benchmarking purposes.

## Language and architecture

- Primary language: **Python** (3.10+)
- Performance-critical components may be rewritten in **Rust** with Python bindings (via PyO3/maturin)
- Core simulation pipeline lives in the `genome_blender/` package; `generate_reads.py` is a backwards-compatible shim that delegates to it
- CLI framework: **Typer** (with `Annotated` type hints for options); supports `--config` YAML files via **PyYAML**
- Probability distributions: **pyro.distributions** (wraps torch.distributions) -- use `pyro.distributions.NegativeBinomial`, `LogNormal`, etc.
- Use **torch** for random number generation and tensor operations
- Use **numpy** for numerical work where torch is not needed
- Use **pysam** for reading/writing FASTA/FASTQ and BAM files
- Use **Biopython** (`Bio.SeqIO`, `Bio.Seq`) for sequence parsing and manipulation where pysam is insufficient
- Packaging: use a `pyproject.toml` with hatchling; no `setup.py`
- Analysis scripts in `scripts/` use **Typer** (with `Annotated` type hints) for CLI arguments, **rustworkx** for graph algorithms, and **rich** for progress bars

## Domain terminology

These terms appear throughout the codebase. Use them consistently.

- **Reference genome**: input FASTA file containing one or more contigs/chromosomes
- **Contig**: a single sequence record within a FASTA file
- **Fragment**: a contiguous piece of DNA excised from a genome, simulating physical shearing or enzymatic digestion during library preparation
- **Read**: a sequence string produced by a simulated sequencing instrument from a fragment; has an associated quality string
- **Insert size**: the length of the original DNA fragment between adapters in paired-end sequencing; the two paired reads are sequenced inward from each end of the insert
- **GC bias**: non-uniform coverage caused by GC content affecting PCR amplification efficiency; both high-GC and low-GC regions tend to be underrepresented
- **Phred quality score**: Q = -10 * log10(P_error); encoded as ASCII characters in FASTQ using Phred+33 offset
- **Coverage / depth**: the average number of times each base in the genome is sequenced (e.g. 30x)
- **SNP**: single nucleotide polymorphism -- a single base substitution
- **Indel**: small insertion or deletion (typically < 50 bp)
- **Structural variant (SV)**: large-scale genomic alteration (> 50 bp) including deletions, duplications, inversions, translocations
- **HGT (horizontal gene transfer)**: transfer of genetic material between organisms outside of vertical inheritance
- **Error model**: a statistical model describing the type, rate, and positional distribution of sequencing errors for a given platform

## Coding conventions

- Follow PEP 8 and PEP 257
- Use type hints throughout (`from __future__ import annotations` at top of every module)
- Use `pathlib.Path` for all file paths, not `os.path`
- Prefer dataclasses or named tuples over plain dicts for structured data
- Use `logging` module, not print statements, for status messages
- Coordinate systems: use 0-based, half-open intervals internally (BED convention); convert at I/O boundaries
- Random number generation: always accept a `seed` or `rng` (`torch.Generator`) parameter for reproducibility; never use bare `random.random()` or unseeded RNGs
- FASTQ output: always use Phred+33 encoding

## Testing

- Use **pytest** for all tests
- Test files go in `tests/` mirroring the source structure
- Aim for unit tests on core logic (fragment generation, error models, coordinate math)
- Use small synthetic genomes (a few hundred bases) in tests, not real genome files

## File formats

- Input genomes: FASTA (.fa, .fasta, .fna), optionally gzipped
- Output reads: FASTQ (.fq, .fastq), optionally gzipped
- Ground-truth BAM: `{output_prefix}.bam` produced alongside FASTQ by the `genome_blender` CLI (`generate_reads.py` shim or `python -m genome_blender`); records each read's true alignment to its source fragment
  - Reference names use `{genome_id}:{contig_id}` format
  - CIGAR reflects the error model: all-match (`M`) when no error model is applied; mix of `M`/`I`/`D` ops when errors are introduced
  - MAPQ 255 (ground truth)
  - Paired-end reads have proper FLAG bits (`is_paired`, `is_proper_pair`, `is_read1`/`is_read2`), mate positions, and template length
  - Written unsorted; user can `samtools sort` if needed
- Ground truth (variants): BED or custom TSV for variant locations (not yet implemented)
- Input CSV table for the simulation CLI: columns `genome_id,fasta_path,abundance` -- abundances are relative and will be normalised to sum to 1

## Performance considerations

- Fragment and read generation can be memory-intensive for large genomes; prefer streaming/chunked approaches over loading entire genomes into memory when practical
- Candidate modules for Rust acceleration: fragment sampling, error model application, quality score generation
- Profile before optimising; use `cProfile` or `py-spy`

## Containers

The `containers/` directory holds Dockerfiles (and occasionally Singularity definition files) for external tools being trialled for estimating simulation parameters from existing metagenome sequencing data. Each tool has its own subdirectory; built images are saved as `.tar` files (excluded from git).

- **cuttlefish3** (`containers/cuttlefish3/`) -- compacted de Bruijn graph construction (Dockerfile + Singularity def)
- **rust-mdbg** (`containers/rust-mdbg/`) -- minimizer-space de Bruijn graph assembler
- **metaMDBG** (`containers/metaMDBG/`) -- metagenome assembler based on minimizer-space de Bruijn graphs
- **bifrost** (`containers/bifrost/`) -- parallel construction and indexing of compacted de Bruijn graphs
- **spades** (`containers/spades/`) -- SPAdes genome assembler (pre-built image from BioContainers, no Dockerfile)

## Run scripts

The `scripts/` directory contains shell scripts that execute external tools via Docker against simulated FASTQ files, and Python analysis scripts. Shell scripts have configurable variables at the top for all tool-specific parameters.

- **`run_cuttlefish3.sh`** -- compacted de Bruijn graph from reads (cuttlefish3); parameters: `K`, `MIN_LEN`, `CUTOFF`, `INPUT_MODE`, `COLOR`
- **`run_bifrost.sh`** -- compacted de Bruijn graph in GFA (Bifrost); parameters: `K`, `THREADS`, `BLOOM_BITS`, `CLIP_TIPS`, `DEL_ISOLATED`, `COLORS`, `FASTA_OUT`, `BFG_OUT`, `VERBOSE`, `NO_COMPRESS`, `NO_INDEX`
- **`run_megahit.sh`** -- succinct de Bruijn graph assembly with FASTG→GFA conversion (MEGAHIT)
- **`run_spades.sh`** -- metagenomic assembly with default simplification (SPAdes)
- **`run_spades_raw.sh`** -- metagenomic assembly with all graph simplification disabled (SPAdes); uses the hidden `--configs-dir` flag with a patched `simplification.info` and stripped `meta_mode.info` to prevent tip clipping, bubble removal, erroneous-connection removal, and isolated-edge removal
- **`run_metaMDBG.sh`** -- minimizer-space de Bruijn graph assembly (metaMDBG)
- **`run_rust_mdbg.sh`** -- minimizer-space de Bruijn graph in GFA (rust-mdbg)
- **`parse_gfa.py`** -- Python script; parses a GFA file into an undirected rustworkx graph and reports properties (node/edge counts, component sizes, segment length stats, degree stats, k-mer count stats); samples random simple paths from leaf nodes and reports length statistics (min/max/mean/std dev); `--weight kmer|overlap|unweighted` controls neighbour selection

## Existing tools for reference

These tools solve related problems and can inform design decisions:

- **ART** -- Illumina read simulator with empirical error profiles (C++)
- **wgsim** -- lightweight short-read simulator bundled with samtools
- **InSilicoSeq** -- Python-based Illumina simulator using KDE quality models, designed for metagenomics
- **pbsim3** -- PacBio and ONT long-read simulator using HMM-based error models
- **Badread** -- ONT simulator focused on realistic artefacts (chimeras, junk reads, glitches)
- **Mason2** -- multi-platform simulator supporting VCF-based haplotype simulation
