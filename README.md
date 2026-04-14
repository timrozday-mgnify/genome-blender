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

# --- Error rate scaling ----------------------------------------------
error_rate_scale: 1.0  # multiplier on error probabilities; <1 fewer errors, >1 more

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

The `scripts/` directory contains ready-to-run shell scripts and Python analysis scripts. Shell scripts execute each tool via Docker against simulated FASTQ files; all read paired-end input from `../genome-blender_run/single_short_shallow/output/` and write output to a tool-specific subdirectory alongside the input.

#### `run_cuttlefish3.sh` — Compacted de Bruijn graph (unitigs)

Runs [cuttlefish3](https://github.com/COMBINE-lab/cuttlefish/tree/cuttlefish3) to build a compacted de Bruijn graph from reads, outputting unitigs as FASTA.

| Variable | Flag | Default | Description |
|----------|------|---------|-------------|
| `K` | `-k` | `31` | K-mer length (must be odd, max 63) |
| `MIN_LEN` | `--min-len` | `12` | Minimizer length |
| `CUTOFF` | `-c` | `1` | Minimum (k+1)-mer frequency cutoff (default for reads is 2) |
| `INPUT_MODE` | `--read`/`--ref` | `--read` | `--read` for FASTQ input, `--ref` for FASTA reference input |
| `COLOR` | `--color` | `false` | Color the compacted de Bruijn graph |
| — | `-w` | `/tmp/cf3` | Working directory (uses a Docker volume to avoid macOS filesystem issues) |

Input files are passed as `--seq` arguments. Alternatives: `--list` for a text file of paths, `--dir` for a directory.

#### `run_bifrost.sh` — Compacted coloured de Bruijn graph (GFA)

Runs [Bifrost](https://github.com/pmelsted/bifrost) to build a compacted de Bruijn graph, outputting GFA directly.

| Variable | Flag | Default | Description |
|----------|------|---------|-------------|
| `K` | `-k` | `31` | K-mer length |
| `THREADS` | `-t` | `4` | Number of threads |
| `BLOOM_BITS` | `-B` | `24` | Bloom filter bits per k-mer |
| `CLIP_TIPS` | `-i` | `true` | Clip tips shorter than k k-mers |
| `DEL_ISOLATED` | `-d` | `true` | Delete isolated contigs shorter than k k-mers |
| `COLORS` | `-c` | `false` | Color the compacted de Bruijn graph |
| `FASTA_OUT` | `-f` | `false` | Output FASTA instead of GFA |
| `BFG_OUT` | `-b` | `false` | Output bfg/bfi format instead of GFA |
| `VERBOSE` | `-v` | `false` | Print info messages during execution |
| `NO_COMPRESS` | `-n` | `false` | Do not compress output files |
| `NO_INDEX` | `-N` | `false` | Do not make index file |

Input files are passed as `-s` (sequencing reads; k-mers with exactly 1 occurrence are filtered). Use `-r` instead for reference files (all k-mers kept).

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

Runs [SPAdes](https://github.com/ablab/spades) in metagenomic mode with default graph simplification. SPAdes natively outputs `assembly_graph.gfa` alongside contigs.

| Argument | Value | Description |
|----------|-------|-------------|
| `--meta` | — | Metagenomic assembly mode |
| `-k` | `21,31,41,51` | K-mer sizes |
| `-1`, `-2` | R1, R2 FASTQs | Paired-end input files |
| `-o` | output path | Output directory |
| `-t` | `4` | Number of threads |

#### `run_spades_raw.sh` — Metagenomic assembly with simplification disabled (GFA)

Runs [SPAdes](https://github.com/ablab/spades) in metagenomic mode with all graph simplification disabled (no tip clipping, bubble removal, erroneous-connection removal, or isolated-edge removal). This produces a raw de Bruijn graph. Uses the hidden `--configs-dir` flag to supply a patched `simplification.info` and strips the `simp`/`preliminary_simp` overrides from `meta_mode.info`.

| Argument | Value | Description |
|----------|-------|-------------|
| `--meta` | — | Metagenomic assembly mode |
| `--only-assembler` | — | Skip read error correction |
| `--disable-rr` | — | Skip repeat resolution |
| `--configs-dir` | custom path | Use patched configs with simplification disabled |
| `-k` | `31` | K-mer sizes (configurable via `K_LIST`) |
| `-1`, `-2` | R1, R2 FASTQs | Paired-end input files |
| `-o` | output path | Output directory |
| `-t` | `4` | Number of threads |

On first run the script extracts the default SPAdes configs from the container, replaces `simplification.info` with a version where all simplification steps use no-op conditions or are disabled, and removes the `simp` and `preliminary_simp` blocks from `meta_mode.info` (which would otherwise re-enable simplification). The patched configs are cached in the output directory.

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

#### `asf_sample.py` — Assembly-free minimizer path sampling

Samples paths through the implicit minimizer-space de Bruijn graph without constructing an assembly, using only the LMDB indexes produced by rust-mdbg. Starting from a random read, it extends a path forward and backward by finding unambiguous (single-supported) successor/predecessor minimizers until the path reaches a branch, a dead end, or the length cap. Also estimates the insert size distribution of the input library in both minimizer space and basepair space.

```bash
python scripts/asf_sample.py rust_mdbg_out \
    --n-paths 200 \
    --max-path-mers 5000 \
    --min-support 2 \
    --insert-size-inference nuts \
    --output paths.jsonl
```

Output is a JSONL file where each line is a JSON object with keys `minimizer_ids` (ordered list of u64 hashes), `distances` (minimizer-count distances between consecutive nodes), and `support` (read-vote counts for each extension step).

---

#### `reconstruct_sequences.py` — Basepair sequence reconstruction from minimizer paths

Translates sampled minimizer paths back into basepair sequences by mapping the original paired-end reads onto each path in minimizer space, scoring the alignments by a composite likelihood, extracting inter-minimizer gap sequences from high-scoring reads, and stitching everything together.

This is necessary because the path sampling step (`asf_sample.py`) and the LMDB indexes it uses store only minimizer hashes — no basepair information. The original FASTQ files are the sole source of basepair sequence.

```bash
python scripts/reconstruct_sequences.py paths.jsonl \
    --prefix rust_mdbg_out \
    --reads R1.fq.gz --reads R2.fq.gz \
    --insert-size-json insert_estimate.json \
    --mode best \
    --output reconstructed.fa
```

| Option | Default | Description |
|--------|---------|-------------|
| `paths_file` (positional) | — | JSONL paths file from `asf_sample.py` |
| `--prefix` | — | rust-mdbg output prefix (locates `.read_minimizers` files and `minimizer_table`) |
| `--reads` | — | FASTQ/FASTA input; repeat twice for paired-end (R1 then R2) |
| `--output` / `-o` | `reconstructed.fa` | Output FASTA |
| `--mode` | `best` | Gap sequence selection strategy: `best`, `random`, `common`, or `consensus` |
| `--insert-size-json` | — | JSON from `asf_sample.py` with `mu_bp`/`sigma_bp` or `median_bp`/`sigma_bp` keys |
| `--insert-size-mean` / `--insert-size-std` | — | Explicit insert size in bp (alternative to `--insert-size-json`) |
| `--min-anchors` | `3` | Minimum chained minimizer hits to keep a read alignment |
| `--max-reads-per-minimizer` | `200` | Cap on candidate reads examined per path minimizer |
| `--min-coverage-fraction` | `0.5` | Skip paths where fewer than this fraction of minimizer positions have read coverage |
| `--weight-coverage` / `--no-weight-coverage` | on | Include anchor-count in alignment scoring |
| `--weight-insert` / `--no-weight-insert` | on | Include insert-size log-likelihood in scoring |
| `--weight-mate` / `--no-weight-mate` | on | Add mate-concordance bonus to scoring |
| `--gap-fill-char` | `N` | Character used to fill uncovered gaps |
| `--seed` | `42` | Random seed for reproducibility |

##### Algorithm

The pipeline has six stages for each path:

1. **Seed.** For each minimizer in the path, look up candidate reads from an in-memory reverse index built by scanning the `.read_minimizers` files. Only reads sharing at least one minimizer with any path are retained in memory, keeping memory usage proportional to the subset of reads that is actually relevant.

2. **Chain.** For each candidate read, a linear-chaining DP (analogous to the seed-and-chain step in minimap2) finds the longest set of consistently placed anchor hits in minimizer space. Anchors are (path position, read minimizer index) pairs where the hashes match. A chain is valid when path offsets and read offsets agree within a configurable gap tolerance. Both strands are tried; the longer chain wins and the strand is recorded. Chains shorter than `--min-anchors` are discarded.

3. **Score.** Each alignment receives a composite score (see below).

4. **Extract.** For each pair of consecutive path minimizers (*i* → *i*+1), every alignment whose chain covers both positions contributes a candidate gap sequence: the raw read bytes between the end of the l-mer at position *i* and the start of the l-mer at position *i*+1 (reverse-complemented for minus-strand alignments). The known l-mer length and the per-read minimizer bp positions from the `.read_minimizers` files provide exact extraction coordinates.

5. **Choose.** One gap sequence is selected per span according to `--mode`:
   - **`best`** — the gap sequence from the highest-scoring alignment.
   - **`random`** — one gap sequence sampled at random, optionally weighted by alignment score (softmax-style) when `--weight-coverage` is set.
   - **`common`** — the most frequently observed exact gap sequence across all covering reads; ties broken by highest alignment score.
   - **`consensus`** — all candidate gap sequences are submitted to `abpoa` (preferred) or `spoa` for partial-order alignment; the resulting consensus is used. Requires the aligner to be on `PATH`; raises an error otherwise.

6. **Stitch.** The final basepair sequence is assembled as:

   ```
   lmer(m₀) + gap(m₀→m₁) + lmer(m₁) + gap(m₁→m₂) + … + lmer(mₙ)
   ```

   L-mer sequences are extracted from covering reads (in the correct strand orientation). For positions with no covering read, the l-mer falls back to the `minimizer_table` lookup (applying reverse complement when the majority of reads aligned on the minus strand). Uncovered gaps are filled with `--gap-fill-char` characters repeated for an estimated gap length derived from the minimizer-count distances stored in the path.

##### Read likelihood scoring

**Theory.** The goal is to find, for each inter-minimizer gap, the basepair sequence most likely to have produced the observed read data. Rather than computing a full probabilistic alignment, the script uses a tractable composite score that captures three independent sources of evidence:

1. **Coverage (anchor count).** The probability that a read genuinely originates from a particular path region grows with the number of its minimizers that form a consistent chain with the path. If a read has *n* minimizers, and *k* of them chain consistently to the path, this is analogous to *k* independent observations confirming the placement. The score contribution is proportional to *k*.

2. **Insert size log-likelihood.** For paired-end reads, the original DNA fragment spans both mates. The fragment length distribution was estimated from the data by `asf_sample.py` and is modelled as a log-normal:

   $$P(\text{insert size} = x) = \frac{1}{x \, \sigma \sqrt{2\pi}} \exp\!\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)$$

   The parameters $\mu$ and $\sigma$ are the log-scale mean and standard deviation inferred by `asf_sample.py` (keys `mu_bp`/`sigma_bp` or `median_bp`/`sigma_bp` in its output JSON). The insert size for a given PE alignment is estimated from the path span covered by the pair (in minimizer units) converted to basepairs using the local bp-per-minimizer ratio estimated from each mate's own anchor positions. The log-probability $\log P(\text{insert size})$ is added to the score (weighted by `--weight-insert`, default 0.5).

3. **Mate concordance.** Both mates of a pair independently search for a chain to the path. When both succeed, there are two orthogonal alignment signals confirming the read originates from this path region. This is captured by a binary bonus (default weight 2.0) added when `mate_concordant = True`.

**Implementation.** The composite score is:

```
score = w_cov × n_anchors
      + w_insert × log P(insert_size | LogNormal(μ, σ))
      + w_mate × concordant
```

where `w_cov = 1.0`, `w_insert = 0.5`, `w_mate = 2.0` by default (all adjustable). The insert size in basepairs is computed as:

```
path_span_bp ≈ (outer_path_end − outer_path_start) × bp_per_minimizer
```

where `bp_per_minimizer` is the mean of the local estimates from R1 and R2:

```
bp_per_minimizer_R1 = (positions[last_anchor_R1] + lmer_len − positions[first_anchor_R1])
                      / (len(R1_chain) − 1)
```

This avoids needing a global density estimate and adapts to local variation in minimizer spacing along the path.

---

#### `parse_gfa.py` — GFA graph analysis

Parses a GFA file into an undirected graph using [rustworkx](https://www.rustworkx.org/) and reports structural properties. Segments become nodes, links become edges.

```bash
python scripts/parse_gfa.py assembly.gfa
python scripts/parse_gfa.py assembly.gfa --samples 5000
python scripts/parse_gfa.py assembly.gfa --weight overlap
python scripts/parse_gfa.py assembly.gfa --no-sample
```

| Option | Default | Description |
|--------|---------|-------------|
| `-n`/`--samples` | `1000` | Number of random paths to sample |
| `--weight` | `kmer` | Neighbour-selection weighting: `kmer` (k-mer count tag), `overlap` (CIGAR overlap length), `unweighted` (uniform) |
| `--no-sample` | off | Skip path sampling entirely |
| `-v`/`--verbose` | off | Enable debug logging |

**Summary output** includes node/edge counts, connected component sizes, segment length statistics (min/max/mean), node degree statistics, k-mer count statistics (when the `KC` tag is present), and sampled path length statistics (min/max/mean/std dev/variance).

**Path sampling** performs random simple walks starting from leaf nodes (degree 1), so each walk travels from one graph boundary to another. If the graph has no leaves (e.g. a pure cycle), all nodes are used as start points. Walk neighbours are selected according to `--weight`.

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

Core simulation pipeline (`genome_blender` package):
- Python 3.10+
- PyTorch
- Pyro-PPL
- pysam
- Biopython
- Typer
- PyYAML

Analysis scripts (`scripts/`):
- rustworkx
- Typer
- rich

## License

TBD
