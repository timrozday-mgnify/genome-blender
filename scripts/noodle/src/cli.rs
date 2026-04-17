//! CLI definitions for `noodle index` and `noodle view` subcommands.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

// ── Top-level CLI ─────────────────────────────────────────────────────────────

#[derive(Debug, Parser)]
#[command(name = "noodle", about = "Minimizer-space LMDB index builder and viewer")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Build LMDB indexes from FASTQ/FASTA reads.
    Index(IndexArgs),
    /// Print index contents to stdout as TSV.
    View(ViewArgs),
}

// ── `noodle index` ────────────────────────────────────────────────────────────

#[derive(Debug, Args)]
pub struct IndexArgs {
    /// Input FASTQ/FASTA file (gzip accepted).
    #[arg(long)]
    pub reads: PathBuf,

    /// Optional R2 paired-end file (gzip accepted).
    #[arg(long)]
    pub reads2: Option<PathBuf>,

    /// l-mer length for minimizer extraction.
    #[arg(short, long, default_value_t = 17)]
    pub l: usize,

    /// Minimizer selection density (fraction of l-mers kept).
    #[arg(long, default_value_t = 0.1)]
    pub density: f64,

    /// Minimum occurrence count for minimizer to be included in the index.
    #[arg(long, default_value_t = 2)]
    pub minabund: u32,

    /// Output file prefix (directory must exist).
    #[arg(long)]
    pub prefix: PathBuf,

    /// PE combo hash retention fraction.
    #[arg(long, default_value_t = 0.05)]
    pub pe_combo_density: f64,

    /// Intra-read combo hash retention fraction.
    #[arg(long, default_value_t = 0.05)]
    pub intra_combo_density: f64,

    /// Maximum R1/R2 position distance for PE combo inclusion.
    #[arg(long, default_value_t = 500)]
    pub combo_max_distance: u32,

    /// Reads index shard size in reads (0 = single shard).
    #[arg(long, default_value_t = 0)]
    pub reads_shard_size: u64,

    /// Minimizer index shard size in reads (0 = single shard).
    #[arg(long, default_value_t = 0)]
    pub minimizer_shard_size: u64,

    /// Combo index shard size in reads (0 = single shard).
    #[arg(long, default_value_t = 0)]
    pub combo_shard_size: u64,

    /// In-memory batch size before reads LMDB flush.
    #[arg(long, default_value_t = 100_000)]
    pub reads_batch_size: usize,

    /// In-memory batch size before minimizer LMDB flush.
    #[arg(long, default_value_t = 500_000)]
    pub minimizer_batch_size: usize,

    /// In-memory batch size before combo LMDB flush.
    #[arg(long, default_value_t = 500_000)]
    pub combo_batch_size: usize,
}

// ── `noodle view` ─────────────────────────────────────────────────────────────

#[derive(Debug, Args)]
pub struct ViewArgs {
    #[command(subcommand)]
    pub what: ViewWhat,
}

#[derive(Debug, Subcommand)]
pub enum ViewWhat {
    /// Print reads index as TSV: read_id<TAB>hex_hashes (space-separated).
    Reads {
        /// Index prefix (same value passed to `noodle index --prefix`).
        prefix: PathBuf,
    },
    /// Print minimizer index as TSV: minimizer_hash<TAB>read_id.
    Minimizers {
        prefix: PathBuf,
    },
    /// Print PE combo index as TSV: combo_hash<TAB>count.
    PeCombo {
        prefix: PathBuf,
    },
    /// Print intra-read combo index as TSV: combo_hash<TAB>count.
    IntraCombo {
        prefix: PathBuf,
    },
}
