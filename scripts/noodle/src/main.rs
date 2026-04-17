mod bloom;
mod build;
mod cli;
mod fastq;
mod minimizers;
mod nthash;
mod view;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Command};

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Index(args) => build::build_indexes(&args),
        Command::View(args) => view::run_view(&args),
    }
}
