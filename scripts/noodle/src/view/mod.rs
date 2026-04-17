//! `noodle view` subcommand dispatch.

pub mod combo;
pub mod minim;
pub mod reads;

use anyhow::Result;

use crate::cli::{ViewArgs, ViewWhat};

pub fn run_view(args: &ViewArgs) -> Result<()> {
    match &args.what {
        ViewWhat::Reads { prefix } => reads::view_reads(prefix),
        ViewWhat::Minimizers { prefix } => minim::view_minimizers(prefix),
        ViewWhat::PeCombo { prefix } => combo::view_pe_combo(prefix),
        ViewWhat::IntraCombo { prefix } => combo::view_intra_combo(prefix),
    }
}
