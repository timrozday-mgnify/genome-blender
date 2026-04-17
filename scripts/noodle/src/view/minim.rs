//! View: minimizer index → TSV (minimizer_hash \t read_id).

use std::io::{self, BufWriter, Write};
use std::path::Path;

use anyhow::Result;
use lmdb::{Cursor, Transaction};

use crate::build::lmdb::{open_env, MAP_SIZE};

pub fn view_minimizers(prefix: &Path) -> Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let shard_paths = resolve_shards(prefix, ".minimizer_index_shard_list")?;
    for shard in &shard_paths {
        let env = open_env(shard, MAP_SIZE, 4)?;
        let db = env.open_db(Some("minimizers"))?;
        let txn = env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(db)?;
        for inner in cursor.iter_dup_start() {
            for (k, v) in inner {
                let hash = u64::from_le_bytes(k.try_into().unwrap_or([0; 8]));
                let read_id = u32::from_le_bytes(v.try_into().unwrap_or([0; 4]));
                writeln!(out, "{:016x}\t{}", hash, read_id)?;
            }
        }
    }
    Ok(())
}

fn resolve_shards(prefix: &Path, list_suffix: &str) -> Result<Vec<std::path::PathBuf>> {
    let base = prefix
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let list_path = prefix
        .parent()
        .unwrap_or(Path::new("."))
        .join(format!("{base}{list_suffix}"));

    if list_path.exists() {
        let content = std::fs::read_to_string(&list_path)?;
        Ok(content
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| std::path::PathBuf::from(l.trim()))
            .collect())
    } else {
        let single = prefix
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{base}_shard_0.lmdb"));
        if single.exists() { Ok(vec![single]) } else { Ok(vec![]) }
    }
}
