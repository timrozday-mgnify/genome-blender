//! View: reads index → TSV (read_id \t hex_hashes).

use std::io::{self, BufWriter, Write};
use std::path::Path;

use anyhow::Result;
use lmdb::{Cursor, Transaction};

use crate::build::lmdb::{open_env, MAP_SIZE};

pub fn view_reads(prefix: &Path) -> Result<()> {
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let shard_paths = resolve_shards(prefix, ".index_shard_list")?;
    for shard in &shard_paths {
        let env = open_env(shard, MAP_SIZE, 4)?;
        let db = env.open_db(Some("reads"))?;
        let txn = env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(db)?;
        for (k, v) in cursor.iter_start() {
            let read_id = u32::from_le_bytes(k.try_into().unwrap_or([0; 4]));
            let hashes: Vec<String> = v
                .chunks_exact(8)
                .map(|b| format!("{:016x}", u64::from_le_bytes(b.try_into().unwrap())))
                .collect();
            writeln!(out, "{}\t{}", read_id, hashes.join(" "))?;
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
