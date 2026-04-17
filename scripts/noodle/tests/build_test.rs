//! Integration tests: build indexes from fixture FASTQ files, then verify
//! LMDB contents are correct.

use std::path::PathBuf;
use std::process::Command;

fn noodle_bin() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("target/release/noodle");
    p
}

fn fixtures() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("tests/fixtures");
    p
}

fn open_env(path: &std::path::Path) -> lmdb::Environment {
    lmdb::Environment::new()
        .set_map_size(64 * 1024 * 1024)
        .set_max_dbs(4)
        .open(path)
        .unwrap()
}

fn count_entries(env: &lmdb::Environment, db_name: &str) -> usize {
    use lmdb::{Cursor, Transaction};
    let db = env.open_db(Some(db_name)).unwrap();
    let txn = env.begin_ro_txn().unwrap();
    let mut cursor = txn.open_ro_cursor(db).unwrap();
    cursor.iter_start().count()
}

// ── Single-end ────────────────────────────────────────────────────────────────

#[test]
fn single_end_index_read_count() {
    let dir = tempfile::TempDir::new().unwrap();
    let prefix = dir.path().join("se");

    let status = Command::new(noodle_bin())
        .args([
            "index",
            "--reads", fixtures().join("tiny_R1.fastq").to_str().unwrap(),
            "--prefix", prefix.to_str().unwrap(),
            "--l", "11",
            "--density", "1.0",
            "--minabund", "1",
            "--intra-combo-density", "0.5",
        ])
        .status()
        .expect("failed to run noodle");

    assert!(status.success(), "noodle index failed");

    let env = open_env(&dir.path().join("se.index_shard_0.lmdb"));
    let n = count_entries(&env, "reads");
    assert_eq!(n, 10, "expected 10 reads in single-end index, got {n}");
}

#[test]
fn single_end_read_ids_are_sequential() {
    let dir = tempfile::TempDir::new().unwrap();
    let prefix = dir.path().join("se2");

    Command::new(noodle_bin())
        .args([
            "index",
            "--reads", fixtures().join("tiny_R1.fastq").to_str().unwrap(),
            "--prefix", prefix.to_str().unwrap(),
            "--l", "11",
            "--density", "1.0",
            "--minabund", "1",
            "--intra-combo-density", "0.5",
        ])
        .status().unwrap();

    use lmdb::{Cursor, Transaction};
    let env = open_env(&dir.path().join("se2.index_shard_0.lmdb"));
    let db = env.open_db(Some("reads")).unwrap();
    let txn = env.begin_ro_txn().unwrap();
    let mut cursor = txn.open_ro_cursor(db).unwrap();

    let ids: Vec<u32> = cursor
        .iter_start()
        .map(|(k, _v)| u32::from_le_bytes(k.try_into().unwrap()))
        .collect();
    let expected: Vec<u32> = (1..=10).collect();
    assert_eq!(ids, expected);
}

// ── Paired-end ────────────────────────────────────────────────────────────────

#[test]
fn paired_end_index_read_count() {
    let dir = tempfile::TempDir::new().unwrap();
    let prefix = dir.path().join("pe");

    let status = Command::new(noodle_bin())
        .args([
            "index",
            "--reads", fixtures().join("tiny_R1.fastq").to_str().unwrap(),
            "--reads2", fixtures().join("tiny_R2.fastq").to_str().unwrap(),
            "--prefix", prefix.to_str().unwrap(),
            "--l", "11",
            "--density", "1.0",
            "--minabund", "1",
            "--intra-combo-density", "0.5",
            "--pe-combo-density", "0.5",
            "--combo-max-distance", "100",
        ])
        .status()
        .expect("failed to run noodle");

    assert!(status.success(), "noodle index failed");

    let env = open_env(&dir.path().join("pe.index_shard_0.lmdb"));
    let n = count_entries(&env, "reads");
    assert_eq!(n, 20, "expected 20 reads (10 R1 + 10 R2) in paired-end index");
}

#[test]
fn paired_end_r1_odd_r2_even_ids() {
    let dir = tempfile::TempDir::new().unwrap();
    let prefix = dir.path().join("pe2");

    Command::new(noodle_bin())
        .args([
            "index",
            "--reads", fixtures().join("tiny_R1.fastq").to_str().unwrap(),
            "--reads2", fixtures().join("tiny_R2.fastq").to_str().unwrap(),
            "--prefix", prefix.to_str().unwrap(),
            "--l", "11",
            "--density", "1.0",
            "--minabund", "1",
            "--intra-combo-density", "0.5",
            "--pe-combo-density", "0.5",
            "--combo-max-distance", "100",
        ])
        .status().unwrap();

    use lmdb::{Cursor, Transaction};
    let env = open_env(&dir.path().join("pe2.index_shard_0.lmdb"));
    let db = env.open_db(Some("reads")).unwrap();
    let txn = env.begin_ro_txn().unwrap();
    let mut cursor = txn.open_ro_cursor(db).unwrap();

    let ids: Vec<u32> = cursor
        .iter_start()
        .map(|(k, _v)| u32::from_le_bytes(k.try_into().unwrap()))
        .collect();

    let odd: Vec<u32> = ids.iter().copied().filter(|&id| id % 2 == 1).collect();
    let even: Vec<u32> = ids.iter().copied().filter(|&id| id % 2 == 0).collect();

    let expected_odd: Vec<u32> = (0..10).map(|i| 1 + 2 * i).collect();
    let expected_even: Vec<u32> = (0..10).map(|i| 2 + 2 * i).collect();

    assert_eq!(odd, expected_odd, "R1 reads should have odd IDs 1,3,5,...");
    assert_eq!(even, expected_even, "R2 reads should have even IDs 2,4,6,...");
}

#[test]
fn paired_end_combo_indexes_nonempty() {
    let dir = tempfile::TempDir::new().unwrap();
    let prefix = dir.path().join("pe3");

    Command::new(noodle_bin())
        .args([
            "index",
            "--reads", fixtures().join("tiny_R1.fastq").to_str().unwrap(),
            "--reads2", fixtures().join("tiny_R2.fastq").to_str().unwrap(),
            "--prefix", prefix.to_str().unwrap(),
            "--l", "11",
            "--density", "1.0",
            "--minabund", "1",
            "--intra-combo-density", "1.0",
            "--pe-combo-density", "1.0",
            "--combo-max-distance", "100",
        ])
        .status().unwrap();

    for name in &["pe", "intra"] {
        let env = open_env(&dir.path().join(format!("pe3.{name}_combo_shard_0.lmdb")));
        let n = count_entries(&env, "combo");
        assert!(n > 0, "{name} combo index should be non-empty");
    }
}

#[test]
fn bloom_files_are_created() {
    let dir = tempfile::TempDir::new().unwrap();
    let prefix = dir.path().join("bl");

    Command::new(noodle_bin())
        .args([
            "index",
            "--reads", fixtures().join("tiny_R1.fastq").to_str().unwrap(),
            "--reads2", fixtures().join("tiny_R2.fastq").to_str().unwrap(),
            "--prefix", prefix.to_str().unwrap(),
            "--l", "11",
            "--density", "1.0",
            "--minabund", "1",
            "--intra-combo-density", "0.5",
            "--pe-combo-density", "0.5",
            "--combo-max-distance", "100",
        ])
        .status().unwrap();

    for suffix in &["minimizer_bloom.bin", "pe_combo_bloom.bin", "intra_combo_bloom.bin"] {
        let p = dir.path().join(format!("bl.{suffix}"));
        assert!(p.exists(), "bloom file {suffix} not created");
        assert!(p.metadata().unwrap().len() > 0, "bloom file {suffix} is empty");
    }
}
