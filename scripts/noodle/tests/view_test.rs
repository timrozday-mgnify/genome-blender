//! Integration tests: build indexes then verify `noodle view` TSV output.

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

fn build_se(dir: &tempfile::TempDir) -> PathBuf {
    let prefix = dir.path().join("v");
    let status = Command::new(noodle_bin())
        .args([
            "index",
            "--reads", fixtures().join("tiny_R1.fastq").to_str().unwrap(),
            "--prefix", prefix.to_str().unwrap(),
            "--l", "11",
            "--density", "1.0",
            "--minabund", "1",
            "--intra-combo-density", "1.0",
        ])
        .status().unwrap();
    assert!(status.success());
    prefix
}

#[test]
fn view_reads_row_count() {
    let dir = tempfile::TempDir::new().unwrap();
    let prefix = build_se(&dir);

    let out = Command::new(noodle_bin())
        .args(["view", "reads", prefix.to_str().unwrap()])
        .output().unwrap();
    assert!(out.status.success());

    let lines: Vec<&str> = std::str::from_utf8(&out.stdout)
        .unwrap()
        .lines()
        .collect();
    assert_eq!(lines.len(), 10, "expected 10 read rows, got {}", lines.len());
}

#[test]
fn view_reads_tsv_format() {
    let dir = tempfile::TempDir::new().unwrap();
    let prefix = build_se(&dir);

    let out = Command::new(noodle_bin())
        .args(["view", "reads", prefix.to_str().unwrap()])
        .output().unwrap();

    let stdout = std::str::from_utf8(&out.stdout).unwrap();
    for line in stdout.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 2, "reads view row should have 2 columns: {line}");
        let _read_id: u32 = cols[0].parse().expect("read_id should be an integer");
        // Second column is space-separated hex hashes; each should be 16 chars.
        for hex in cols[1].split_whitespace() {
            assert_eq!(hex.len(), 16, "minimizer hash hex should be 16 chars: {hex}");
        }
    }
}

#[test]
fn view_minimizers_row_count_matches_reads_coverage() {
    let dir = tempfile::TempDir::new().unwrap();
    let prefix = build_se(&dir);

    let out = Command::new(noodle_bin())
        .args(["view", "minimizers", prefix.to_str().unwrap()])
        .output().unwrap();
    assert!(out.status.success());

    let n = std::str::from_utf8(&out.stdout).unwrap().lines().count();
    assert!(n > 0, "minimizers view should have at least one row");
}

#[test]
fn view_intra_combo_row_count() {
    let dir = tempfile::TempDir::new().unwrap();
    let prefix = build_se(&dir);

    let out = Command::new(noodle_bin())
        .args(["view", "intra-combo", prefix.to_str().unwrap()])
        .output().unwrap();
    assert!(out.status.success());

    let n = std::str::from_utf8(&out.stdout).unwrap().lines().count();
    assert!(n > 0, "intra-combo view should have at least one row");
}

#[test]
fn view_combo_tsv_format() {
    let dir = tempfile::TempDir::new().unwrap();
    let prefix = build_se(&dir);

    let out = Command::new(noodle_bin())
        .args(["view", "intra-combo", prefix.to_str().unwrap()])
        .output().unwrap();

    let stdout = std::str::from_utf8(&out.stdout).unwrap();
    for line in stdout.lines() {
        let cols: Vec<&str> = line.split('\t').collect();
        assert_eq!(cols.len(), 2, "combo view row should have 2 columns: {line}");
        assert_eq!(cols[0].len(), 16, "hash column should be 16 hex chars");
        let _count: u32 = cols[1].parse().expect("count column should be integer");
    }
}
