//! Index-building orchestration: two-pass FASTQ processing.
//!
//! **Pass 1** (count pass): scan all reads to accumulate per-minimizer
//! occurrence counts needed for `--minabund` filtering.
//!
//! **Pass 2** (write pass): re-scan reads to build:
//!   * Reads index       (`{prefix}.index_shard_*.lmdb`)
//!   * Minimizer index   (`{prefix}.minimizer_index_shard_*.lmdb`)
//!   * Intra-combo index (`{prefix}.intra_combo_shard_*.lmdb`)
//!   * PE combo index    (`{prefix}.pe_combo_shard_*.lmdb`)  [paired-end only]
//!   * Bloom filters     (`{prefix}.minimizer_bloom.bin`, etc.)
//!
//! For paired-end mode: during the write pass, R1 minimizer data is spilled to
//! a temporary file so it can be paired with R2 reads for PE combo computation.

pub mod combo;
pub mod lmdb;
pub mod minim;
pub mod reads;

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use anyhow::{Context, Result};

use crate::bloom::{
    BloomFilter, MAGIC_INTRA_COMBO_BLOOM, MAGIC_MINIMIZER_BLOOM, MAGIC_PE_COMBO_BLOOM,
    DEFAULT_BLOOM_BITS,
};
use crate::cli::IndexArgs;
use crate::fastq::FastxReader;
use crate::minimizers::{extract_minimizers, MinParams};

use combo::{
    intra_combo_hashes, pe_combo_hashes, spill_read, spill_write, ComboIndexWriter,
};
use minim::MinimIndexWriter;
use reads::ReadsIndexWriter;

/// Build all LMDB indexes for the given arguments.
pub fn build_indexes(args: &IndexArgs) -> Result<()> {
    let params = MinParams::new(args.l, args.density);

    eprintln!("Pass 1: counting minimizer occurrences…");
    let mut minim_writer = count_pass(args, params)?;

    eprintln!("Pass 2: writing indexes…");
    write_pass(args, params, &mut minim_writer)?;

    eprintln!("Building bloom filters…");
    build_bloom_filters(args, &minim_writer)?;
    minim_writer.finalize()?;

    Ok(())
}

// ── Pass 1: count ─────────────────────────────────────────────────────────────

fn count_pass(args: &IndexArgs, params: MinParams) -> Result<MinimIndexWriter> {
    let minim_prefix = args.prefix.with_file_name(format!(
        "{}.minimizer_index",
        args.prefix.file_name().unwrap_or_default().to_string_lossy()
    ));
    let mut minim_writer = MinimIndexWriter::new(
        &minim_prefix,
        args.minimizer_batch_size,
        args.minimizer_shard_size,
        args.l as u32,
        args.minabund,
    )?;

    for_each_read(args, |_id, seq| {
        let mins = extract_minimizers(seq, params);
        minim_writer.count(&mins);
        Ok(())
    })?;

    Ok(minim_writer)
}

// ── Pass 2: write ─────────────────────────────────────────────────────────────

fn write_pass(
    args: &IndexArgs,
    params: MinParams,
    minim_writer: &mut MinimIndexWriter,
) -> Result<()> {
    let reads_prefix = args.prefix.with_file_name(format!(
        "{}.index",
        args.prefix.file_name().unwrap_or_default().to_string_lossy()
    ));
    let intra_prefix = args.prefix.with_file_name(format!(
        "{}.intra_combo",
        args.prefix.file_name().unwrap_or_default().to_string_lossy()
    ));
    let pe_prefix = args.prefix.with_file_name(format!(
        "{}.pe_combo",
        args.prefix.file_name().unwrap_or_default().to_string_lossy()
    ));

    let mut reads_writer = ReadsIndexWriter::new(
        &reads_prefix,
        args.reads_batch_size,
        args.reads_shard_size,
        args.l as u32,
    )?;
    let mut intra_writer = ComboIndexWriter::new(
        &intra_prefix,
        args.combo_batch_size,
        args.combo_shard_size,
        args.l as u64,
        args.intra_combo_density,
    )?;
    let mut pe_writer = ComboIndexWriter::new(
        &pe_prefix,
        args.combo_batch_size,
        args.combo_shard_size,
        args.l as u64,
        args.pe_combo_density,
    )?;

    if let Some(r2_path) = &args.reads2 {
        // Paired-end: two-file mode.
        // Sub-pass 2a: R1 reads (odd IDs 1, 3, 5, …) + spill R1 minimizers.
        let spill_path = args.prefix.with_extension("r1_spill.bin");
        {
            let spill_file = File::create(&spill_path)
                .with_context(|| format!("creating spill file {:?}", spill_path))?;
            let mut spill_writer = BufWriter::new(spill_file);
            let mut read_id = 1u32;

            for record in FastxReader::open(&args.reads)? {
                let record = record?;
                let mins = extract_minimizers(&record.seq, params);

                reads_writer.push(read_id, &mins)?;
                minim_writer.push(read_id, &mins)?;

                let intra = intra_combo_hashes(&mins, args.intra_combo_density);
                intra_writer.push_hashes(&intra)?;

                spill_write(&mut spill_writer, &mins)
                    .context("writing to R1 spill file")?;

                read_id += 2; // R1 gets odd IDs
            }
        }

        // Sub-pass 2b: R2 reads (even IDs 2, 4, 6, …).
        {
            let spill_file = File::open(&spill_path)
                .with_context(|| format!("opening spill file {:?}", spill_path))?;
            let mut spill_reader = BufReader::new(spill_file);
            let mut read_id = 2u32;

            for record in FastxReader::open(r2_path)? {
                let record = record?;
                let r2_mins = extract_minimizers(&record.seq, params);

                reads_writer.push(read_id, &r2_mins)?;
                minim_writer.push(read_id, &r2_mins)?;

                let intra = intra_combo_hashes(&r2_mins, args.intra_combo_density);
                intra_writer.push_hashes(&intra)?;

                // Read paired R1 minimizers from spill.
                let r1_mins = spill_read(&mut spill_reader)
                    .context("reading from R1 spill file")?;
                let pe = pe_combo_hashes(
                    &r1_mins,
                    &r2_mins,
                    args.pe_combo_density,
                    args.combo_max_distance,
                );
                pe_writer.push_hashes(&pe)?;

                read_id += 2; // R2 gets even IDs
            }
        }

        // Remove spill file.
        let _ = std::fs::remove_file(&spill_path);
    } else {
        // Single-end: sequential IDs starting at 1.
        let mut read_id = 1u32;
        for record in FastxReader::open(&args.reads)? {
            let record = record?;
            let mins = extract_minimizers(&record.seq, params);

            reads_writer.push(read_id, &mins)?;
            minim_writer.push(read_id, &mins)?;

            let intra = intra_combo_hashes(&mins, args.intra_combo_density);
            intra_writer.push_hashes(&intra)?;

            read_id += 1;
        }
    }

    reads_writer.finalize()?;
    // minim_writer is finalized by the caller after bloom-filter construction.
    intra_writer.finalize()?;
    pe_writer.finalize()?;
    Ok(())
}

// ── Bloom filter construction ─────────────────────────────────────────────────

fn build_bloom_filters(args: &IndexArgs, minim_writer: &MinimIndexWriter) -> Result<()> {
    // Minimizer bloom: insert every hash that passed minabund.
    let mut min_bloom = BloomFilter::new(DEFAULT_BLOOM_BITS);
    for (&hash, &count) in minim_writer.counts() {
        if count >= args.minabund {
            min_bloom.insert_u64(hash);
        }
    }
    let min_bloom_path = args.prefix.with_file_name(format!(
        "{}.minimizer_bloom.bin",
        args.prefix.file_name().unwrap_or_default().to_string_lossy()
    ));
    min_bloom.write_to_file(&min_bloom_path, MAGIC_MINIMIZER_BLOOM, args.l as u64)?;

    // PE combo bloom and intra-combo bloom are built from the combo LMDB
    // entries that were accepted (density-threshold filtering was already
    // applied in the ComboIndexWriter).  We rebuild from the LMDB to avoid
    // holding all hashes in memory a second time.
    write_combo_bloom(
        args,
        ".pe_combo_shard_list",
        ".pe_combo_bloom.bin",
        MAGIC_PE_COMBO_BLOOM,
        args.pe_combo_density,
    )?;
    write_combo_bloom(
        args,
        ".intra_combo_shard_list",
        ".intra_combo_bloom.bin",
        MAGIC_INTRA_COMBO_BLOOM,
        args.intra_combo_density,
    )?;
    Ok(())
}

fn write_combo_bloom(
    args: &IndexArgs,
    shard_list_suffix: &str,
    bloom_file_suffix: &str,
    magic: &[u8; 8],
    density: f64,
) -> Result<()> {
    use ::lmdb::{Cursor, Transaction};

    let mut bloom = BloomFilter::new(DEFAULT_BLOOM_BITS);
    let prefix = &args.prefix;

    let base = prefix
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();

    // Resolve shard paths.
    let list_path = prefix
        .parent()
        .unwrap_or(Path::new("."))
        .join(format!("{base}{shard_list_suffix}"));
    let shard_paths: Vec<_> = if list_path.exists() {
        std::fs::read_to_string(&list_path)?
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| std::path::PathBuf::from(l.trim()))
            .collect()
    } else {
        // Fall back to single monolithic file.
        let single = prefix
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{base}_shard_0.lmdb"));
        if single.exists() { vec![single] } else { vec![] }
    };

    for shard in shard_paths {
        let env = lmdb::open_env(&shard, lmdb::MAP_SIZE, 4)?;
        let db = env.open_db(Some("combo"))?;
        let txn = env.begin_ro_txn()?;
        let mut cursor = txn.open_ro_cursor(db)?;
        for (k, _v) in cursor.iter() {
            bloom.insert_raw(k);
        }
    }

    let bloom_path = prefix
        .parent()
        .unwrap_or(Path::new("."))
        .join(format!("{base}{bloom_file_suffix}"));
    bloom.write_to_file(
        &bloom_path,
        magic,
        density.to_bits(),
    )
}

// ── Helper: iterate all reads ─────────────────────────────────────────────────

fn for_each_read<F>(args: &IndexArgs, mut f: F) -> Result<()>
where
    F: FnMut(u32, &[u8]) -> Result<()>,
{
    let mut id = 1u32;
    for record in FastxReader::open(&args.reads)? {
        let record = record?;
        f(id, &record.seq)?;
        id += 1;
    }
    if let Some(r2) = &args.reads2 {
        let mut id2 = 2u32;
        for record in FastxReader::open(r2)? {
            let record = record?;
            f(id2, &record.seq)?;
            id2 += 2;
        }
    }
    Ok(())
}
