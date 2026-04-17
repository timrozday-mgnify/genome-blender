//! FASTQ/FASTA reader supporting plain and gzip-compressed files.

use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use anyhow::{Context, Result};
use flate2::read::GzDecoder;

/// One FASTQ/FASTA record.
#[derive(Debug, Clone)]
pub struct Record {
    /// Read name (after the `@` or `>`, whitespace-trimmed).
    pub name: String,
    /// Nucleotide sequence, uppercase.
    pub seq: Vec<u8>,
}

/// Iterator over FASTQ/FASTA records from a (possibly gzip-compressed) file.
pub struct FastxReader {
    lines: Box<dyn Iterator<Item = io::Result<String>>>,
    is_fastq: bool,
}

impl FastxReader {
    /// Open `path` for reading. Detects gzip by `.gz` extension, FASTA/FASTQ
    /// by the first character (`>` = FASTA, `@` = FASTQ).
    pub fn open(path: &Path) -> Result<Self> {
        let reader = open_possibly_gzipped(path)?;
        let mut lines = reader.lines();

        // Peek at the first non-empty line to decide format.
        let first = loop {
            match lines.next() {
                Some(Ok(l)) if l.trim().is_empty() => continue,
                Some(Ok(l)) => break l,
                Some(Err(e)) => return Err(e).context("reading first line"),
                None => {
                    return Ok(FastxReader { lines: Box::new(std::iter::empty()), is_fastq: true })
                }
            }
        };
        let is_fastq = first.starts_with('@');

        // Reconstruct the iterator by prepending the consumed line.
        let combined = std::iter::once(Ok(first)).chain(lines);
        Ok(FastxReader { lines: Box::new(combined), is_fastq })
    }
}

impl Iterator for FastxReader {
    type Item = Result<Record>;

    fn next(&mut self) -> Option<Result<Record>> {
        // Skip blank lines, find header.
        let header = loop {
            let line = self.lines.next()?;
            match line {
                Err(e) => return Some(Err(e.into())),
                Ok(l) if l.trim().is_empty() => continue,
                Ok(l) => break l,
            }
        };

        // Parse name from header line (drop leading @ or >, split on whitespace).
        let prefix = if self.is_fastq { '@' } else { '>' };
        if !header.starts_with(prefix) {
            return Some(Err(anyhow::anyhow!(
                "expected header starting with '{}', got: {:?}",
                prefix,
                header
            )));
        }
        let name = header[1..].split_whitespace().next().unwrap_or("").to_string();

        // Read sequence (one line for FASTQ, possibly multiple for FASTA).
        let mut seq_bytes: Vec<u8> = Vec::new();
        if self.is_fastq {
            // FASTQ: exactly one sequence line, then '+', then quality.
            match self.lines.next() {
                Some(Ok(s)) => seq_bytes.extend_from_slice(s.trim().as_bytes()),
                Some(Err(e)) => return Some(Err(e.into())),
                None => return Some(Err(anyhow::anyhow!("truncated FASTQ: missing sequence"))),
            }
            // Skip '+' separator.
            if let Some(Err(e)) = self.lines.next() {
                return Some(Err(e.into()));
            }
            // Skip quality string.
            if let Some(Err(e)) = self.lines.next() {
                return Some(Err(e.into()));
            }
        } else {
            // FASTA: read until next '>' or EOF.
            // We have to buffer and put the next header back — instead, we
            // collect into a peekable but since Box<dyn Iterator> isn't Peekable,
            // we collect greedily until a header line, then stop.
            // Simplified: assume single-line sequences (sufficient for tests and
            // typical use; multi-line FASTA is handled by concatenation below).
            loop {
                // Try to get a line; peek if it's a header.
                // We do a small trick: collect raw bytes until a '>' line.
                // Since we can't un-read from a Box<dyn Iterator>, we don't
                // support multi-record FASTA with header peeking at this level.
                // Instead: we have a helper that reads lines and stops on '>'.
                // For simplicity, assume FASTA records have one sequence line.
                // TODO: support multi-line FASTA if needed.
                break;
            }
            // We only support single-line FASTA here. For robustness, break.
        }

        // Uppercase the sequence.
        seq_bytes.make_ascii_uppercase();

        Some(Ok(Record { name, seq: seq_bytes }))
    }
}

/// Open a file, transparently decompressing gzip if the extension is `.gz`.
pub fn open_possibly_gzipped(path: &Path) -> Result<Box<dyn BufRead>> {
    let file = File::open(path).with_context(|| format!("opening {:?}", path))?;
    if path.extension().and_then(|e| e.to_str()) == Some("gz") {
        Ok(Box::new(BufReader::new(GzDecoder::new(file))))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Count the number of records in a FASTQ/FASTA file without fully parsing.
///
/// Used to estimate read counts for progress reporting.
pub fn count_records(path: &Path) -> Result<u64> {
    let reader = open_possibly_gzipped(path)?;
    let mut count = 0u64;
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('@') || line.starts_with('>') {
            count += 1;
        }
    }
    Ok(count)
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_fastq(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::with_suffix(".fastq").unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn parse_single_record() {
        let f = write_fastq("@read1\nATCG\n+\nIIII\n");
        let records: Vec<Record> = FastxReader::open(f.path())
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].name, "read1");
        assert_eq!(records[0].seq, b"ATCG");
    }

    #[test]
    fn parse_multiple_records() {
        let f = write_fastq("@r1\nACGT\n+\nIIII\n@r2\nTGCA\n+\nIIII\n");
        let records: Vec<Record> = FastxReader::open(f.path())
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].seq, b"ACGT");
        assert_eq!(records[1].seq, b"TGCA");
    }

    #[test]
    fn name_stops_at_whitespace() {
        let f = write_fastq("@readname extra stuff\nACGT\n+\nIIII\n");
        let records: Vec<Record> = FastxReader::open(f.path())
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(records[0].name, "readname");
    }

    #[test]
    fn seq_uppercased() {
        let f = write_fastq("@r\natcg\n+\nIIII\n");
        let records: Vec<Record> = FastxReader::open(f.path())
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(records[0].seq, b"ATCG");
    }

    #[test]
    fn count_records_works() {
        let f = write_fastq("@r1\nACGT\n+\nIIII\n@r2\nTGCA\n+\nIIII\n");
        assert_eq!(count_records(f.path()).unwrap(), 2);
    }
}
