"""Sequencing error application via the Skiver ``generate`` subprocess.

genome-blender no longer implements error models itself.  All error
application is delegated to Skiver's generative interface (``skiver-generate``):
genome-blender prepares reads upstream, hands a chunk of sequences to the
subprocess, and reads back observed bases, Phred quality, and a CIGAR for the
ground-truth BAM.

The model is selected one of three ways (see :class:`SkiverModelConfig`):

* a named platform **preset** (``illumina`` → ``hq-illumina``, ``pacbio`` →
  ``pacbio``, ``nanopore`` → ``ont``),
* a trained **model** ``.pt`` artifact, or
* a **component string** plus a parameter file.

Performance: one subprocess call per chunk amortises interpreter / model-load
startup over the whole chunk, and Skiver samples each read with a vectorised
hot loop, so there is no per-read process overhead.
"""

from __future__ import annotations

import logging
import re
import shlex
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from genome_blender.models import ErrorModel, Read, ReadBatch

logger = logging.getLogger(__name__)

# ErrorModel enum value → Skiver preset name.
_PRESET_MAP: dict[str, str] = {
    "illumina": "hq-illumina",
    "pacbio": "pacbio",
    "nanopore": "ont",
}

# CIGAR op character → BAM/pysam op code.
_CIGAR_OP: dict[str, int] = {"M": 0, "I": 1, "D": 2}
_CIGAR_RE = re.compile(r"(\d+)([MID])")
_DEFAULT_GENERATE_CMD = "skiver-generate"


@dataclass
class SkiverModelConfig:
    """How to invoke ``skiver-generate`` for a run.

    Exactly one of ``preset`` / ``model_path`` / (``components`` + ``params_path``)
    is used (resolved in that precedence by :func:`resolve_skiver_model`).
    """

    preset: str | None = None
    model_path: Path | None = None
    components: str | None = None
    params_path: Path | None = None
    use_vi: bool = False
    phred_calibration: Path | None = None
    max_ins_run: int = 10
    error_rate_scale: float = 1.0
    generate_cmd: list[str] = field(
        default_factory=lambda: [_DEFAULT_GENERATE_CMD],
    )

    def build_command(self, *, input_path: Path, paired: bool, seed: int) -> list[str]:
        """Assemble the full ``skiver-generate`` argv for a chunk."""
        cmd = list(self.generate_cmd)
        if self.model_path is not None:
            cmd += ["--model", str(self.model_path)]
        elif self.preset is not None:
            cmd += ["--preset", self.preset]
        else:
            cmd += ["--components", str(self.components),
                    "--params", str(self.params_path)]
        if self.use_vi:
            cmd.append("--use-vi")
        cmd += ["--input", str(input_path), "--seed", str(seed),
                "--max-ins-run", str(self.max_ins_run)]
        if paired:
            cmd.append("--paired")
        if self.error_rate_scale != 1.0:
            cmd += ["--error-rate-scale", str(self.error_rate_scale)]
        if self.phred_calibration is not None:
            cmd += ["--phred-calibration", str(self.phred_calibration)]
        return cmd


def resolve_skiver_model(
    *,
    error_model: ErrorModel,
    skiver_model: Path | None,
    skiver_components: str | None,
    skiver_params: Path | None,
    skiver_use_vi: bool,
    skiver_phred_calibration: Path | None,
    skiver_max_ins_run: int,
    error_rate_scale: float,
    skiver_generate_cmd: str,
) -> SkiverModelConfig | None:
    """Resolve CLI/config inputs to a model config, or ``None`` to skip errors.

    Precedence: an explicit ``skiver_model`` or ``skiver_components`` overrides
    the ``error_model`` preset.  ``error_model == none`` with no explicit Skiver
    model means no error application.
    """
    cmd = shlex.split(skiver_generate_cmd) or [_DEFAULT_GENERATE_CMD]

    def _make(
        *,
        preset: str | None = None,
        model_path: Path | None = None,
        components: str | None = None,
        params_path: Path | None = None,
    ) -> SkiverModelConfig:
        return SkiverModelConfig(
            preset=preset,
            model_path=model_path,
            components=components,
            params_path=params_path,
            use_vi=skiver_use_vi,
            phred_calibration=skiver_phred_calibration,
            max_ins_run=skiver_max_ins_run,
            error_rate_scale=error_rate_scale,
            generate_cmd=cmd,
        )

    if skiver_model is not None:
        logger.info("Skiver model artifact: %s", skiver_model)
        return _make(model_path=skiver_model)

    if skiver_components is not None:
        if skiver_params is None:
            raise ValueError("skiver_components requires skiver_params")
        logger.info("Skiver model from components: %s", skiver_components)
        return _make(components=skiver_components, params_path=skiver_params)

    if error_model.value == "none":
        return None

    preset = _PRESET_MAP[error_model.value]
    logger.info("Skiver preset: %s (%s)", preset, error_model.value)
    return _make(preset=preset)


# ── Subprocess application ───────────────────────────────────────────────────────


def _flatten(read_batch: ReadBatch) -> list[Read]:
    """Flatten a batch to an ordered read list (pairs as r1, r2, r1, r2, …)."""
    if read_batch.is_paired:
        assert read_batch.paired is not None
        flat: list[Read] = []
        for r1, r2 in read_batch.paired:
            flat.append(r1)
            flat.append(r2)
        return flat
    assert read_batch.single is not None
    return list(read_batch.single)


def _parse_cigar(text: str) -> list[tuple[int, int]]:
    """Parse a CIGAR string (e.g. ``30M1I19M``) to ``[(op, length), …]``."""
    return [(_CIGAR_OP[op], int(n)) for n, op in _CIGAR_RE.findall(text)]


def _write_fasta(flat: list[Read], paired: bool, handle) -> None:
    """Write each read as a FASTA record named by index (and mate for pairs)."""
    for i, read in enumerate(flat):
        if paired:
            name = f"{i}/1" if i % 2 == 0 else f"{i}/2"
        else:
            name = str(i)
        handle.write(f">{name}\n{read.sequence}\n")


def _iter_fastq_records(stream):
    """Yield (sequence, quality, header) from a FASTQ text stream."""
    while True:
        header = stream.readline()
        if not header:
            return
        seq = stream.readline().rstrip("\n")
        stream.readline()                 # '+'
        qual = stream.readline().rstrip("\n")
        yield seq, qual, header.rstrip("\n")


def apply_error_model(
    read_batch: ReadBatch,
    model_cfg: SkiverModelConfig | None,
    seed: int,
) -> ReadBatch:
    """Apply a Skiver error model to a chunk of reads via subprocess.

    Args:
        read_batch: Reads prepared upstream (single-end or paired-end).
        model_cfg: Resolved model config, or ``None`` to leave reads unchanged.
        seed: Deterministic seed for this chunk.

    Returns:
        A new ``ReadBatch`` with observed sequences, quality, and CIGARs.
    """
    if model_cfg is None:
        return read_batch

    paired = read_batch.is_paired
    flat = _flatten(read_batch)
    if not flat:
        return read_batch

    with tempfile.NamedTemporaryFile(
        "w", suffix=".fasta", delete=False,
    ) as tf:
        _write_fasta(flat, paired, tf)
        input_path = Path(tf.name)

    cmd = model_cfg.build_command(input_path=input_path, paired=paired, seed=seed)
    logger.debug("Running: %s", " ".join(cmd))
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        assert proc.stdout is not None
        new_flat: list[Read] = []
        records = _iter_fastq_records(proc.stdout)
        for src in flat:
            try:
                seq, qual, header = next(records)
            except StopIteration:
                proc.wait()
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(
                    f"skiver-generate returned fewer reads than expected; "
                    f"stderr:\n{stderr}"
                )
            _, _, cigar_field = header.partition("cigar:")
            new_flat.append(Read(
                name=src.name,
                sequence=seq,
                quality=qual,
                cigar=_parse_cigar(cigar_field.strip()),
            ))
        returncode = proc.wait()
        stderr = proc.stderr.read() if proc.stderr else ""
        if returncode != 0:
            raise RuntimeError(
                f"skiver-generate failed (exit {returncode}):\n{stderr}"
            )
    finally:
        input_path.unlink(missing_ok=True)

    if paired:
        result = [
            (new_flat[i], new_flat[i + 1])
            for i in range(0, len(new_flat), 2)
        ]
        return ReadBatch(paired=result)
    return ReadBatch(single=new_flat)
