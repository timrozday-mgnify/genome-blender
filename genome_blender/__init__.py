"""Synthetic whole genome sequencing data generator.

Re-export public API for convenience.
"""

from genome_blender._utils import (
    gc_fraction as _gc_fraction,
    lognormal_params_from_mean_variance
    as _lognormal_params_from_mean_variance,
    nb_params_from_mean_variance
    as _nb_params_from_mean_variance,
    reverse_complement as _reverse_complement,
)
from genome_blender.cli import (
    _apply_yaml_config,
    _load_yaml_config,
    app,
    main,
)
from genome_blender.error_model import (
    _build_emission_logits,
    _build_sticky_transitions,
    apply_error_model,
    apply_errors_to_sequence,
    batch_sample_quality_scores,
    build_quality_calibration,
    default_illumina_profile,
    default_nanopore_profile,
    default_pacbio_profile,
)
from genome_blender.fragments import (
    amplicon_fragments,
    sample_fragments,
)
from genome_blender.genomes import load_genomes
from genome_blender.io import (
    build_bam_header,
    write_bam,
    write_bam_chunk,
    write_fastq,
)
from genome_blender.models import (
    ErrorModel,
    ErrorModelProfile,
    Fragment,
    LogLinearCalibration,
    PhredCalibration,
    QualityCalibration,
    QualityCalibrationModel,
    Read,
    ReadBatch,
    SigmoidCalibration,
)
from genome_blender.reads import generate_reads

__all__ = [
    "Fragment",
    "Read",
    "ReadBatch",
    "ErrorModel",
    "ErrorModelProfile",
    "QualityCalibration",
    "QualityCalibrationModel",
    "PhredCalibration",
    "LogLinearCalibration",
    "SigmoidCalibration",
    "app",
    "main",
    "load_genomes",
    "sample_fragments",
    "amplicon_fragments",
    "generate_reads",
    "apply_error_model",
    "apply_errors_to_sequence",
    "batch_sample_quality_scores",
    "build_quality_calibration",
    "default_illumina_profile",
    "default_pacbio_profile",
    "default_nanopore_profile",
    "write_fastq",
    "write_bam",
    "write_bam_chunk",
    "build_bam_header",
    "_build_emission_logits",
    "_build_sticky_transitions",
    "_gc_fraction",
    "_reverse_complement",
    "_nb_params_from_mean_variance",
    "_lognormal_params_from_mean_variance",
    "_load_yaml_config",
    "_apply_yaml_config",
]
