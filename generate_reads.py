#!/usr/bin/env python3
"""Backwards-compatible shim — delegates to genome_blender package."""

from genome_blender import *  # noqa: F401, F403
from genome_blender.cli import app  # noqa: F401

if __name__ == "__main__":
    app()
