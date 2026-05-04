#!/usr/bin/env python3
"""Entry point for Data Quality Analyzer."""

import sys
from dataset_doctor.cli import run_cli

if __name__ == "__main__":
    sys.exit(run_cli())
