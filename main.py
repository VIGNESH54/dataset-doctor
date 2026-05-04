#!/usr/bin/env python3
"""Entry point for Data Quality Analyzer."""

import sys
from data_quality_analyzer.cli import run_cli

if __name__ == "__main__":
    sys.exit(run_cli())
