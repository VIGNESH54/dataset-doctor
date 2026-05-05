# 🩺 Dataset Doctor

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](https://github.com/VIGNESH54/Automated-Data-Quality-Analyzer/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready Python tool that "diagnoses" your data. It takes any CSV dataset as input, runs automated statistical quality checks, uses an LLM (Anthropic Claude 3 Haiku) to generate human-readable insights, and produces a comprehensive HTML report.

## Why Dataset Doctor?

| Feature | Dataset Doctor | ydata-profiling | Great Expectations | Deepchecks |
| :--- | :---: | :---: | :---: | :---: |
| **LLM Cleaning Code** | ✅ | ❌ | ❌ | ❌ |
| **ML Leakage Detection** | ✅ | ❌ | ❌ | ✅ |
| **TS Health Scoring** | ✅ | ⚠️ | ❌ | ⚠️ |
| **Zero-Server HTML** | ✅ | ✅ | ❌ | ✅ |
| **Pipeline Integration** | ⚠️ | ⚠️ | ✅ | ⚠️ |
| **Statistical Depth** | ⚠️ | ✅ | ⚠️ | ✅ |

> **Comparison Note**: While `Great Expectations` excels at production pipeline integration and `ydata-profiling` offers deeper statistical distributions, **Dataset Doctor** is uniquely focused on actionable AI-driven insights and specialized ML/Time-Series diagnostics in a lightweight, zero-config package.

## Features

- **Data Profiling**: Computes missing values, uniqueness, data types, and statistical metrics (mean, median, std, skewness, kurtosis, outliers).
- **Quality Scoring**: Assigns a score (0-100) based on missing values, outliers, duplicate rows, and type inconsistencies.
- **LLM Insights**: Leverages Anthropic Claude to analyze the profile and provide critical issues, cleaning steps, and warnings.
- **ML Readiness**: Detects target leakage, ID columns, and train-test contamination.
- **Time Series Health**: Identifies gaps, out-of-order timestamps, and non-stationary series.
- **HTML Report**: Generates a self-contained, beautifully styled HTML report with gauge charts.

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Anthropic API key in a `.env` file (copy from `.env.example`):
   ```bash
   cp .env.example .env
   # Edit .env and add your key
   ```

## Usage

Run the tool via the CLI:

```bash
python main.py --input sample_data/messy.csv --output reports/report.html --llm-insights
```

- `--input`: Path to your CSV dataset.
- `--output`: Path where the HTML report will be saved.
- `--llm-insights`: (Optional) Include to fetch insights from Claude. Requires the `ANTHROPIC_API_KEY` to be set.
- `--target-column`: (Optional) Enable ML leakage detection for a specific target.

### Sample Data
Sample datasets are included for demonstration:
- `sample_data/clean.csv`: A clean dataset.
- `sample_data/messy.csv`: Contains missing values and outliers.
- `sample_data/leaky_dataset.csv`: Demonstrates ML leakage detection.
- `sample_data/timeseries.csv`: Demonstrates time series health checks.

## Architecture Overview

- `profiler.py`: `DataProfiler` calculates statistics.
- `scorer.py`: `QualityScorer` computes penalty-based scoring.
- `leakage_detector.py`: `LeakageDetector` identifies ML risks.
- `timeseries_checker.py`: `TimeSeriesChecker` analyzes temporal integrity.
- `llm.py`: `LLMInsightGenerator` handles communication with Claude.
- `report.py`: `ReportGenerator` creates the final HTML.
- `cli.py`: Connects all components using `rich`.

## Project Structure

```text
dataset-doctor/
├── data_quality_analyzer/
│   ├── __init__.py
│   ├── profiler.py          # DataProfiler — statistical column analysis
│   ├── scorer.py            # QualityScorer — penalty-based scoring
│   ├── leakage_detector.py  # LeakageDetector — ML risk identification
│   ├── timeseries_checker.py # TimeSeriesChecker — temporal integrity
│   ├── llm.py               # LLMInsightGenerator — Claude API integration
│   └── report.py            # ReportGenerator — HTML report creation
├── tests/
│   ├── test_profiler.py
│   ├── test_scorer.py
│   ├── test_leakage_detector.py
│   └── test_timeseries_checker.py
├── sample_data/
│   ├── clean.csv
│   ├── messy.csv
│   ├── leaky_dataset.csv
│   └── timeseries.csv
├── main.py
├── cli.py
├── requirements.txt
├── .env.example
└── README.md
```

## Design Decisions and Known Limitations

- **Scalability**: Entire dataset is loaded into memory.
- **LLM Context**: We pass statistical summaries, not raw rows, to ensure privacy and token efficiency.
- **Stationarity**: Uses Augmented Dickey-Fuller (ADF) which works best on series with >20 points.

## Testing

Run the full test suite using pytest. All tests use mocked external calls — no API key required to run tests. 30 tests pass when you run `python -m pytest tests/ -v`.
