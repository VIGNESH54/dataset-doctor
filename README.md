# Data Quality Analyzer

A production-ready Python tool that takes any CSV dataset as input, runs automated statistical quality checks, uses an LLM (Anthropic Claude 3 Haiku) to generate human-readable insights and recommendations, and produces a comprehensive HTML report.

## Features

- **Data Profiling**: Computes missing values, uniqueness, data types, and statistical metrics (mean, median, std, skewness, kurtosis, outliers).
- **Quality Scoring**: Assigns a score (0-100) based on missing values, outliers, duplicate rows, and type inconsistencies.
- **LLM Insights**: Leverages Anthropic Claude to analyze the profile and provide critical issues, cleaning steps, and warnings (e.g., target leakage).
- **HTML Report**: Generates a self-contained, beautifully styled HTML report with gauge charts and color-coded metrics.

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

### Sample Data
Three sample datasets are included for demonstration:
- `sample_data/clean.csv`: A clean dataset with no missing values.
- `sample_data/messy.csv`: Contains missing values, outliers, and type inconsistencies.
- `sample_data/leakage.csv`: Contains columns that strongly leak the target variable.

## Architecture Overview

- `profiler.py`: `DataProfiler` calculates statistics without modifying the data.
- `scorer.py`: `QualityScorer` computes penalty-based scoring.
- `llm.py`: `LLMInsightGenerator` handles communication with the Anthropic API.
- `report.py`: `ReportGenerator` creates the final HTML using Jinja2 and Matplotlib.
- `cli.py`: Connects all components and provides console output using `rich`.

## Design Decisions and Known Limitations

- **Scalability**: The `DataProfiler` currently loads the entire dataset into memory via pandas. Extremely large datasets might cause memory issues.
- **LLM Insights Constraints**: The prompt passes a JSON summary of the dataset profile, not the raw data. The LLM can only infer based on statistical summaries. It cannot detect complex row-level data entry errors or nuanced domain-specific anomalies that aren't reflected in the summary stats.
- **Data Type Inference**: The type consistency check is simplified and relies on pandas' internal dtype handling and checking the python types of non-null values.
- **Correlation Matrix**: Currently calculated only for purely numeric columns. Mixed-type columns (even if mostly numeric) are excluded to prevent errors.

## Testing

Run the test suite using `pytest`:

```bash
pytest tests/
```
