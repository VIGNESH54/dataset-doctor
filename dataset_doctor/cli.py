"""Command Line Interface for Dataset Doctor."""

import argparse
import pandas as pd
import sys
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from dotenv import load_dotenv

from .profiler import DataProfiler
from .scorer import QualityScorer
from .llm import LLMInsightGenerator
from .report import ReportGenerator
from .leakage_detector import LeakageDetector
from .timeseries_checker import TimeSeriesChecker

def run_cli() -> int:
    """Run the command line interface."""
    load_dotenv()
    console = Console()
    
    parser = argparse.ArgumentParser(description="Dataset Doctor: Automated Data Quality Analyzer")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=True, help="Path to output HTML report")
    parser.add_argument("--llm-insights", action="store_true", help="Enable LLM insights generation")
    parser.add_argument("--target-column", help="Target column for leakage detection")
    parser.add_argument("--test-data", help="Path to test CSV for contamination check")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        console.print(f"[red]Error: Input file '{args.input}' not found.[/red]")
        return 1

    try:
        with console.status(f"[bold green]Loading dataset {args.input}..."):
            df = pd.read_csv(args.input)
        console.print(f"[green]✓[/green] Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
        
        test_df = None
        if args.test_data:
            if os.path.exists(args.test_data):
                test_df = pd.read_csv(args.test_data)
                console.print(f"[green]✓[/green] Loaded test dataset with {len(test_df)} rows")
            else:
                console.print(f"[yellow]⚠[/yellow] Test data '{args.test_data}' not found. Skipping contamination check.")

        with console.status("[bold green]Profiling dataset..."):
            profiler = DataProfiler(df)
            profile_data = profiler.profile()
        console.print("[green]✓[/green] Completed data profiling")
        
        with console.status("[bold green]Scoring dataset quality..."):
            # Calculate a basic duplicate ratio for the scorer
            duplicate_ratio = (df.duplicated().sum() / len(df)) if len(df) > 0 else 0.0
            scorer = QualityScorer(profile_data)
            score_data = scorer.score(duplicate_ratio=duplicate_ratio)
        console.print(f"[green]✓[/green] Calculated quality score: [bold]{score_data['score']}/100[/bold]")
        
        leakage_report = None
        if args.target_column or test_df is not None:
            with console.status("[bold green]Running leakage detection..."):
                detector = LeakageDetector()
                leakage_report = detector.run_full_detection(df, target_column=args.target_column, test_df=test_df)
            console.print("[green]✓[/green] Completed leakage detection")

        ts_report = None
        with console.status("[bold green]Checking time series health..."):
            ts_checker = TimeSeriesChecker()
            ts_report = ts_checker.run_checks(df)
        if ts_report:
            console.print("[green]✓[/green] Completed time series health check")

        llm_insights = {
            "overall_assessment": "LLM insights not requested.",
            "critical_issues": [],
            "cleaning_steps": [],
            "warnings": []
        }
        
        if args.llm_insights:
            with console.status("[bold green]Generating LLM insights..."):
                llm = LLMInsightGenerator()
                if llm.client:
                    # Provide a minimal summary to avoid context window explosion
                    summary = {
                        "num_rows": profile_data["num_rows"],
                        "num_cols": profile_data["num_cols"],
                        "score": score_data["score"],
                        "columns_summary": {
                            col: {
                                "missing": data.get("missing_percentage"),
                                "outliers": data.get("outlier_count", 0),
                                "type": data.get("inferred_type")
                            } for col, data in profile_data["columns"].items()
                        }
                    }
                    llm_insights = llm.generate_insights(summary)
                    console.print("[green]✓[/green] Generated LLM insights")
                else:
                    console.print("[yellow]⚠[/yellow] Skipped LLM insights: ANTHROPIC_API_KEY not found.")
        
        with console.status("[bold green]Generating HTML report..."):
            report_gen = ReportGenerator()
            report_gen.generate_report(profile_data, score_data, llm_insights, args.output, 
                                     leakage_report=leakage_report, ts_report=ts_report)
        console.print(f"[green]✓[/green] Report generated at {args.output}")
        
        # Display summary table
        table = Table(title="Data Quality Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        table.add_row("Total Rows", str(profile_data["num_rows"]))
        table.add_row("Total Columns", str(profile_data["num_cols"]))
        table.add_row("Quality Score", f"{score_data['score']}/100")
        if leakage_report:
            table.add_row("ID Columns", str(len(leakage_report.id_columns)))
            table.add_row("Leaky Columns", str(len(leakage_report.high_risk_columns)))
        console.print(table)
        
        if score_data['warnings']:
            console.print(Panel("\n".join(score_data['warnings']), title="Warnings", border_style="yellow"))

        return 0

    except Exception as e:
        console.print(f"[bold red]An error occurred:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
        return 1
