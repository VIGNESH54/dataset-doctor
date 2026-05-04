"""Time series checker module for temporal integrity and stationarity."""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from statsmodels.tsa.stattools import adfuller

@dataclass
class TimeSeriesQualityReport:
    """Dataclass for time series quality results."""
    datetime_columns: List[str] = field(default_factory=list)
    gap_count: int = 0
    out_of_order_count: int = 0
    duplicate_timestamp_count: int = 0
    spike_count: int = 0
    non_stationary_columns: List[str] = field(default_factory=list)
    overall_ts_health_score: float = 100.0

class TimeSeriesChecker:
    """Checks temporal integrity and stationarity of time series data."""

    def __init__(self):
        """Initialize the time series checker."""
        pass

    def detect_datetime_columns(self, df: pd.DataFrame) -> List[str]:
        """Identify datetime columns."""
        datetime_cols = []
        common_patterns = ['date', 'time', 'timestamp', 'created', 'updated', 'recorded']
        
        for col in df.columns:
            # Check by name pattern
            name_match = any(pattern in col.lower() for pattern in common_patterns)
            
            # Check by attempting conversion
            if name_match or df[col].dtype == 'object':
                try:
                    # Attempt conversion on a sample to be efficient
                    sample = df[col].dropna().head(10)
                    if not sample.empty:
                        pd.to_datetime(sample)
                        datetime_cols.append(col)
                except (ValueError, TypeError):
                    pass
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
                
        return datetime_cols

    def check_temporal_integrity(self, df: pd.DataFrame, datetime_col: str) -> Dict[str, Any]:
        """Check gaps, order, duplicates, and spikes."""
        results = {
            "gap_count": 0,
            "out_of_order_count": 0,
            "duplicate_timestamp_count": 0,
            "spike_count": 0
        }
        
        if df.empty or datetime_col not in df.columns:
            return results

        # Ensure datetime type
        temp_df = df.copy()
        temp_df[datetime_col] = pd.to_datetime(temp_df[datetime_col])
        temp_df = temp_df.dropna(subset=[datetime_col])
        
        # 1. Duplicate timestamps
        results["duplicate_timestamp_count"] = int(temp_df[datetime_col].duplicated().sum())
        
        # 2. Out-of-order timestamps
        # A simple check: if sorting changes the order
        is_sorted = temp_df[datetime_col].is_monotonic_increasing
        if not is_sorted:
            # Count how many rows are not in increasing order relative to predecessor
            results["out_of_order_count"] = int((temp_df[datetime_col].diff().dt.total_seconds() < 0).sum())

        # Sort for gap and spike detection
        temp_df = temp_df.sort_values(datetime_col)
        
        # 3. Gaps (missing regular intervals)
        if len(temp_df) > 2:
            intervals = temp_df[datetime_col].diff().dropna()
            mode_interval = intervals.mode()
            if not mode_interval.empty:
                expected_interval = mode_interval[0]
                # Count gaps larger than 1.5 * expected interval
                results["gap_count"] = int((intervals > 1.5 * expected_interval).sum())

        # 4. Spikes (values > 5 std from 10-period rolling mean)
        numeric_cols = temp_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            global_std = temp_df[col].std()
            rolling_mean = temp_df[col].rolling(window=10).mean()
            # We use global_std to define the threshold, but rolling_mean for the baseline
            spikes = (np.abs(temp_df[col] - rolling_mean) > 5 * global_std).sum()
            results["spike_count"] += int(spikes)

        return results

    def stationarity_check(self, df: pd.DataFrame, numeric_col: str) -> Dict[str, Any]:
        """Augmented Dickey-Fuller test for stationarity."""
        series = df[numeric_col].dropna()
        if len(series) < 20: # ADF needs sufficient data
            return {"stationary": True, "p_value": 1.0, "statistic": 0.0}
            
        try:
            result = adfuller(series)
            return {
                "stationary": bool(result[1] < 0.05),
                "p_value": float(result[1]),
                "statistic": float(result[0])
            }
        except Exception:
            return {"stationary": True, "p_value": 1.0, "statistic": 0.0}

    def run_checks(self, df: pd.DataFrame) -> Optional[TimeSeriesQualityReport]:
        """Run all time series checks if datetime columns are found."""
        datetime_cols = self.detect_datetime_columns(df)
        if not datetime_cols:
            return None
            
        report = TimeSeriesQualityReport(datetime_columns=datetime_cols)
        
        # We focus on the first detected datetime column for integrity checks
        primary_dt = datetime_cols[0]
        integrity = self.check_temporal_integrity(df, primary_dt)
        report.gap_count = integrity["gap_count"]
        report.out_of_order_count = integrity["out_of_order_count"]
        report.duplicate_timestamp_count = integrity["duplicate_timestamp_count"]
        report.spike_count = integrity["spike_count"]
        
        # Stationarity check for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            res = self.stationarity_check(df, col)
            if not res["stationary"]:
                report.non_stationary_columns.append(col)
        
        # Calculate health score (simple heuristic)
        penalties = (report.gap_count * 2) + (report.out_of_order_count * 5) + \
                    (report.duplicate_timestamp_count * 5) + (report.spike_count * 1)
        report.overall_ts_health_score = max(0.0, 100.0 - penalties)
        
        return report
