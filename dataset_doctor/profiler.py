"""Data profiler module for computing statistics and metrics on datasets."""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from typing import Dict, Any

class DataProfiler:
    """Profiles a pandas DataFrame to extract column statistics and quality metrics."""

    def __init__(self, df: pd.DataFrame):
        """Initialize the profiler with a dataset."""
        self.df = df
        self.num_rows = len(df)
        self.num_cols = len(df.columns)

    def compute_missing_stats(self, col: str) -> Dict[str, float]:
        """Compute missing value count and percentage for a column."""
        missing = int(self.df[col].isna().sum())
        percentage = (missing / self.num_rows * 100) if self.num_rows > 0 else 0.0
        return {"missing_count": missing, "missing_percentage": percentage}

    def compute_uniqueness_stats(self, col: str) -> Dict[str, float]:
        """Compute unique value count and cardinality ratio."""
        unique_count = int(self.df[col].nunique(dropna=False))
        cardinality_ratio = (unique_count / self.num_rows) if self.num_rows > 0 else 0.0
        return {"unique_count": unique_count, "cardinality_ratio": cardinality_ratio}

    def infer_data_type(self, col: str) -> Dict[str, str]:
        """Infer data type and check for consistency."""
        dtype = str(self.df[col].dtype)
        # Simplified consistency check: all non-null values share the same type
        non_nulls = self.df[col].dropna()
        if len(non_nulls) == 0:
            consistent = True
        else:
            first_type = type(non_nulls.iloc[0])
            consistent = all(isinstance(x, first_type) for x in non_nulls)
        return {"inferred_type": dtype, "type_consistent": consistent}

    def compute_numeric_stats(self, col: str) -> Dict[str, Any]:
        """Compute statistics and outlier count for numeric columns."""
        if not pd.api.types.is_numeric_dtype(self.df[col]):
            return {}
        
        series = self.df[col].dropna()
        if len(series) == 0:
            return {}

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        outliers = int(((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum())

        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()) if len(series) > 1 else 0.0,
            "skewness": float(skew(series)),
            "kurtosis": float(kurtosis(series)),
            "outlier_count": outliers
        }

    def compute_categorical_stats(self, col: str) -> Dict[str, Any]:
        """Compute top 10 value frequencies for categorical columns."""
        if pd.api.types.is_numeric_dtype(self.df[col]):
            return {}
            
        freqs = self.df[col].value_counts().head(10).to_dict()
        # Convert keys to strings and values to ints for JSON serialization
        freqs_clean = {str(k): int(v) for k, v in freqs.items()}
        return {"top_frequencies": freqs_clean}

    def compute_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """Compute the correlation matrix for numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {}
        
        corr = numeric_df.corr().to_dict()
        # Clean up NaNs from correlation matrix
        clean_corr = {}
        for col1, row in corr.items():
            clean_corr[col1] = {
                col2: (0.0 if pd.isna(val) else float(val)) 
                for col2, val in row.items()
            }
        return clean_corr

    def profile(self) -> Dict[str, Any]:
        """Generate a complete profile of the dataset."""
        if self.df.empty:
            return {"columns": {}, "correlation_matrix": {}, "num_rows": 0, "num_cols": 0}

        profile_data = {
            "num_rows": self.num_rows,
            "num_cols": self.num_cols,
            "columns": {},
            "correlation_matrix": self.compute_correlation_matrix()
        }

        for col in self.df.columns:
            col_stats = {}
            col_stats.update(self.compute_missing_stats(col))
            col_stats.update(self.compute_uniqueness_stats(col))
            col_stats.update(self.infer_data_type(col))
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                col_stats.update(self.compute_numeric_stats(col))
                col_stats["is_numeric"] = True
            else:
                col_stats.update(self.compute_categorical_stats(col))
                col_stats["is_numeric"] = False
                
            profile_data["columns"][col] = col_stats

        return profile_data
