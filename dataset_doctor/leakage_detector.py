"""Leakage detector module for identifying ID columns and target leakage."""

import hashlib
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

@dataclass
class LeakageReport:
    """Dataclass for leakage detection results."""
    id_columns: List[str] = field(default_factory=list)
    high_risk_columns: List[str] = field(default_factory=list)
    medium_risk_columns: List[str] = field(default_factory=list)
    contamination_percentage: float = 0.0
    recommendations: List[str] = field(default_factory=list)

class LeakageDetector:
    """Detects ID columns, target leakage, and train-test contamination."""

    def __init__(self):
        """Initialize the leakage detector."""
        pass

    def detect_id_columns(self, df: pd.DataFrame) -> List[str]:
        """Flag columns that are likely ID columns."""
        id_cols = []
        num_rows = len(df)
        if num_rows == 0:
            return id_cols

        for col in df.columns:
            # Cardinality ratio
            cardinality_ratio = df[col].nunique() / num_rows
            
            # Name matching
            name_match = any(pattern in col.lower() for pattern in ['id', 'key', 'index', 'uuid'])
            
            # Pattern matching for values (simplified)
            is_uuid = False
            if df[col].dtype == 'object':
                sample = df[col].dropna().head(10).astype(str)
                if not sample.empty:
                    # Very basic UUID pattern check
                    is_uuid = all(len(s) >= 32 for s in sample)
            
            is_sequential = False
            if pd.api.types.is_integer_dtype(df[col]):
                sample = df[col].dropna().head(10)
                if len(sample) >= 2:
                    diffs = sample.diff().dropna()
                    is_sequential = all(diffs == 1)

            if (cardinality_ratio > 0.95 and name_match) or is_uuid or (cardinality_ratio > 0.99 and is_sequential):
                id_cols.append(col)
                
        return id_cols

    def detect_target_leakage(self, df: pd.DataFrame, target_column: str) -> Dict[str, List[str]]:
        """Identify potential target leakage."""
        results = {"high_risk": [], "medium_risk": []}
        
        if target_column not in df.columns or df.empty:
            return results

        # Drop rows with NaN in target
        df_clean = df.dropna(subset=[target_column])
        if df_clean.empty:
            return results

        y = df_clean[target_column]
        X = df_clean.drop(columns=[target_column])

        # Prepare features for mutual info (simplified: only numeric)
        X_numeric = X.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            return results

        # Determine if classification or regression
        if pd.api.types.is_numeric_dtype(y) and y.nunique() > 10:
            mi_scores = mutual_info_regression(X_numeric, y)
        else:
            mi_scores = mutual_info_classif(X_numeric, y)

        # Convert nats to bits
        mi_scores = mi_scores / np.log(2)

        for col, score in zip(X_numeric.columns, mi_scores):
            if score > 0.95:
                results["high_risk"].append(col)
            elif score > 0.7:
                results["medium_risk"].append(col)

        # Pearson correlation check
        if pd.api.types.is_numeric_dtype(y):
            corrs = X_numeric.corrwith(y).abs()
            for col, corr in corrs.items():
                if corr > 0.98 and col not in results["high_risk"]:
                    results["high_risk"].append(col)

        return results

    def detect_train_test_contamination(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
        """Check for row-level overlap using hashing."""
        if train_df.empty or test_df.empty:
            return {"count": 0, "percentage": 0.0}

        def hash_row(row):
            return hashlib.sha256(str(row.values).encode()).hexdigest()

        train_hashes = set(train_df.apply(hash_row, axis=1))
        test_hashes = test_df.apply(hash_row, axis=1)
        
        duplicates = test_hashes.isin(train_hashes).sum()
        percentage = (duplicates / len(test_df)) * 100
        
        return {"count": int(duplicates), "percentage": float(percentage)}

    def run_full_detection(self, df: pd.DataFrame, target_column: Optional[str] = None, 
                          test_df: Optional[pd.DataFrame] = None) -> LeakageReport:
        """Run all leakage detection methods and return a report."""
        report = LeakageReport()
        report.id_columns = self.detect_id_columns(df)
        
        if target_column:
            leakage_results = self.detect_target_leakage(df, target_column)
            report.high_risk_columns = leakage_results["high_risk"]
            report.medium_risk_columns = leakage_results["medium_risk"]
            
        if test_df is not None:
            contamination = self.detect_train_test_contamination(df, test_df)
            report.contamination_percentage = contamination["percentage"]

        # Generate recommendations
        if report.id_columns:
            report.recommendations.append(f"Remove ID columns before training: {', '.join(report.id_columns)}")
        if report.high_risk_columns:
            report.recommendations.append(f"CRITICAL: High leakage risk detected in: {', '.join(report.high_risk_columns)}")
        if report.medium_risk_columns:
            report.recommendations.append(f"Review potential leakage in: {', '.join(report.medium_risk_columns)}")
        if report.contamination_percentage > 1.0:
            report.recommendations.append(f"High train-test contamination ({report.contamination_percentage:.2f}%). Resplit your data.")

        return report
