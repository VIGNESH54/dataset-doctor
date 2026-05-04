"""Quality scorer module for calculating an overall dataset quality score."""

from typing import Dict, Any

class QualityScorer:
    """Assigns a quality score from 0 to 100 based on dataset profiling metrics."""

    def __init__(self, profile_data: Dict[str, Any]):
        """Initialize with dataset profile data."""
        self.profile = profile_data

    def _calculate_missing_penalty(self) -> float:
        """Calculate penalty for missing values across all columns."""
        if not self.profile.get("columns") or self.profile.get("num_rows", 0) == 0:
            return 0.0
            
        total_missing = sum(
            col_data.get("missing_count", 0) 
            for col_data in self.profile["columns"].values()
        )
        total_cells = self.profile["num_rows"] * self.profile["num_cols"]
        missing_ratio = total_missing / total_cells
        # Up to 30 points penalty for missing data
        return min(30.0, missing_ratio * 100.0)

    def _calculate_outlier_penalty(self) -> float:
        """Calculate penalty for outlier density in numeric columns."""
        if not self.profile.get("columns") or self.profile.get("num_rows", 0) == 0:
            return 0.0

        total_outliers = 0
        numeric_cols = 0
        for col_data in self.profile["columns"].values():
            if col_data.get("is_numeric"):
                total_outliers += col_data.get("outlier_count", 0)
                numeric_cols += 1
                
        if numeric_cols == 0:
            return 0.0
            
        outlier_ratio = total_outliers / (self.profile["num_rows"] * numeric_cols)
        # Up to 20 points penalty for outliers
        return min(20.0, outlier_ratio * 200.0)

    def _calculate_type_inconsistency_penalty(self) -> float:
        """Calculate penalty for columns with inconsistent types."""
        if not self.profile.get("columns"):
            return 0.0

        inconsistent_cols = sum(
            1 for col_data in self.profile["columns"].values()
            if not col_data.get("type_consistent", True)
        )
        # 10 points penalty per inconsistent column, up to 20 points
        return min(20.0, inconsistent_cols * 10.0)

    def _calculate_duplicate_penalty(self, duplicate_ratio: float = 0.0) -> float:
        """Calculate penalty for duplicate rows."""
        # Note: duplicate rows need to be provided to the scorer, 
        # or we assume 0 if not passed (profiler didn't calculate row-wise duplicates)
        # Let's assume the user passes duplicate ratio when scoring
        return min(20.0, duplicate_ratio * 100.0)

    def get_low_cardinality_warnings(self) -> list[str]:
        """Identify columns with suspiciously low cardinality."""
        warnings = []
        if not self.profile.get("columns") or self.profile.get("num_rows", 0) < 100:
            return warnings
            
        for col, data in self.profile["columns"].items():
            # Flag non-numeric columns with > 0 but very low cardinality ratio
            # e.g. < 1% unique values but maybe they are leaking labels?
            ratio = data.get("cardinality_ratio", 1.0)
            if not data.get("is_numeric") and ratio < 0.01 and data.get("unique_count", 0) > 1:
                warnings.append(f"Column '{col}' has low cardinality. Check for target leakage.")
                
        return warnings

    def score(self, duplicate_ratio: float = 0.0) -> Dict[str, Any]:
        """Calculate and return the overall dataset quality score."""
        if not self.profile.get("columns"):
            return {"score": 0.0, "penalties": {}, "warnings": []}

        penalties = {
            "missing_values": self._calculate_missing_penalty(),
            "outliers": self._calculate_outlier_penalty(),
            "type_inconsistency": self._calculate_type_inconsistency_penalty(),
            "duplicates": self._calculate_duplicate_penalty(duplicate_ratio)
        }

        total_penalty = sum(penalties.values())
        final_score = max(0.0, 100.0 - total_penalty)

        return {
            "score": round(final_score, 2),
            "penalties": {k: round(v, 2) for k, v in penalties.items()},
            "warnings": self.get_low_cardinality_warnings()
        }
