import pytest
from dataset_doctor.scorer import QualityScorer

def test_scorer_empty_dataset():
    scorer = QualityScorer({"num_rows": 0, "num_cols": 0, "columns": {}})
    result = scorer.score()
    
    assert result["score"] == 0.0
    assert result["penalties"] == {}
    assert result["warnings"] == []

def test_scorer_all_nulls():
    profile = {
        "num_rows": 100,
        "num_cols": 2,
        "columns": {
            "A": {"missing_count": 100, "is_numeric": True, "outlier_count": 0, "type_consistent": True},
            "B": {"missing_count": 100, "is_numeric": True, "outlier_count": 0, "type_consistent": True}
        }
    }
    scorer = QualityScorer(profile)
    result = scorer.score()
    
    assert result["penalties"]["missing_values"] == 30.0 # max penalty for missing
    assert result["score"] == 70.0

def test_scorer_single_column():
    profile = {
        "num_rows": 10,
        "num_cols": 1,
        "columns": {
            "A": {"missing_count": 0, "is_numeric": True, "outlier_count": 1, "type_consistent": True}
        }
    }
    scorer = QualityScorer(profile)
    result = scorer.score()
    
    # outlier ratio = 1 / 10 = 0.1
    # penalty = min(20, 0.1 * 200) = 20.0
    assert result["penalties"]["outliers"] == 20.0
    assert result["score"] == 80.0

def test_scorer_type_inconsistency():
    profile = {
        "num_rows": 10,
        "num_cols": 1,
        "columns": {
            "A": {"missing_count": 0, "is_numeric": False, "type_consistent": False}
        }
    }
    scorer = QualityScorer(profile)
    result = scorer.score()
    
    assert result["penalties"]["type_inconsistency"] == 10.0
    assert result["score"] == 90.0

def test_scorer_low_cardinality_warning():
    profile = {
        "num_rows": 1000,
        "num_cols": 1,
        "columns": {
            "target": {"is_numeric": False, "cardinality_ratio": 0.002, "unique_count": 2} # 2 unique values out of 1000
        }
    }
    scorer = QualityScorer(profile)
    warnings = scorer.get_low_cardinality_warnings()
    
    assert len(warnings) == 1
    assert "target" in warnings[0]
    assert "low cardinality" in warnings[0]
