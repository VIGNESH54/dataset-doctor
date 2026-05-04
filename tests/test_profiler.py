import pytest
import pandas as pd
import numpy as np
from dataset_doctor.profiler import DataProfiler

def test_profiler_empty_dataset():
    df = pd.DataFrame()
    profiler = DataProfiler(df)
    profile = profiler.profile()
    
    assert profile["num_rows"] == 0
    assert profile["num_cols"] == 0
    assert profile["columns"] == {}
    assert profile["correlation_matrix"] == {}

def test_profiler_numeric_stats():
    df = pd.DataFrame({'A': [1, 2, 3, 4, 100]}) # 100 is an outlier
    profiler = DataProfiler(df)
    profile = profiler.profile()
    
    stats_A = profile["columns"]["A"]
    assert stats_A["is_numeric"] is True
    assert stats_A["mean"] == 22.0
    assert stats_A["outlier_count"] == 1
    assert stats_A["missing_count"] == 0

def test_profiler_missing_values():
    df = pd.DataFrame({'B': [1, np.nan, 3, np.nan]})
    profiler = DataProfiler(df)
    profile = profiler.profile()
    
    stats_B = profile["columns"]["B"]
    assert stats_B["missing_count"] == 2
    assert stats_B["missing_percentage"] == 50.0

def test_profiler_categorical_stats():
    df = pd.DataFrame({'C': ['apple', 'banana', 'apple', 'apple', 'orange']})
    profiler = DataProfiler(df)
    profile = profiler.profile()
    
    stats_C = profile["columns"]["C"]
    assert stats_C["is_numeric"] is False
    assert stats_C["top_frequencies"]["apple"] == 3
    assert stats_C["unique_count"] == 3
