import pytest
import pandas as pd
import numpy as np
from dataset_doctor.timeseries_checker import TimeSeriesChecker

@pytest.fixture
def checker():
    return TimeSeriesChecker()

def test_detect_datetime_columns(checker):
    df = pd.DataFrame({
        'timestamp': ['2023-01-01', '2023-01-02'],
        'value': [1, 2]
    })
    cols = checker.detect_datetime_columns(df)
    assert 'timestamp' in cols

def test_check_temporal_integrity_gaps(checker):
    # Regular daily data with one day missing
    dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-04'])
    df = pd.DataFrame({'ds': dates, 'val': [1, 2, 4]})
    results = checker.check_temporal_integrity(df, 'ds')
    assert results['gap_count'] == 1

def test_check_temporal_integrity_order(checker):
    dates = pd.to_datetime(['2023-01-02', '2023-01-01', '2023-01-03'])
    df = pd.DataFrame({'ds': dates})
    results = checker.check_temporal_integrity(df, 'ds')
    assert results['out_of_order_count'] >= 1

def test_check_temporal_integrity_duplicates(checker):
    dates = pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02'])
    df = pd.DataFrame({'ds': dates})
    results = checker.check_temporal_integrity(df, 'ds')
    assert results['duplicate_timestamp_count'] == 1

def test_check_temporal_integrity_spikes(checker):
    # Create 100 points, one is a huge spike
    vals = [10.0] * 100
    vals[50] = 5000.0
    df = pd.DataFrame({
        'ds': pd.date_range('2023-01-01', periods=100),
        'val': vals
    })
    results = checker.check_temporal_integrity(df, 'ds')
    assert results['spike_count'] >= 1

def test_stationarity_check_stationary(checker):
    # White noise is stationary
    df = pd.DataFrame({'val': np.random.normal(0, 1, 100)})
    res = checker.stationarity_check(df, 'val')
    assert res['stationary'] == True

def test_stationarity_check_non_stationary(checker):
    # Random walk is non-stationary
    df = pd.DataFrame({'val': np.cumsum(np.random.normal(0, 1, 100))})
    res = checker.stationarity_check(df, 'val')
    assert res['stationary'] == False

def test_run_checks_integration(checker):
    df = pd.DataFrame({
        'timestamp': pd.date_range('2023-01-01', periods=50),
        'val': np.random.normal(0, 1, 50)
    })
    report = checker.run_checks(df)
    assert report is not None
    assert report.overall_ts_health_score > 90
    assert 'timestamp' in report.datetime_columns
