import pytest
import pandas as pd
import numpy as np
from dataset_doctor.leakage_detector import LeakageDetector

@pytest.fixture
def detector():
    return LeakageDetector()

def test_detect_id_columns_by_name(detector):
    df = pd.DataFrame({
        'user_id': range(100),
        'age': np.random.randint(20, 50, 100)
    })
    ids = detector.detect_id_columns(df)
    assert 'user_id' in ids
    assert 'age' not in ids

def test_detect_id_columns_by_pattern(detector):
    df = pd.DataFrame({
        'uuid': [f"uuid-{i}" for i in range(100)],
        'data': np.random.rand(100)
    })
    ids = detector.detect_id_columns(df)
    assert 'uuid' in ids

def test_detect_target_leakage_high(detector):
    df = pd.DataFrame({
        'target': [0, 1] * 50,
        'leaky': [0, 1] * 50,
        'noise': np.random.rand(100)
    })
    results = detector.detect_target_leakage(df, 'target')
    assert 'leaky' in results['high_risk']

def test_detect_target_leakage_correlation(detector):
    y = np.random.rand(100)
    df = pd.DataFrame({
        'target': y,
        'leaky': y + 0.0001,  # Near perfect correlation
        'normal': np.random.rand(100)
    })
    results = detector.detect_target_leakage(df, 'target')
    assert 'leaky' in results['high_risk']

def test_detect_train_test_contamination(detector):
    train = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    test = pd.DataFrame({'a': [1, 9, 10], 'b': [4, 11, 12]}) # Row 1 is same
    results = detector.detect_train_test_contamination(train, test)
    assert results['count'] == 1
    assert abs(results['percentage'] - 33.33) < 0.1

def test_leakage_report_recommendations(detector):
    df = pd.DataFrame({
        'id': range(100),
        'target': [0, 1] * 50,
        'leaky': [0, 1] * 50
    })
    report = detector.run_full_detection(df, target_column='target')
    assert len(report.recommendations) >= 2
    assert any("Remove ID columns" in r for r in report.recommendations)
    assert any("High leakage risk" in r for r in report.recommendations)

def test_detect_id_sequential(detector):
    df = pd.DataFrame({'seq': range(1000, 1100)})
    # Cardinality ratio is 1.0, but name doesn't match ID. 
    # Our logic: (cardinality_ratio > 0.99 and is_sequential)
    ids = detector.detect_id_columns(df)
    assert 'seq' in ids

def test_detect_target_leakage_medium(detector):
    # This is trickier to synthesize perfectly for mutual info, 
    # but let's try a strong but not perfect relationship.
    target = np.random.randint(0, 2, 1000)
    leaky = target.copy()
    # Flip 2% of labels to keep MI high but not perfect
    mask = np.random.rand(1000) < 0.02
    leaky[mask] = 1 - leaky[mask]
    
    df = pd.DataFrame({
        'target': target,
        'feature': leaky
    })
    results = detector.detect_target_leakage(df, 'target')
    # With 2% noise, MI should be > 0.7
    assert 'feature' in results['high_risk'] or 'feature' in results['medium_risk']

def test_empty_dataframe(detector):
    df = pd.DataFrame()
    ids = detector.detect_id_columns(df)
    assert ids == []
    results = detector.detect_target_leakage(df, 'target')
    assert results['high_risk'] == []

def test_no_contamination(detector):
    train = pd.DataFrame({'a': [1, 2, 3]})
    test = pd.DataFrame({'a': [4, 5, 6]})
    results = detector.detect_train_test_contamination(train, test)
    assert results['count'] == 0
    assert results['percentage'] == 0.0
