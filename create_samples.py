import pandas as pd
import numpy as np
import os

def create_sample_datasets():
    os.makedirs('sample_data', exist_ok=True)
    
    # 1. Clean dataset
    clean_df = pd.DataFrame({
        'id': range(1, 101),
        'age': np.random.randint(18, 65, 100),
        'salary': np.random.normal(60000, 10000, 100),
        'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], 100)
    })
    clean_df.to_csv('sample_data/clean.csv', index=False)
    
    # 2. Messy dataset (missing values, outliers, mixed types)
    messy_df = clean_df.copy()
    # Add missing values
    messy_df.loc[5:15, 'age'] = np.nan
    messy_df.loc[20:30, 'salary'] = np.nan
    # Add outliers
    messy_df.loc[0, 'salary'] = 1000000
    messy_df.loc[1, 'salary'] = 2000000
    # Add type inconsistency (represented as strings in a numeric column)
    messy_df.loc[50:55, 'age'] = 'Unknown'
    messy_df.to_csv('sample_data/messy.csv', index=False)
    
    # 3. Leakage dataset (target leakage)
    leakage_df = clean_df.copy()
    # Target variable 'churn'
    leakage_df['churn'] = np.random.choice([0, 1], 100)
    # Leakage variable perfectly correlated with target
    leakage_df['churn_reason'] = leakage_df['churn'].apply(lambda x: 'Price' if x == 1 else 'N/A')
    # Low cardinality categorical that could be leakage
    leakage_df['has_cancelled_plan'] = leakage_df['churn']
    leakage_df.to_csv('sample_data/leakage.csv', index=False)
    # 4. Leaky ML dataset
    leaky_df = pd.DataFrame({
        'user_id': range(1000),
        'feature_1': np.random.rand(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    # Add leaky feature (target + small noise)
    leaky_df['leaky_feature'] = leaky_df['target'] + np.random.normal(0, 0.01, 1000)
    leaky_df.to_csv('sample_data/leaky_dataset.csv', index=False)

    # 5. Clean ML dataset
    clean_ml_df = pd.DataFrame({
        'id': range(1000),
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.normal(50000, 15000, 1000),
        'target': np.random.randint(0, 2, 1000)
    })
    clean_ml_df.to_csv('sample_data/clean_ml_dataset.csv', index=False)

    # 6. Contaminated splits
    os.makedirs('sample_data/contaminated_splits', exist_ok=True)
    base_data = pd.DataFrame({
        'feature': np.random.rand(200),
        'label': np.random.randint(0, 2, 200)
    })
    train = base_data.iloc[:150]
    # Test has 10 rows from train (5% overlap of 200 total, but user said 5% of test/total?)
    # "5% row overlap between train and test" - let's make it 10 rows shared.
    shared = base_data.iloc[:10]
    test = pd.concat([base_data.iloc[150:], shared]).sample(frac=1)
    train.to_csv('sample_data/contaminated_splits/train.csv', index=False)
    test.to_csv('sample_data/contaminated_splits/test.csv', index=False)
    
    print("Sample datasets created in sample_data/")

if __name__ == "__main__":
    create_sample_datasets()
