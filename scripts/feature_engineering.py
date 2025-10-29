"""
Feature Engineering Module
Creates derived features for improved model performance
"""

import pandas as pd
import numpy as np
import json

def load_data(filepath='data/user_engagement_data.csv'):
    """Load the cleaned dataset"""
    print(f"[v0] Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[v0] Loaded {len(df)} records")
    return df

def create_interaction_features(df):
    """
    Create interaction features between existing variables
    """
    print("[v0] Creating interaction features...")
    
    # Engagement intensity: session duration * page views
    df['engagement_intensity'] = df['session_duration'] * df['page_views']
    
    # Click efficiency: clicks per minute
    df['click_efficiency'] = df['click_rate'] / (df['session_duration'] + 1)
    
    # User maturity: total sessions / days since signup
    df['user_maturity'] = df['total_sessions'] / (df['days_since_signup'] + 1)
    
    # Session quality: avg session length * click rate
    df['session_quality'] = df['avg_session_length'] * df['click_rate'] / 100
    
    print(f"[v0] Created 4 interaction features")
    
    return df

def create_categorical_features(df):
    """
    Encode categorical variables
    """
    print("[v0] Encoding categorical features...")
    
    # One-hot encode device type
    device_dummies = pd.get_dummies(df['device_type'], prefix='device')
    df = pd.concat([df, device_dummies], axis=1)
    
    # One-hot encode user segment
    segment_dummies = pd.get_dummies(df['user_segment'], prefix='segment')
    df = pd.concat([df, segment_dummies], axis=1)
    
    # One-hot encode traffic source
    traffic_dummies = pd.get_dummies(df['traffic_source'], prefix='traffic')
    df = pd.concat([df, traffic_dummies], axis=1)
    
    print(f"[v0] Encoded categorical features")
    
    return df

def create_binned_features(df):
    """
    Create binned versions of continuous features
    """
    print("[v0] Creating binned features...")
    
    # Bin session duration
    df['session_duration_bin'] = pd.cut(
        df['session_duration'], 
        bins=[0, 5, 15, 30, 120], 
        labels=['very_short', 'short', 'medium', 'long']
    )
    
    # Bin days since signup
    df['user_age_bin'] = pd.cut(
        df['days_since_signup'], 
        bins=[0, 30, 90, 180, 365], 
        labels=['new', 'recent', 'established', 'veteran']
    )
    
    print(f"[v0] Created binned features")
    
    return df

def normalize_features(df, feature_cols):
    """
    Normalize numerical features to 0-1 range
    """
    print("[v0] Normalizing numerical features...")
    
    for col in feature_cols:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            min_val = df[col].min()
            max_val = df[col].max()
            if max_val > min_val:
                df[f'{col}_normalized'] = (df[col] - min_val) / (max_val - min_val)
    
    print(f"[v0] Normalized features")
    
    return df

def save_engineered_data(df, filepath='data/engineered_features.csv'):
    """
    Save the feature-engineered dataset
    """
    df.to_csv(filepath, index=False)
    print(f"[v0] Engineered data saved to {filepath}")
    
    # Save feature list for model training
    feature_info = {
        'total_features': len(df.columns),
        'numerical_features': df.select_dtypes(include=['float64', 'int64']).columns.tolist(),
        'categorical_features': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'target': 'engaged'
    }
    
    with open('public/feature_info.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    return df

# Main execution
if __name__ == "__main__":
    print("[v0] Starting feature engineering pipeline...")
    
    # Load data
    df = load_data()
    
    # Create interaction features
    df = create_interaction_features(df)
    
    # Encode categorical features
    df = create_categorical_features(df)
    
    # Create binned features
    df = create_binned_features(df)
    
    # Normalize key features
    numerical_cols = ['session_duration', 'page_views', 'click_rate', 'total_sessions']
    df = normalize_features(df, numerical_cols)
    
    # Save engineered data
    df = save_engineered_data(df)
    
    print("[v0] Feature engineering complete!")
    print(f"[v0] Final dataset shape: {df.shape}")
