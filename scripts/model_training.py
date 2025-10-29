"""
Model Training Module
Trains multiple ML models and saves the best performer
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import json
from datetime import datetime

def load_engineered_data(filepath='data/engineered_features.csv'):
    """Load the feature-engineered dataset"""
    print(f"[v0] Loading engineered data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[v0] Loaded {len(df)} records with {len(df.columns)} features")
    return df

def prepare_train_test_split(df, test_size=0.2, random_state=42):
    """
    Prepare training and testing datasets
    """
    print("[v0] Preparing train/test split...")
    
    # Select features for modeling (exclude categorical originals and user_id)
    exclude_cols = ['user_id', 'engaged', 'device_type', 'user_segment', 
                    'traffic_source', 'session_duration_bin', 'user_age_bin']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df['engaged']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"[v0] Training set: {len(X_train)} samples")
    print(f"[v0] Testing set: {len(X_test)} samples")
    print(f"[v0] Features used: {len(feature_cols)}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save feature names
    with open('models/feature_names.json', 'w') as f:
        json.dump(feature_cols, f)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols

def train_logistic_regression(X_train, y_train):
    """Train Logistic Regression model"""
    print("[v0] Training Logistic Regression...")
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    print("[v0] Logistic Regression training complete")
    return model

def train_random_forest(X_train, y_train):
    """Train Random Forest model"""
    print("[v0] Training Random Forest...")
    
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("[v0] Random Forest training complete")
    return model

def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting model"""
    print("[v0] Training Gradient Boosting...")
    
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    print("[v0] Gradient Boosting training complete")
    return model

def save_models(models, model_names):
    """Save trained models"""
    print("[v0] Saving models...")
    
    for model, name in zip(models, model_names):
        filepath = f'models/{name}.pkl'
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"[v0] Saved {name} to {filepath}")
    
    # Save training metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'models_trained': model_names,
        'framework': 'scikit-learn'
    }
    
    with open('models/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

# Main execution
if __name__ == "__main__":
    print("[v0] Starting model training pipeline...")
    
    # Load data
    df = load_engineered_data()
    
    # Prepare train/test split
    X_train, X_test, y_train, y_test, feature_cols = prepare_train_test_split(df)
    
    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)
    
    # Save models
    models = [lr_model, rf_model, gb_model]
    model_names = ['logistic_regression', 'random_forest', 'gradient_boosting']
    save_models(models, model_names)
    
    # Save test data for evaluation
    test_data = {
        'X_test': X_test.tolist(),
        'y_test': y_test.tolist()
    }
    with open('models/test_data.json', 'w') as f:
        json.dump(test_data, f)
    
    print("[v0] Model training complete!")
    print(f"[v0] Trained {len(models)} models successfully")
