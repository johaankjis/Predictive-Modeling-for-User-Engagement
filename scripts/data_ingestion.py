"""
Data Ingestion and Cleaning Module
Generates synthetic user engagement data for MVP demonstration
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def generate_synthetic_data(n_samples=10000, random_state=42):
    """
    Generate synthetic user engagement data
    Simulates realistic user behavior patterns
    """
    np.random.seed(random_state)
    
    print(f"[v0] Generating {n_samples} synthetic user records...")
    
    # Generate user IDs
    user_ids = [f"user_{i:05d}" for i in range(n_samples)]
    
    # Device types with realistic distribution
    device_types = np.random.choice(
        ['mobile', 'desktop', 'tablet'], 
        size=n_samples, 
        p=[0.6, 0.35, 0.05]
    )
    
    # User segments
    user_segments = np.random.choice(
        ['new', 'active', 'power_user', 'at_risk'], 
        size=n_samples, 
        p=[0.3, 0.4, 0.2, 0.1]
    )
    
    # Traffic sources
    traffic_sources = np.random.choice(
        ['organic', 'paid', 'social', 'direct', 'referral'], 
        size=n_samples, 
        p=[0.35, 0.25, 0.2, 0.15, 0.05]
    )
    
    # Numerical features with correlations
    days_since_signup = np.random.exponential(scale=60, size=n_samples).astype(int)
    days_since_signup = np.clip(days_since_signup, 1, 365)
    
    # Session duration (minutes) - influenced by device and segment
    base_duration = np.random.gamma(shape=2, scale=5, size=n_samples)
    device_multiplier = np.where(device_types == 'desktop', 1.3, 
                                 np.where(device_types == 'tablet', 1.1, 1.0))
    segment_multiplier = np.where(user_segments == 'power_user', 2.0,
                                  np.where(user_segments == 'active', 1.3,
                                          np.where(user_segments == 'new', 0.7, 0.5)))
    session_duration = base_duration * device_multiplier * segment_multiplier
    session_duration = np.clip(session_duration, 0.5, 120)
    
    # Page views - correlated with session duration
    page_views = (session_duration / 2 + np.random.poisson(lam=3, size=n_samples)).astype(int)
    page_views = np.clip(page_views, 1, 100)
    
    # Click rate - percentage of pages with clicks
    click_rate = np.random.beta(a=2, b=5, size=n_samples) * 100
    click_rate = np.clip(click_rate, 0, 100)
    
    # Total sessions - influenced by segment
    base_sessions = np.random.poisson(lam=10, size=n_samples)
    segment_session_mult = np.where(user_segments == 'power_user', 3.0,
                                    np.where(user_segments == 'active', 1.5,
                                            np.where(user_segments == 'new', 0.5, 0.3)))
    total_sessions = (base_sessions * segment_session_mult).astype(int)
    total_sessions = np.clip(total_sessions, 1, 200)
    
    # Average session length
    avg_session_length = session_duration * np.random.uniform(0.8, 1.2, size=n_samples)
    avg_session_length = np.clip(avg_session_length, 0.5, 100)
    
    # Target variable: engaged (1) or not engaged (0)
    # Based on multiple factors with realistic probabilities
    engagement_score = (
        (session_duration / 120) * 0.3 +
        (page_views / 100) * 0.2 +
        (click_rate / 100) * 0.25 +
        (total_sessions / 200) * 0.15 +
        (user_segments == 'power_user') * 0.1
    )
    
    # Add noise and convert to binary
    engagement_score += np.random.normal(0, 0.1, size=n_samples)
    engaged = (engagement_score > 0.5).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': user_ids,
        'device_type': device_types,
        'user_segment': user_segments,
        'traffic_source': traffic_sources,
        'session_duration': np.round(session_duration, 2),
        'page_views': page_views,
        'click_rate': np.round(click_rate, 2),
        'days_since_signup': days_since_signup,
        'total_sessions': total_sessions,
        'avg_session_length': np.round(avg_session_length, 2),
        'engaged': engaged
    })
    
    print(f"[v0] Generated dataset shape: {df.shape}")
    print(f"[v0] Engagement rate: {df['engaged'].mean():.2%}")
    print(f"[v0] Feature summary:\n{df.describe()}")
    
    return df

def clean_data(df):
    """
    Clean and validate the dataset
    """
    print("[v0] Cleaning data...")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print(f"[v0] Missing values found:\n{missing[missing > 0]}")
        df = df.dropna()
    
    # Remove duplicates
    initial_rows = len(df)
    df = df.drop_duplicates(subset=['user_id'])
    if len(df) < initial_rows:
        print(f"[v0] Removed {initial_rows - len(df)} duplicate records")
    
    # Validate numerical ranges
    df = df[df['session_duration'] > 0]
    df = df[df['page_views'] > 0]
    df = df[df['click_rate'] >= 0]
    
    print(f"[v0] Cleaned dataset shape: {df.shape}")
    
    return df

def save_data(df, filepath='data/user_engagement_data.csv'):
    """
    Save processed data to CSV
    """
    df.to_csv(filepath, index=False)
    print(f"[v0] Data saved to {filepath}")
    
    # Also save summary statistics as JSON for the web app
    summary = {
        'total_records': len(df),
        'engagement_rate': float(df['engaged'].mean()),
        'avg_session_duration': float(df['session_duration'].mean()),
        'avg_page_views': float(df['page_views'].mean()),
        'device_distribution': df['device_type'].value_counts().to_dict(),
        'segment_distribution': df['user_segment'].value_counts().to_dict()
    }
    
    with open('public/data_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[v0] Summary statistics saved")
    
    return df

# Main execution
if __name__ == "__main__":
    print("[v0] Starting data ingestion pipeline...")
    
    # Generate synthetic data
    df = generate_synthetic_data(n_samples=10000, random_state=42)
    
    # Clean data
    df = clean_data(df)
    
    # Save data
    df = save_data(df)
    
    print("[v0] Data ingestion complete!")
    print(f"[v0] Final dataset: {len(df)} records with {df['engaged'].sum()} engaged users")
