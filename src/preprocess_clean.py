"""
Enhanced Data Preprocessing for Agricultural Commodity Price Prediction

This script creates a clean, well-preprocessed dataset for deep learning models.

Key improvements:
1. Filter to agricultural commodities only (remove wages, fuel, exchange rates)
2. Per-commodity robust normalization
3. Log-return transformation for stationarity
4. Comprehensive feature engineering
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "combined_all.csv"
OUTPUT_PATH = DATA_DIR / "processed_agricultural.csv"

# Agricultural commodities to keep (exclude wages, fuel, exchange rates, etc.)
AGRICULTURAL_COMMODITIES = [
    # Grains & Cereals
    'wheat', 'rice', 'maize', 'millet', 'sorghum', 'barley', 'corn',
    'wheat flour', 'rice (low quality)', 'rice (high quality)', 'rice (imported)',
    'maize (white)', 'maize (yellow)', 'maize flour',
    
    # Legumes & Pulses
    'beans', 'beans (dry)', 'lentils', 'peas', 'chickpeas', 'groundnuts',
    'cowpeas', 'pigeon peas',
    
    # Vegetables
    'potatoes', 'tomatoes', 'onions', 'cabbage', 'carrots', 'garlic',
    'peppers', 'eggplant', 'spinach', 'cucumber', 'cauliflower',
    'green beans', 'okra', 'pumpkin',
    
    # Fruits
    'bananas', 'apples', 'oranges', 'mangoes', 'grapes', 'papaya',
    
    # Oils & Fats
    'oil (vegetable)', 'oil (palm)', 'oil (sunflower)', 'oil (groundnut)',
    'cooking oil', 'palm oil', 'vegetable oil', 'sunflower oil',
    
    # Sugar & Sweeteners
    'sugar', 'sugar (brown)', 'sugar (white)', 'honey',
    
    # Dairy & Protein
    'milk', 'eggs', 'fish', 'meat', 'chicken', 'beef', 'goat',
    
    # Other Food Items
    'bread', 'pasta', 'cassava', 'yam', 'salt', 'tea', 'coffee',
    
    # Generic categories
    'food', 'vegetable', 'fruit', 'grain', 'cereal', 'pulse', 'legume'
]

# Items to explicitly exclude
EXCLUDE_PATTERNS = [
    'wage', 'fuel', 'diesel', 'petrol', 'gasoline', 'kerosene',
    'exchange', 'rate', 'transport', 'rent', 'labour', 'labor',
    'charcoal', 'firewood', 'electricity', 'water'
]


def is_agricultural(commodity_name):
    """Check if a commodity is agricultural based on name."""
    name_lower = commodity_name.lower().strip()
    
    # Exclude non-agricultural items
    for pattern in EXCLUDE_PATTERNS:
        if pattern in name_lower:
            return False
    
    # Include known agricultural items
    for agri_item in AGRICULTURAL_COMMODITIES:
        if agri_item in name_lower or name_lower in agri_item:
            return True
    
    # Default: include if not explicitly excluded (conservative approach)
    return True


def preprocess_data():
    """Main preprocessing pipeline."""
    print("\n" + "=" * 70)
    print("🌾 ENHANCED DATA PREPROCESSING FOR AGRICULTURAL PRICE PREDICTION")
    print("=" * 70)
    
    # Load raw data
    print("\n📊 Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date', 'price'])
    print(f"   Raw records: {len(df):,}")
    print(f"   Unique commodities: {df['commodity'].nunique()}")
    print(f"   Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Step 1: Filter to agricultural commodities only
    print("\n🔍 Step 1: Filtering to agricultural commodities...")
    df['commodity_clean'] = df['commodity'].str.lower().str.strip()
    
    # Check each commodity
    commodity_is_agri = df['commodity_clean'].apply(is_agricultural)
    df_agri = df[commodity_is_agri].copy()
    
    excluded = df[~commodity_is_agri]['commodity_clean'].unique()
    print(f"   ✓ Kept: {len(df_agri):,} records ({len(df_agri['commodity_clean'].unique())} commodities)")
    print(f"   ✗ Excluded: {len(excluded)} non-agricultural items")
    if len(excluded) > 0:
        print(f"     Examples excluded: {list(excluded[:5])}")
    
    df = df_agri
    
    # Step 2: Remove extreme outliers using IQR method
    print("\n🧹 Step 2: Removing outliers (IQR method)...")
    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # For prices, lower bound should be at least 0
    lower_bound = max(0, lower_bound)
    
    outlier_mask = (df['price'] >= lower_bound) & (df['price'] <= upper_bound)
    n_outliers = (~outlier_mask).sum()
    df = df[outlier_mask].copy()
    print(f"   ✓ Removed {n_outliers:,} outliers")
    print(f"   Price range now: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    
    # Step 3: Sort by date and reset index
    print("\n📅 Step 3: Sorting and organizing data...")
    df = df.sort_values(['commodity_clean', 'date']).reset_index(drop=True)
    
    # Step 4: Log transform for better distribution
    print("\n📈 Step 4: Applying log transformation...")
    df['price_log'] = np.log1p(df['price'])
    print(f"   Log price range: {df['price_log'].min():.4f} - {df['price_log'].max():.4f}")
    
    # Step 5: Per-commodity z-score normalization
    print("\n⚖️ Step 5: Per-commodity z-score normalization...")
    
    # Calculate per-commodity stats
    commodity_stats = df.groupby('commodity_clean')['price_log'].agg(['mean', 'std']).reset_index()
    commodity_stats.columns = ['commodity_clean', 'commodity_mean', 'commodity_std']
    commodity_stats['commodity_std'] = commodity_stats['commodity_std'].replace(0, 1)  # Avoid div by zero
    
    # Merge stats back
    df = df.merge(commodity_stats, on='commodity_clean', how='left')
    
    # Apply z-score normalization
    df['price_normalized'] = (df['price_log'] - df['commodity_mean']) / df['commodity_std']
    
    # Clip extreme z-scores
    df['price_normalized'] = df['price_normalized'].clip(-5, 5)
    print(f"   Normalized price range: {df['price_normalized'].min():.4f} - {df['price_normalized'].max():.4f}")
    
    # Step 6: Calculate log-returns (for stationarity)
    print("\n📉 Step 6: Computing log-returns...")
    
    # Group by commodity to calculate returns within each commodity
    df['log_return'] = df.groupby('commodity_clean')['price_log'].diff()
    df['log_return'] = df['log_return'].fillna(0)
    
    # Clip extreme returns
    df['log_return'] = df['log_return'].clip(-0.5, 0.5)  # ±50% max daily change
    print(f"   Log-return range: {df['log_return'].min():.4f} - {df['log_return'].max():.4f}")
    
    # Step 7: Feature engineering
    print("\n🔧 Step 7: Feature engineering...")
    
    # Rolling statistics per commodity
    for window in [7, 14, 30]:
        df[f'ma_{window}'] = df.groupby('commodity_clean')['price_normalized'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'std_{window}'] = df.groupby('commodity_clean')['price_normalized'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    # Fill NaN std values
    df[[f'std_{w}' for w in [7, 14, 30]]] = df[[f'std_{w}' for w in [7, 14, 30]]].fillna(0)
    
    # Momentum
    df['momentum'] = df['price_normalized'] - df['ma_7']
    
    # Temporal features (cyclical encoding)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofweek / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['date'].dt.dayofyear / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['date'].dt.dayofyear / 365)
    
    print(f"   ✓ Created rolling statistics (7, 14, 30 days)")
    print(f"   ✓ Created cyclical temporal features")
    
    # Step 8: One-hot encode top commodities
    print("\n🏷️ Step 8: Encoding commodities...")
    
    # Get top 20 commodities by record count
    top_commodities = df['commodity_clean'].value_counts().head(20).index.tolist()
    
    for comm in top_commodities:
        safe_name = comm.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        df[f'is_{safe_name}'] = (df['commodity_clean'] == comm).astype(float)
    
    print(f"   ✓ One-hot encoded top {len(top_commodities)} commodities")
    
    # Step 9: Clean up and save
    print("\n💾 Step 9: Saving processed data...")
    
    # Select final columns
    feature_cols = [
        'date', 'price', 'price_log', 'price_normalized', 'log_return',
        'commodity_clean', 'commodity_mean', 'commodity_std',
        'ma_7', 'ma_14', 'ma_30', 'std_7', 'std_14', 'std_30', 'momentum',
        'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos',
        'day_of_year_sin', 'day_of_year_cos'
    ]
    
    # Add commodity one-hot columns
    commodity_cols = [col for col in df.columns if col.startswith('is_')]
    feature_cols.extend(commodity_cols)
    
    # Filter to existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    df_final = df[feature_cols].copy()
    
    # Replace infinities and NaNs
    df_final = df_final.replace([np.inf, -np.inf], 0)
    df_final = df_final.fillna(0)
    
    # Save
    df_final.to_csv(OUTPUT_PATH, index=False)
    
    print(f"\n" + "=" * 70)
    print("✅ PREPROCESSING COMPLETE!")
    print("=" * 70)
    print(f"   📁 Output: {OUTPUT_PATH}")
    print(f"   📊 Records: {len(df_final):,}")
    print(f"   📋 Features: {len(feature_cols)}")
    print(f"   📅 Date range: {df_final['date'].min()} to {df_final['date'].max()}")
    print(f"   🌾 Commodities: {df_final['commodity_clean'].nunique()}")
    
    # Print commodity distribution
    print("\n📊 Top 10 Commodities by Record Count:")
    top_10 = df_final['commodity_clean'].value_counts().head(10)
    for comm, count in top_10.items():
        print(f"   {comm:25s}: {count:,} records")
    
    return df_final


if __name__ == "__main__":
    df = preprocess_data()
