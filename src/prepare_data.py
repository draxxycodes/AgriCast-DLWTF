"""
Dataset Preparation Script

Combines all downloaded datasets into a unified format for training.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path(__file__).parent.parent / "data"
KAGGLE_DIR = DATA_DIR / "kaggle"
OUTPUT_DIR = DATA_DIR / "combined"


def load_original_dataset():
    """Load the original agricultural prices dataset."""
    path = DATA_DIR / "original" / "Price_Agriculture_commodities_Week.csv"
    if path.exists():
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
        
        # Standardize columns
        df['date'] = pd.to_datetime(df['arrival_date'], format='%d-%m-%Y', errors='coerce')
        df['price'] = pd.to_numeric(df['modal_price'], errors='coerce')
        df['commodity'] = df['commodity'].str.lower().str.strip()
        df['source'] = 'original_india'
        
        print(f"  Original: {len(df):,} rows")
        return df[['date', 'price', 'commodity', 'source']].dropna()
    return pd.DataFrame()


def load_india_food():
    """Load India food prices dataset."""
    path = KAGGLE_DIR / "india_food" / "wfp_food_prices_ind.csv"
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.lower().str.strip()
        
        # Find date and price columns
        date_col = [c for c in df.columns if 'date' in c]
        price_col = [c for c in df.columns if 'price' in c]
        commodity_col = [c for c in df.columns if 'commodity' in c or 'item' in c]
        
        if date_col and price_col:
            result = pd.DataFrame()
            result['date'] = pd.to_datetime(df[date_col[0]], errors='coerce')
            result['price'] = pd.to_numeric(df[price_col[0]], errors='coerce')
            result['commodity'] = df[commodity_col[0]].str.lower() if commodity_col else 'unknown'
            result['source'] = 'india_food'
            print(f"  India food: {len(result):,} rows")
            return result.dropna()
    return pd.DataFrame()


def load_veg_fruits():
    """Load vegetables and fruits time series."""
    path = KAGGLE_DIR / "veg_fruits" / "kalimati_tarkari_dataset.csv"
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.lower().str.strip()
        
        date_col = [c for c in df.columns if 'date' in c]
        price_col = [c for c in df.columns if 'price' in c or 'average' in c]
        commodity_col = [c for c in df.columns if 'commodity' in c or 'item' in c or 'product' in c]
        
        if date_col and price_col:
            result = pd.DataFrame()
            result['date'] = pd.to_datetime(df[date_col[0]], errors='coerce')
            result['price'] = pd.to_numeric(df[price_col[0]], errors='coerce')
            result['commodity'] = df[commodity_col[0]].str.lower() if commodity_col else 'vegetable'
            result['source'] = 'veg_fruits'
            print(f"  Veg/Fruits: {len(result):,} rows")
            return result.dropna()
    return pd.DataFrame()


def load_wfp_global():
    """Load WFP global food prices."""
    path = KAGGLE_DIR / "wfp_food" / "wfp_market_food_prices.csv"
    if path.exists():
        try:
            df = pd.read_csv(path, low_memory=False, encoding='latin-1')
        except:
            df = pd.read_csv(path, low_memory=False, encoding='utf-8', errors='ignore')
        
        df.columns = df.columns.str.lower().str.strip()
        
        date_col = [c for c in df.columns if 'date' in c]
        price_col = [c for c in df.columns if 'price' in c]
        commodity_col = [c for c in df.columns if 'commodity' in c or 'cm_name' in c]
        
        if date_col and price_col:
            result = pd.DataFrame()
            result['date'] = pd.to_datetime(df[date_col[0]], errors='coerce')
            result['price'] = pd.to_numeric(df[price_col[0]], errors='coerce')
            result['commodity'] = df[commodity_col[0]].str.lower() if commodity_col else 'food'
            result['source'] = 'wfp_global'
            print(f"  WFP Global: {len(result):,} rows")
            return result.dropna()
    return pd.DataFrame()


def load_commodity_prices():
    """Load commodity prices datasets."""
    dfs = []
    
    # Commodity prices 1960-2021
    path = KAGGLE_DIR / "commodity_prices" / "commodity_prices.csv"
    if path.exists():
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()
        if 'date' in df.columns:
            result = pd.DataFrame()
            result['date'] = pd.to_datetime(df['date'], errors='coerce')
            price_cols = [c for c in df.columns if c not in ['date', 'unnamed']]
            if price_cols:
                # Melt to long format
                for col in price_cols[:5]:  # Take first 5 commodities
                    temp = pd.DataFrame({
                        'date': result['date'],
                        'price': pd.to_numeric(df[col], errors='coerce'),
                        'commodity': col,
                        'source': 'commodity_historical'
                    }).dropna()
                    dfs.append(temp)
    
    # Commodity 2000-2023
    path = KAGGLE_DIR / "commodity_2000_2023" / "commodity_futures.csv"
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.lower().str.strip()
        date_col = [c for c in df.columns if 'date' in c]
        price_col = [c for c in df.columns if 'close' in c or 'price' in c]
        name_col = [c for c in df.columns if 'name' in c or 'symbol' in c]
        
        if date_col and price_col:
            result = pd.DataFrame()
            result['date'] = pd.to_datetime(df[date_col[0]], errors='coerce')
            result['price'] = pd.to_numeric(df[price_col[0]], errors='coerce')
            result['commodity'] = df[name_col[0]].str.lower() if name_col else 'commodity'
            result['source'] = 'commodity_futures'
            dfs.append(result.dropna())
    
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        print(f"  Commodities: {len(combined):,} rows")
        return combined
    return pd.DataFrame()


def load_crop_price():
    """Load crop price prediction dataset."""
    path = KAGGLE_DIR / "crop_price" / "Crop_Yield_Prediction.csv"
    if path.exists():
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()
        
        # This dataset might have different structure
        year_col = [c for c in df.columns if 'year' in c]
        price_col = [c for c in df.columns if 'price' in c or 'production' in c]
        crop_col = [c for c in df.columns if 'crop' in c or 'item' in c]
        
        if year_col and price_col:
            result = pd.DataFrame()
            result['date'] = pd.to_datetime(df[year_col[0]].astype(str) + '-01-01', errors='coerce')
            result['price'] = pd.to_numeric(df[price_col[0]], errors='coerce')
            result['commodity'] = df[crop_col[0]].str.lower() if crop_col else 'crop'
            result['source'] = 'crop_prediction'
            print(f"  Crop prices: {len(result):,} rows")
            return result.dropna()
    return pd.DataFrame()


def prepare_combined_dataset():
    """Combine all datasets into unified format."""
    print("\n" + "=" * 60)
    print("📊 PREPARING COMBINED DATASET")
    print("=" * 60)
    print("\nLoading datasets...")
    
    datasets = []
    
    # Load each dataset
    datasets.append(load_original_dataset())
    datasets.append(load_india_food())
    datasets.append(load_veg_fruits())
    datasets.append(load_wfp_global())
    datasets.append(load_commodity_prices())
    datasets.append(load_crop_price())
    
    # Combine all
    datasets = [df for df in datasets if len(df) > 0]
    combined = pd.concat(datasets, ignore_index=True)
    
    print(f"\n✓ Combined: {len(combined):,} total rows")
    print(f"✓ Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"✓ Unique commodities: {combined['commodity'].nunique()}")
    print(f"✓ Sources: {combined['source'].unique().tolist()}")
    
    # Remove outliers
    Q1, Q99 = combined['price'].quantile([0.01, 0.99])
    combined = combined[(combined['price'] >= Q1) & (combined['price'] <= Q99)]
    print(f"✓ After outlier removal: {len(combined):,} rows")
    
    # Save combined dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "combined_prices.csv"
    combined.to_csv(output_path, index=False)
    print(f"\n✓ Saved to: {output_path}")
    
    # Create daily aggregated version
    daily = combined.groupby('date').agg({
        'price': 'mean',
        'commodity': 'count'
    }).reset_index()
    daily.columns = ['date', 'price', 'n_records']
    daily = daily.sort_values('date')
    
    daily_path = OUTPUT_DIR / "daily_prices.csv"
    daily.to_csv(daily_path, index=False)
    print(f"✓ Daily aggregated: {len(daily):,} days -> {daily_path}")
    
    print("\n" + "=" * 60)
    
    return combined, daily


if __name__ == "__main__":
    combined, daily = prepare_combined_dataset()
    
    print("\n📊 Dataset Summary:")
    print(f"   Total records: {len(combined):,}")
    print(f"   Daily records: {len(daily):,}")
    print(f"   Date range: {daily['date'].min()} to {daily['date'].max()}")
    print(f"   Price range: ${daily['price'].min():.2f} - ${daily['price'].max():.2f}")
