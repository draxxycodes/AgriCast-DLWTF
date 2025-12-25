"""
Combine All Datasets - Fixed Version

Handles all column naming conventions properly.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(__file__).parent.parent / "data"
KAGGLE_DIR = DATA_DIR / "kaggle"

def read_csv_safe(path):
    """Read CSV with multiple encoding attempts."""
    for encoding in ['utf-8', 'latin-1', 'cp1252']:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, on_bad_lines='skip')
        except:
            continue
    return pd.DataFrame()


def process_wfp_global(path):
    """Process WFP global format: mp_year, mp_month, mp_price."""
    df = read_csv_safe(path)
    if 'mp_price' in df.columns and 'mp_year' in df.columns:
        df['date'] = pd.to_datetime(df['mp_year'].astype(str) + '-' + df['mp_month'].astype(str) + '-15', errors='coerce')
        result = pd.DataFrame({
            'date': df['date'],
            'price': pd.to_numeric(df['mp_price'], errors='coerce'),
            'commodity': df['cm_name'].str.lower() if 'cm_name' in df.columns else 'food',
            'source': 'wfp_global'
        })
        return result.dropna()
    return pd.DataFrame()


def process_india_food(path):
    """Process India food format with 'price' column."""
    df = read_csv_safe(path)
    # Skip header row if present
    if df.iloc[0, 0].startswith('#'):
        df = df.iloc[1:]
    
    if 'price' in df.columns and 'date' in df.columns:
        result = pd.DataFrame({
            'date': pd.to_datetime(df['date'], errors='coerce'),
            'price': pd.to_numeric(df['price'], errors='coerce'),
            'commodity': df['commodity'].str.lower() if 'commodity' in df.columns else 'food',
            'source': 'india_wfp'
        })
        return result.dropna()
    return pd.DataFrame()


def process_commodity_wide(path):
    """Process wide format commodity data (Date, GOLD, SILVER, etc.)."""
    df = read_csv_safe(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        price_cols = [c for c in df.columns if c != 'Date']
        
        records = []
        for col in price_cols:
            temp = pd.DataFrame({
                'date': df['Date'],
                'price': pd.to_numeric(df[col], errors='coerce'),
                'commodity': col.lower(),
                'source': 'commodity_futures'
            })
            records.append(temp.dropna())
        
        if records:
            return pd.concat(records, ignore_index=True)
    return pd.DataFrame()


def process_agriculture(path):
    """Process agriculture dataset with Modal Price."""
    df = read_csv_safe(path)
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    
    if 'modal_price' in df.columns:
        date_col = 'arrival_date' if 'arrival_date' in df.columns else 'date'
        result = pd.DataFrame({
            'date': pd.to_datetime(df[date_col], format='%d-%m-%Y', errors='coerce'),
            'price': pd.to_numeric(df['modal_price'], errors='coerce'),
            'commodity': df['commodity'].str.lower() if 'commodity' in df.columns else 'agriculture',
            'source': 'agriculture_india'
        })
        return result.dropna()
    return pd.DataFrame()


def process_kalimati(path):
    """Process Kalimati vegetable/fruits dataset."""
    df = read_csv_safe(path)
    df.columns = df.columns.str.lower().str.strip()
    
    price_col = None
    for col in df.columns:
        if 'average' in col or 'price' in col:
            price_col = col
            break
    
    date_col = None
    for col in df.columns:
        if 'date' in col:
            date_col = col
            break
    
    if price_col and date_col:
        commodity_col = [c for c in df.columns if 'commodity' in c or 'item' in c]
        result = pd.DataFrame({
            'date': pd.to_datetime(df[date_col], errors='coerce'),
            'price': pd.to_numeric(df[price_col], errors='coerce'),
            'commodity': df[commodity_col[0]].str.lower() if commodity_col else 'vegetable',
            'source': 'kalimati_nepal'
        })
        return result.dropna()
    return pd.DataFrame()


def main():
    print("\n" + "=" * 60)
    print("📊 COMBINING ALL DATASETS (FIXED)")
    print("=" * 60)
    
    all_data = []
    
    # Process each dataset type
    processors = [
        (KAGGLE_DIR / "wfp" / "wfp_market_food_prices.csv", process_wfp_global),
        (KAGGLE_DIR / "india_food" / "wfp_food_prices_ind.csv", process_india_food),
        (KAGGLE_DIR / "food_india" / "food_prices_ind.csv", process_india_food),
        (KAGGLE_DIR / "commodity" / "commodity_futures.csv", process_commodity_wide),
        (KAGGLE_DIR / "veg_fruits" / "kalimati_tarkari_dataset.csv", process_kalimati),
        (KAGGLE_DIR / "daily_india" / "Price_Agriculture_commodities_Week.csv", process_agriculture),
        (KAGGLE_DIR / "original" / "Price_Agriculture_commodities_Week.csv", process_agriculture),
    ]
    
    for path, processor in processors:
        if path.exists():
            print(f"Processing: {path.name}...")
            result = processor(path)
            if len(result) > 0:
                print(f"   ✓ Extracted {len(result):,} rows")
                all_data.append(result)
            else:
                print(f"   ⚠ No data")
        else:
            print(f"Missing: {path}")
    
    if not all_data:
        print("❌ No data!")
        return
    
    # Combine
    combined = pd.concat(all_data, ignore_index=True)
    combined = combined.drop_duplicates(subset=['date', 'price', 'commodity'])
    
    print(f"\n✓ Combined: {len(combined):,} rows")
    print(f"✓ Date range: {combined['date'].min()} to {combined['date'].max()}")
    print(f"✓ Commodities: {combined['commodity'].nunique()}")
    print(f"✓ Sources: {combined['source'].unique().tolist()}")
    
    # Remove outliers
    Q05, Q95 = combined['price'].quantile([0.005, 0.995])
    combined = combined[(combined['price'] > 0) & (combined['price'] >= Q05) & (combined['price'] <= Q95)]
    print(f"✓ After cleanup: {len(combined):,} rows")
    
    # Save
    output = DATA_DIR / "combined_all.csv"
    combined.to_csv(output, index=False)
    size = output.stat().st_size / 1024 / 1024
    print(f"\n✓ Saved: {output} ({size:.1f} MB)")
    
    # Daily aggregation
    daily = combined.groupby('date').agg({'price': 'mean', 'commodity': 'count'}).reset_index()
    daily.columns = ['date', 'price', 'n_records']
    daily = daily.sort_values('date')
    daily.to_csv(DATA_DIR / "daily_prices.csv", index=False)
    print(f"✓ Daily: {len(daily):,} days")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
