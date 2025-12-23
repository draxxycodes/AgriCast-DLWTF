"""
Kaggle Dataset Fetcher for Agricultural Commodity Price Prediction

This script fetches multiple agricultural price datasets from Kaggle
and combines them into a comprehensive training dataset.

Requirements:
    pip install kaggle
    Set KAGGLE_USERNAME and KAGGLE_KEY environment variables
    Or place kaggle.json in ~/.kaggle/

Usage:
    python fetch_kaggle_data.py
"""

import os
import sys
import subprocess
import zipfile
import shutil
import pandas as pd
from pathlib import Path

# Kaggle datasets for agricultural commodity prices
KAGGLE_DATASETS = [
    # India Agricultural Price Data
    "srinuti/agriculture-indian-dataset",
    "karthikbhandary2/agriculture-commodities-price-index",
    "varshitanalluri/agricultural-commodities-price-in-india",
    
    # India Crops & Agriculture
    "akshatgupta7/crop-production-in-india",
    "abhinand05/crop-production-in-india",
    
    # Global Agricultural Data
    "sercanyesiloz/turkey-price-index",
    "unitednations/global-food-agriculture-statistics",
    
    # Commodity Prices
    "gauravduttakiit/commodity-price-prediction",
    "debashis74017/stock-market-data-nifty-50-stocks-1-min-data",
    
    # Wholesale Price Index
    "mexwell/wholesale-price-index-india",
]

# Output directories
DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
COMBINED_DATA_DIR = DATA_DIR / "combined"


def check_kaggle_setup():
    """Check if Kaggle API is properly set up."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if not kaggle_json.exists():
        print("⚠️  Kaggle API not configured!")
        print("   Please follow these steps:")
        print("   1. Go to https://www.kaggle.com/account")
        print("   2. Click 'Create New API Token' to download kaggle.json")
        print("   3. Place kaggle.json in ~/.kaggle/")
        print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False
    
    # Set correct permissions
    os.chmod(kaggle_json, 0o600)
    return True


def download_dataset(dataset_name, output_dir):
    """Download a Kaggle dataset."""
    try:
        print(f"\n📥 Downloading: {dataset_name}")
        cmd = [
            "kaggle", "datasets", "download",
            "-d", dataset_name,
            "-p", str(output_dir),
            "--unzip"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"   ✓ Downloaded successfully")
            return True
        else:
            print(f"   ⚠ Download failed: {result.stderr[:100]}")
            return False
    except Exception as e:
        print(f"   ⚠ Error: {str(e)[:100]}")
        return False


def find_csv_files(directory):
    """Find all CSV files in a directory."""
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files


def standardize_columns(df):
    """Standardize column names across different datasets."""
    # Common column mappings
    column_mappings = {
        # Date columns
        'arrival_date': 'date',
        'arrival date': 'date',
        'date': 'date',
        'year': 'year',
        'month': 'month',
        
        # Price columns
        'modal_price': 'price',
        'modal price': 'price',
        'price': 'price',
        'modal_price_rs_quintal': 'price',
        'wholesale_price': 'price',
        'avg_price': 'price',
        
        # Location columns
        'state': 'state',
        'state_name': 'state',
        'district': 'district',
        'district_name': 'district',
        'market': 'market',
        
        # Commodity columns
        'commodity': 'commodity',
        'crop': 'commodity',
        'item': 'commodity',
        'product': 'commodity',
    }
    
    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    df = df.rename(columns=column_mappings)
    
    return df


def process_and_combine_datasets(raw_dir, output_dir):
    """Process all downloaded datasets and combine them."""
    print("\n📊 Processing datasets...")
    
    csv_files = find_csv_files(raw_dir)
    print(f"   Found {len(csv_files)} CSV files")
    
    all_dataframes = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            df = standardize_columns(df)
            
            # Check for required columns
            if 'price' in df.columns or 'modal_price' in df.columns:
                df['source_file'] = os.path.basename(csv_file)
                all_dataframes.append(df)
                print(f"   ✓ Processed: {os.path.basename(csv_file)} ({len(df)} rows)")
        except Exception as e:
            print(f"   ⚠ Skipped: {os.path.basename(csv_file)} - {str(e)[:50]}")
    
    if not all_dataframes:
        print("   ⚠ No valid dataframes found")
        return None
    
    # Combine all dataframes
    print("\n🔗 Combining datasets...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Save combined dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "combined_agricultural_data.csv"
    combined_df.to_csv(output_path, index=False)
    
    print(f"   ✓ Combined dataset: {len(combined_df)} rows")
    print(f"   ✓ Saved to: {output_path}")
    
    return combined_df


def create_research_dataset(combined_df, output_dir):
    """Create a cleaned research-grade dataset."""
    print("\n🔬 Creating research-grade dataset...")
    
    df = combined_df.copy()
    
    # Standardize price column
    if 'price' not in df.columns and 'modal_price' in df.columns:
        df['price'] = df['modal_price']
    
    # Convert price to numeric
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[df['price'].notna()]
        df = df[df['price'] > 0]
    
    # Handle date columns
    date_columns = ['date', 'arrival_date']
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Remove extreme outliers (price > 99th percentile or < 1st percentile)
    if 'price' in df.columns:
        q1, q99 = df['price'].quantile([0.01, 0.99])
        df = df[(df['price'] >= q1) & (df['price'] <= q99)]
    
    # Save research dataset
    output_path = output_dir / "research_agricultural_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"   ✓ Research dataset: {len(df)} rows")
    print(f"   ✓ Saved to: {output_path}")
    
    # Print statistics
    print("\n📈 Dataset Statistics:")
    if 'commodity' in df.columns:
        print(f"   Unique commodities: {df['commodity'].nunique()}")
    if 'state' in df.columns:
        print(f"   Unique states: {df['state'].nunique()}")
    if 'date' in df.columns:
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    if 'price' in df.columns:
        print(f"   Price range: {df['price'].min():.2f} to {df['price'].max():.2f}")
    
    return df


def main():
    """Main function to fetch and process Kaggle datasets."""
    print("\n" + "=" * 60)
    print("🌾 KAGGLE AGRICULTURAL DATA FETCHER")
    print("=" * 60)
    
    # Create directories
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    COMBINED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check Kaggle setup
    if not check_kaggle_setup():
        print("\n❌ Please configure Kaggle API and retry")
        return
    
    # Download datasets
    print("\n📥 Downloading datasets from Kaggle...")
    successful_downloads = 0
    
    for dataset in KAGGLE_DATASETS:
        dataset_dir = RAW_DATA_DIR / dataset.replace("/", "_")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        if download_dataset(dataset, dataset_dir):
            successful_downloads += 1
    
    print(f"\n✓ Downloaded {successful_downloads}/{len(KAGGLE_DATASETS)} datasets")
    
    # Process and combine
    combined_df = process_and_combine_datasets(RAW_DATA_DIR, COMBINED_DATA_DIR)
    
    if combined_df is not None:
        # Create research dataset
        research_df = create_research_dataset(combined_df, COMBINED_DATA_DIR)
    
    print("\n" + "=" * 60)
    print("✓ DATA FETCHING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
