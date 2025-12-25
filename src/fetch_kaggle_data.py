"""
Kaggle Dataset Fetcher - Lightweight Version

Downloads datasets one at a time with minimal memory usage.
Run directly from command line to avoid system crashes.
"""

import os
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "kaggle"

# Small, verified working datasets (prioritized by size - smallest first)
DATASETS = [
    "jessicali9530/honey-production",           # 25 KB
    "usda/a-year-of-pumpkin-prices",            # 17 KB
    "elmoallistair/commodity-prices-19602021",  # 5 KB
    "thedevastator/food-prices-year-by-year",   # 7 KB
    "varshitanalluri/crop-price-prediction-dataset",  # 68 KB
    "sohier/weekly-dairy-product-prices",       # 60 KB
    "rajanand/rainfall-in-india",               # 500 KB
    "csafrit2/india-food-prices",               # 1.7 MB
    "jocelyndumlao/global-food-prices",         # 228 KB
    "ramkrijal/agriculture-vegetables-fruits-time-series-prices",  # 1.4 MB
]

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kaggle_bin = Path(sys.executable).parent / "kaggle"
    
    print(f"Downloading {len(DATASETS)} datasets to {DATA_DIR}\n")
    
    for i, ds in enumerate(DATASETS, 1):
        out_dir = DATA_DIR / ds.replace("/", "_")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[{i}/{len(DATASETS)}] {ds}")
        
        # Use os.system for simpler execution
        cmd = f'"{kaggle_bin}" datasets download -d {ds} -p "{out_dir}" --unzip -q 2>/dev/null'
        ret = os.system(cmd)
        
        if ret == 0:
            print("   ✓ Done")
        else:
            print("   ✗ Failed (skipping)")
    
    # Count results
    csv_files = list(DATA_DIR.rglob("*.csv"))
    total_mb = sum(f.stat().st_size for f in csv_files) / 1024 / 1024
    print(f"\n✓ Downloaded {len(csv_files)} files ({total_mb:.1f} MB)")

if __name__ == "__main__":
    main()
