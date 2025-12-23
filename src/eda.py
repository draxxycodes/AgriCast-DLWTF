"""
Exploratory Data Analysis Module

Generates comprehensive visualizations and statistics for the dataset.
Can be run standalone or as part of the main pipeline.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import config
from config import DATA_PATH, FIGURES_DIR, get_config


def load_raw_data() -> pd.DataFrame:
    """Load the raw dataset."""
    df = pd.read_csv(DATA_PATH)
    print(f"✓ Loaded {len(df)} records from {DATA_PATH}")
    return df


def basic_info(df: pd.DataFrame):
    """Print basic dataset information."""
    print("\n" + "=" * 50)
    print(" DATASET OVERVIEW")
    print("=" * 50)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def missing_values_analysis(df: pd.DataFrame):
    """Analyze and visualize missing values."""
    print("\n" + "=" * 50)
    print(" MISSING VALUES ANALYSIS")
    print("=" * 50)
    
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Count': missing,
        'Percentage': missing_pct
    }).sort_values('Count', ascending=False)
    
    print(missing_df[missing_df['Count'] > 0])
    
    if missing.sum() > 0:
        plt.figure(figsize=(10, 4))
        plt.bar(missing_df.index, missing_df['Percentage'])
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Missing %')
        plt.title('Missing Values by Column')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'missing_values.png'), dpi=150)
        plt.close()


def categorical_analysis(df: pd.DataFrame):
    """Analyze categorical columns."""
    print("\n" + "=" * 50)
    print(" CATEGORICAL ANALYSIS")
    print("=" * 50)
    
    cat_cols = df.select_dtypes(include=['object']).columns
    
    for col in cat_cols:
        print(f"\n{col}: {df[col].nunique()} unique values")
    
    # Top commodities
    plt.figure(figsize=(14, 6))
    top_commodities = df['Commodity'].value_counts().head(20)
    top_commodities.plot(kind='barh', color='steelblue', edgecolor='black')
    plt.xlabel('Number of Records')
    plt.title('Top 20 Commodities by Record Count')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'top_commodities.png'), dpi=150)
    plt.close()
    print("✓ Saved top_commodities.png")
    
    # State distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    state_counts = df['State'].value_counts().head(15)
    axes[0].barh(range(len(state_counts)), state_counts.values, color='coral')
    axes[0].set_yticks(range(len(state_counts)))
    axes[0].set_yticklabels(state_counts.index)
    axes[0].set_xlabel('Records')
    axes[0].set_title('Top 15 States')
    axes[0].invert_yaxis()
    
    top_5_states = df['State'].value_counts().head(5)
    axes[1].pie(top_5_states.values, labels=top_5_states.index, autopct='%1.1f%%')
    axes[1].set_title('Top 5 States Share')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'state_distribution.png'), dpi=150)
    plt.close()
    print("✓ Saved state_distribution.png")


def price_analysis(df: pd.DataFrame):
    """Analyze price distributions."""
    print("\n" + "=" * 50)
    print(" PRICE DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    price_cols = ['Min Price', 'Max Price', 'Modal Price']
    
    # Convert to numeric
    for col in price_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    print(df[price_cols].describe())
    
    # Distribution plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for idx, col in enumerate(price_cols):
        data = df[col].dropna()
        data = data[(data > 0) & (data < data.quantile(0.99))]
        
        axes[idx].hist(data, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[idx].axvline(data.mean(), color='red', linestyle='--', label=f'Mean: {data.mean():.0f}')
        axes[idx].axvline(data.median(), color='green', linestyle='-', label=f'Median: {data.median():.0f}')
        axes[idx].set_xlabel('Price (INR per Quintal)')
        axes[idx].set_ylabel('Frequency')
        axes[idx].set_title(f'{col} Distribution')
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'price_distributions.png'), dpi=150)
    plt.close()
    print("✓ Saved price_distributions.png")
    
    # Box plots by commodity
    top_commodities = df['Commodity'].value_counts().head(10).index.tolist()
    df_top = df[df['Commodity'].isin(top_commodities)].copy()
    
    plt.figure(figsize=(14, 8))
    df_top.boxplot(column='Modal Price', by='Commodity', vert=True)
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Commodity')
    plt.ylabel('Modal Price (INR per Quintal)')
    plt.title('Price Distribution by Top 10 Commodities')
    plt.suptitle('')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'price_by_commodity.png'), dpi=150)
    plt.close()
    print("✓ Saved price_by_commodity.png")


def time_series_analysis(df: pd.DataFrame, commodity: str = 'Tomato'):
    """Analyze time series patterns for a specific commodity."""
    print("\n" + "=" * 50)
    print(f" TIME SERIES ANALYSIS - {commodity}")
    print("=" * 50)
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Arrival_Date'], format='%d-%m-%Y', errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Filter commodity
    commodity_df = df[df['Commodity'] == commodity].copy()
    print(f"\nRecords for {commodity}: {len(commodity_df)}")
    
    if len(commodity_df) < 30:
        print(f"⚠ Not enough data for {commodity}, trying Onion...")
        commodity = 'Onion'
        commodity_df = df[df['Commodity'] == commodity].copy()
        print(f"Records for {commodity}: {len(commodity_df)}")
    
    # Aggregate daily
    daily = commodity_df.groupby('Date')['Modal Price'].mean().reset_index()
    daily = daily.sort_values('Date')
    daily.columns = ['date', 'price']
    
    # Fill gaps
    date_range = pd.date_range(start=daily['date'].min(), end=daily['date'].max(), freq='D')
    daily = daily.set_index('date').reindex(date_range).interpolate().reset_index()
    daily.columns = ['date', 'price']
    
    print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")
    
    # Price trend
    plt.figure(figsize=(16, 6))
    plt.plot(daily['date'], daily['price'], color='steelblue', linewidth=1.5)
    plt.fill_between(daily['date'], daily['price'], alpha=0.3)
    plt.xlabel('Date')
    plt.ylabel('Modal Price (INR per Quintal)')
    plt.title(f'{commodity} - Daily Price Trend')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'price_trend.png'), dpi=150)
    plt.close()
    print("✓ Saved price_trend.png")
    
    # Moving averages
    daily['MA_7'] = daily['price'].rolling(window=7).mean()
    daily['MA_30'] = daily['price'].rolling(window=30).mean()
    
    plt.figure(figsize=(16, 6))
    plt.plot(daily['date'], daily['price'], label='Daily Price', alpha=0.5, linewidth=1)
    plt.plot(daily['date'], daily['MA_7'], label='7-Day MA', linewidth=2, color='orange')
    plt.plot(daily['date'], daily['MA_30'], label='30-Day MA', linewidth=2, color='red')
    plt.xlabel('Date')
    plt.ylabel('Price (INR per Quintal)')
    plt.title(f'{commodity} - Price with Moving Averages')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'moving_averages.png'), dpi=150)
    plt.close()
    print("✓ Saved moving_averages.png")
    
    # Monthly/Weekly patterns
    daily['month'] = daily['date'].dt.month
    daily['day_of_week'] = daily['date'].dt.dayofweek
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    monthly_avg = daily.groupby('month')['price'].mean()
    axes[0].bar(monthly_avg.index, monthly_avg.values, color='steelblue')
    axes[0].set_xlabel('Month')
    axes[0].set_ylabel('Avg Price')
    axes[0].set_title('Average Price by Month')
    axes[0].set_xticks(range(1, 13))
    
    weekly_avg = daily.groupby('day_of_week')['price'].mean()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1].bar([days[i] for i in weekly_avg.index], weekly_avg.values, color='coral')
    axes[1].set_xlabel('Day of Week')
    axes[1].set_ylabel('Avg Price')
    axes[1].set_title('Average Price by Day of Week')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'seasonal_patterns.png'), dpi=150)
    plt.close()
    print("✓ Saved seasonal_patterns.png")
    
    return daily


def correlation_analysis(df: pd.DataFrame):
    """Analyze correlations between numeric columns."""
    print("\n" + "=" * 50)
    print(" CORRELATION ANALYSIS")
    print("=" * 50)
    
    numeric_cols = ['Min Price', 'Max Price', 'Modal Price']
    
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    corr_matrix = df[numeric_cols].corr()
    print(corr_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.3f', square=True)
    plt.title('Price Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_matrix.png'), dpi=150)
    plt.close()
    print("✓ Saved correlation_matrix.png")


def generate_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics report."""
    print("\n" + "=" * 50)
    print(" SUMMARY STATISTICS")
    print("=" * 50)
    
    summary = {
        'Total Records': len(df),
        'Unique Commodities': df['Commodity'].nunique(),
        'Unique States': df['State'].nunique(),
        'Unique Markets': df['Market'].nunique(),
        'Date Range': f"{df['Arrival_Date'].min()} to {df['Arrival_Date'].max()}",
        'Avg Modal Price': df['Modal Price'].mean(),
        'Price Std Dev': df['Modal Price'].std(),
        'Min Price': df['Modal Price'].min(),
        'Max Price': df['Modal Price'].max()
    }
    
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    
    return pd.DataFrame([summary])


def run_full_eda(config=None):
    """Run complete EDA pipeline."""
    if config is None:
        config = get_config()
    
    print("\n" + "=" * 60)
    print(" 📊 EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Load data
    df = load_raw_data()
    
    # Basic info
    basic_info(df)
    
    # Missing values
    missing_values_analysis(df)
    
    # Categorical
    categorical_analysis(df)
    
    # Price analysis
    price_analysis(df)
    
    # Time series
    time_series_analysis(df, commodity=config.data.commodity)
    
    # Correlations
    correlation_analysis(df)
    
    # Summary
    summary_df = generate_summary_stats(df)
    summary_df.to_csv(os.path.join(FIGURES_DIR, '../reports/eda_summary.csv'), index=False)
    
    print("\n" + "=" * 60)
    print(f" ✓ EDA COMPLETE - Figures saved to {FIGURES_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    run_full_eda()
