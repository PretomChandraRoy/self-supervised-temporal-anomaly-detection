"""
Debug script to check data scaling
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.preprocessing import FinancialDataPreprocessor, load_forex_data
import numpy as np

print("="*80)
print("DATA SCALING DEBUG")
print("="*80)

# Load data
print("\n[1] Loading data...")
df = load_forex_data('../forexPredictor/H4_EURUSD_2015.csv')
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")

# Check for problematic columns
print("\n[2] Checking for large values...")
for col in df.select_dtypes(include=[np.number]).columns:
    col_min = df[col].min()
    col_max = df[col].max()
    col_mean = df[col].mean()

    if abs(col_max) > 1000 or abs(col_min) > 1000:
        print(f"⚠️  {col:20s}: min={col_min:15.2f}, max={col_max:15.2f}, mean={col_mean:15.2f} <- LARGE VALUES!")
    else:
        print(f"✓  {col:20s}: min={col_min:15.6f}, max={col_max:15.6f}, mean={col_mean:15.6f}")

# Preprocess
print("\n[3] Preprocessing with fixes...")
preprocessor = FinancialDataPreprocessor(
    window_size=60,
    stride=1,
    add_technical_indicators=True,
    scaler_type='robust',
    clip_outliers=True
)

sequences, feature_names = preprocessor.prepare_data(df, fit_scaler=True)

print(f"\n[4] Final check:")
print(f"Sequences shape: {sequences.shape}")
print(f"Features used: {len(feature_names)}")
print(f"Data range: [{sequences.min():.4f}, {sequences.max():.4f}]")
print(f"Data mean: {sequences.mean():.4f}")
print(f"Data std: {sequences.std():.4f}")

if abs(sequences.max()) > 100 or abs(sequences.min()) > 100:
    print("\n❌ PROBLEM: Data still has large values after scaling!")
    print("   This will cause reconstruction loss explosion.")
else:
    print("\n✅ SUCCESS: Data is properly scaled!")
    print("   Should be safe for training.")

print("="*80)

