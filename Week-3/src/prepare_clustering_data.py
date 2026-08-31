from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

# ============================================================
# WEEK 3 - CLUSTERING DATA PREPARATION
# ============================================================

INPUT_FILE = Path("Week-3/data/adult_eda_ready.csv")
OUTPUT_FILE = Path("Week-3/data/adult_clustering_ready.csv")

print("=" * 75)
print("WEEK 3 - UNSUPERVISED LEARNING & CLUSTERING")
print("=" * 75)

# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nDATASET OVERVIEW")
print("-" * 75)
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicate rows: {df.duplicated().sum()}")

# ------------------------------------------------------------
# 2. Select numerical features
# ------------------------------------------------------------
# For the first clustering model, we use measurable
# numerical characteristics rather than the income target.

numerical_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

X = df[numerical_features].copy()

print("\nSELECTED CLUSTERING FEATURES")
print("-" * 75)

for column in numerical_features:
    print(f"- {column}")

# ------------------------------------------------------------
# 3. Standardize numerical features
# ------------------------------------------------------------

print("\nSTANDARDIZATION")
print("-" * 75)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=numerical_features
)

print("Standardization completed.")

print("\nScaled feature means:")
print(X_scaled_df.mean().round(4))

print("\nScaled feature standard deviations:")
print(X_scaled_df.std().round(4))

# ------------------------------------------------------------
# 4. Save clustering-ready data
# ------------------------------------------------------------

X_scaled_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nCLUSTERING DATA VALIDATION")
print("-" * 75)
print(f"Rows: {X_scaled_df.shape[0]}")
print(f"Columns: {X_scaled_df.shape[1]}")
print(f"Missing values: {X_scaled_df.isnull().sum().sum()}")
print(f"Duplicate rows: {X_scaled_df.duplicated().sum()}")

print(f"\nClustering-ready dataset saved to:")
print(OUTPUT_FILE)

print("\nWeek 3 data preparation completed successfully.")