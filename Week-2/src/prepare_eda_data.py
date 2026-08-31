from pathlib import Path
import pandas as pd
import numpy as np

# =========================================================
# WEEK 2 - EDA DATA PREPARATION
# UCI ADULT DATASET
# =========================================================

BASE_DIR = Path("Week-2")
DATA_DIR = BASE_DIR / "data"

FEATURE_FILE = DATA_DIR / "adult_features_cleaned.csv"
TARGET_FILE = DATA_DIR / "adult_target_raw.csv"

OUTPUT_FILE = DATA_DIR / "adult_eda_ready.csv"

print("=" * 75)
print("WEEK 2 - EDA DATA PREPARATION")
print("=" * 75)

# ---------------------------------------------------------
# 1. Load feature and target data
# ---------------------------------------------------------

X = pd.read_csv(FEATURE_FILE)
y = pd.read_csv(TARGET_FILE)

print("\nInitial datasets")
print("-" * 75)
print(f"Feature shape : {X.shape}")
print(f"Target shape  : {y.shape}")

# ---------------------------------------------------------
# 2. IMPORTANT ALIGNMENT CHECK
# ---------------------------------------------------------

if len(X) != len(y):
    print("\nWARNING: Feature and target row counts differ.")
    print("The Week 1 feature-cleaning process removed duplicate rows.")
    print("For Week 2, target alignment must be handled explicitly.")

    # Reload original Week 1 raw feature data so that
    # features and target start with identical row positions.
    raw_feature_file = Path(
        "Week-1/data/raw/adult_features_raw.csv"
    )

    X = pd.read_csv(raw_feature_file)

    print("\nReloaded original raw features.")
    print(f"Raw feature shape: {X.shape}")

# ---------------------------------------------------------
# 3. Standardize column names
# ---------------------------------------------------------

X.columns = X.columns.str.strip()
y.columns = y.columns.str.strip()

# ---------------------------------------------------------
# 4. Standardize categorical values
# ---------------------------------------------------------

categorical_cols = X.select_dtypes(
    include=["object", "string"]
).columns

for col in categorical_cols:
    X[col] = (
        X[col]
        .astype("string")
        .str.strip()
        .replace("?", pd.NA)
    )

# ---------------------------------------------------------
# 5. Handle missing categorical values
# ---------------------------------------------------------

missing_before = X.isna().sum().sum()

for col in categorical_cols:
    if X[col].isna().any():
        mode_value = X[col].mode(dropna=True)[0]
        X[col] = X[col].fillna(mode_value)

missing_after = X.isna().sum().sum()

print("\nMissing-value treatment")
print("-" * 75)
print(f"Missing values before treatment: {missing_before}")
print(f"Missing values after treatment : {missing_after}")

# ---------------------------------------------------------
# 6. Normalize income target
# ---------------------------------------------------------

y["income"] = (
    y["income"]
    .astype("string")
    .str.strip()
    .str.replace(".", "", regex=False)
)

print("\nNormalized target labels:")
print(y["income"].value_counts())

# ---------------------------------------------------------
# 7. Combine features and target
# ---------------------------------------------------------

df = pd.concat(
    [
        X.reset_index(drop=True),
        y.reset_index(drop=True)
    ],
    axis=1
)

print("\nCombined dataset before duplicate removal:")
print(df.shape)

# ---------------------------------------------------------
# 8. Remove exact duplicate records
# ---------------------------------------------------------

duplicates = df.duplicated().sum()

df = (
    df
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"\nDuplicate complete records removed: {duplicates}")

# ---------------------------------------------------------
# 9. Final validation
# ---------------------------------------------------------

print("\nFINAL DATASET VALIDATION")
print("=" * 75)

print(f"Final shape              : {df.shape}")
print(f"Missing values           : {df.isna().sum().sum()}")
print(f"Duplicate rows           : {df.duplicated().sum()}")
print(f"Number of columns        : {df.shape[1]}")

print("\nTarget distribution:")
print(df["income"].value_counts())

print("\nNumerical columns:")
print(
    df.select_dtypes(include=np.number)
    .columns
    .tolist()
)

print("\nCategorical columns:")
print(
    df.select_dtypes(include=["object", "string"])
    .columns
    .tolist()
)

# ---------------------------------------------------------
# 10. Save EDA-ready dataset
# ---------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nEDA-ready dataset saved to:")
print(OUTPUT_FILE)

print("\nWeek 2 data preparation completed successfully.")