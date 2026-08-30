from pathlib import Path
import pandas as pd
import numpy as np

RAW_FILE = Path("data/raw/adult_features_raw.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "adult_features_cleaned.csv"

print("=" * 70)
print("WEEK 1 - DATA CLEANING")
print("=" * 70)

# Load raw data
df = pd.read_csv(RAW_FILE)

print(f"\nInitial shape: {df.shape}")

# -------------------------------------------------------
# 1. Standardize categorical text
# -------------------------------------------------------
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

for col in categorical_cols:
    df[col] = df[col].astype("string").str.strip()

print("\nStep 1: Standardized categorical text.")

# -------------------------------------------------------
# 2. Convert ? to missing values
# -------------------------------------------------------
question_marks = (df == "?").sum().sum()
df = df.replace("?", np.nan)

print(f"Step 2: Replaced {question_marks} '?' indicators with NaN.")

# -------------------------------------------------------
# 3. Handle missing categorical values using mode
# -------------------------------------------------------
missing_before = df.isna().sum().sum()

for col in categorical_cols:
    if df[col].isna().any():
        mode_value = df[col].mode(dropna=True)[0]
        missing_count = df[col].isna().sum()
        df[col] = df[col].fillna(mode_value)
        print(
            f"Step 3: Filled {missing_count} missing values "
            f"in '{col}' with mode: {mode_value}"
        )

print(f"Missing values before imputation: {missing_before}")
print(f"Missing values after imputation: {df.isna().sum().sum()}")

# -------------------------------------------------------
# 4. Remove exact duplicate rows
# -------------------------------------------------------
duplicates_before = df.duplicated().sum()
df = df.drop_duplicates().reset_index(drop=True)

print(f"\nStep 4: Removed {duplicates_before} exact duplicate rows.")

# -------------------------------------------------------
# 5. Final validation
# -------------------------------------------------------
print("\nFINAL CLEANING VALIDATION")
print("-" * 70)
print(f"Final shape: {df.shape}")
print(f"Remaining missing values: {df.isna().sum().sum()}")
print(f"Remaining duplicate rows: {df.duplicated().sum()}")

# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)

print(f"\nCleaned dataset saved to: {OUTPUT_FILE}")
print("\nData cleaning completed successfully.")
