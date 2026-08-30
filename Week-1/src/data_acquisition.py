from pathlib import Path
from ucimlrepo import fetch_ucirepo

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("WEEK 1 - DATA ACQUISITION")
print("=" * 70)

print("\nFetching UCI Adult dataset...")

adult = fetch_ucirepo(id=2)

X = adult.data.features.copy()
y = adult.data.targets.copy()

print("\nDataset successfully acquired.")
print(f"Features shape : {X.shape}")
print(f"Target shape   : {y.shape}")

print("\nFeature columns:")
print(list(X.columns))

X.to_csv(
    RAW_DIR / "adult_features_raw.csv",
    index=False
)

y.to_csv(
    RAW_DIR / "adult_target_raw.csv",
    index=False
)

print("\nRaw files saved:")
print(RAW_DIR / "adult_features_raw.csv")
print(RAW_DIR / "adult_target_raw.csv")

print("\nData acquisition completed successfully.")
