from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib


INPUT_FILE = Path("data/processed/adult_features_cleaned.csv")
OUTPUT_FILE = Path("data/processed/adult_features_preprocessed.csv")
PIPELINE_FILE = Path("data/processed/preprocessing_pipeline.joblib")


print("=" * 70)
print("WEEK 1 - DATA PREPROCESSING")
print("=" * 70)

# Load cleaned data
df = pd.read_csv(INPUT_FILE)

print(f"\nInput shape: {df.shape}")

# Identify feature types
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()

print("\nNumerical columns:")
print(numeric_cols)

print("\nCategorical columns:")
print(categorical_cols)

# Create preprocessing transformations
numeric_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_cols),
        ("categorical", categorical_pipeline, categorical_cols)
    ]
)

# Transform data
X_processed = preprocessor.fit_transform(df)

# Retrieve encoded categorical feature names
encoded_cat_cols = (
    preprocessor
    .named_transformers_["categorical"]
    .named_steps["onehot"]
    .get_feature_names_out(categorical_cols)
)

feature_names = numeric_cols + encoded_cat_cols.tolist()

processed_df = pd.DataFrame(
    X_processed,
    columns=feature_names
)

# Save transformed dataset
processed_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# Save preprocessing pipeline
joblib.dump(
    preprocessor,
    PIPELINE_FILE
)

print("\nPREPROCESSING RESULTS")
print("-" * 70)
print(f"Original feature count: {len(df.columns)}")
print(f"Numerical features: {len(numeric_cols)}")
print(f"Categorical features: {len(categorical_cols)}")
print(f"Processed feature count: {len(processed_df.columns)}")
print(f"Processed shape: {processed_df.shape}")

print(f"\nPreprocessed dataset saved to:")
print(OUTPUT_FILE)

print("\nPreprocessing pipeline saved to:")
print(PIPELINE_FILE)

print("\nPreprocessing completed successfully.")
