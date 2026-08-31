from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# WEEK 3 - CLUSTER PROFILING
# ============================================================

INPUT_FILE = Path("Week-3/data/kmeans_cluster_results.csv")
ORIGINAL_FILE = Path("Week-3/data/adult_eda_ready.csv")
SCREENSHOT_DIR = Path("Week-3/screenshots")

PROFILE_FILE = Path("Week-3/data/cluster_profile.csv")

SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 75)
print("WEEK 3 - CLUSTER PROFILING & INTERPRETATION")
print("=" * 75)


# ------------------------------------------------------------
# 1. Load clustering results and original data
# ------------------------------------------------------------

clusters = pd.read_csv(INPUT_FILE)
original = pd.read_csv(ORIGINAL_FILE)

print("\nDATASET OVERVIEW")
print("-" * 75)
print(f"Clustering data shape: {clusters.shape}")
print(f"Original data shape  : {original.shape}")


# ------------------------------------------------------------
# 2. Attach cluster labels to original dataset
# ------------------------------------------------------------

# The clustering results contain the six standardized features.
# The original dataset contains the corresponding records.

cluster_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

analysis_df = original.copy()

analysis_df["cluster"] = clusters["cluster"].values


print("\nCLUSTER LABELS ATTACHED")
print("-" * 75)

print(
    analysis_df["cluster"]
    .value_counts()
    .sort_index()
)


# ------------------------------------------------------------
# 3. Numerical cluster profile
# ------------------------------------------------------------

print("\nNUMERICAL CLUSTER PROFILE")
print("-" * 75)

numerical_profile = (
    analysis_df
    .groupby("cluster")[cluster_features]
    .mean()
    .round(2)
)

print(numerical_profile)


# ------------------------------------------------------------
# 4. Median profile
# ------------------------------------------------------------

print("\nMEDIAN CLUSTER PROFILE")
print("-" * 75)

median_profile = (
    analysis_df
    .groupby("cluster")[cluster_features]
    .median()
    .round(2)
)

print(median_profile)


# ------------------------------------------------------------
# 5. Income distribution by cluster
# ------------------------------------------------------------

print("\nINCOME DISTRIBUTION BY CLUSTER (%)")
print("-" * 75)

income_profile = pd.crosstab(
    analysis_df["cluster"],
    analysis_df["income"],
    normalize="index"
) * 100

income_profile = income_profile.round(2)

print(income_profile)


# ------------------------------------------------------------
# 6. Education distribution by cluster
# ------------------------------------------------------------

print("\nTOP EDUCATION CATEGORY BY CLUSTER")
print("-" * 75)

for cluster_id in sorted(analysis_df["cluster"].unique()):

    subset = analysis_df[
        analysis_df["cluster"] == cluster_id
    ]

    education_counts = (
        subset["education"]
        .value_counts()
        .head(5)
    )

    print(f"\nCluster {cluster_id}:")
    print(education_counts)


# ------------------------------------------------------------
# 7. Workclass distribution by cluster
# ------------------------------------------------------------

print("\nTOP WORKCLASS BY CLUSTER")
print("-" * 75)

for cluster_id in sorted(analysis_df["cluster"].unique()):

    subset = analysis_df[
        analysis_df["cluster"] == cluster_id
    ]

    workclass_counts = (
        subset["workclass"]
        .value_counts()
        .head(5)
    )

    print(f"\nCluster {cluster_id}:")
    print(workclass_counts)


# ------------------------------------------------------------
# 8. Occupation distribution by cluster
# ------------------------------------------------------------

print("\nTOP OCCUPATIONS BY CLUSTER")
print("-" * 75)

for cluster_id in sorted(analysis_df["cluster"].unique()):

    subset = analysis_df[
        analysis_df["cluster"] == cluster_id
    ]

    occupation_counts = (
        subset["occupation"]
        .value_counts()
        .head(5)
    )

    print(f"\nCluster {cluster_id}:")
    print(occupation_counts)


# ------------------------------------------------------------
# 9. Sex distribution by cluster
# ------------------------------------------------------------

print("\nSEX DISTRIBUTION BY CLUSTER (%)")
print("-" * 75)

sex_profile = pd.crosstab(
    analysis_df["cluster"],
    analysis_df["sex"],
    normalize="index"
) * 100

print(sex_profile.round(2))


# ------------------------------------------------------------
# 10. Marital status by cluster
# ------------------------------------------------------------

print("\nTOP MARITAL STATUS BY CLUSTER")
print("-" * 75)

for cluster_id in sorted(analysis_df["cluster"].unique()):

    subset = analysis_df[
        analysis_df["cluster"] == cluster_id
    ]

    marital_counts = (
        subset["marital-status"]
        .value_counts()
        .head(5)
    )

    print(f"\nCluster {cluster_id}:")
    print(marital_counts)


# ------------------------------------------------------------
# 11. Save numerical profile
# ------------------------------------------------------------

numerical_profile.to_csv(PROFILE_FILE)

print("\nPROFILE SAVED")
print("-" * 75)
print(f"File: {PROFILE_FILE}")


# ------------------------------------------------------------
# 12. Cluster size chart
# ------------------------------------------------------------

cluster_sizes = (
    analysis_df["cluster"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(9, 6))

plt.bar(
    cluster_sizes.index.astype(str),
    cluster_sizes.values
)

plt.title("Number of Records in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Records")

for i, value in enumerate(cluster_sizes.values):
    plt.text(
        i,
        value,
        f"{value:,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

cluster_size_file = (
    SCREENSHOT_DIR / "18_cluster_sizes.png"
)

plt.savefig(
    cluster_size_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 13. Cluster age comparison
# ------------------------------------------------------------

age_means = (
    analysis_df
    .groupby("cluster")["age"]
    .mean()
)

plt.figure(figsize=(9, 6))

plt.bar(
    age_means.index.astype(str),
    age_means.values
)

plt.title("Average Age by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Average Age")

plt.tight_layout()

age_file = (
    SCREENSHOT_DIR / "19_cluster_average_age.png"
)

plt.savefig(
    age_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 14. Working hours comparison
# ------------------------------------------------------------

hours_means = (
    analysis_df
    .groupby("cluster")["hours-per-week"]
    .mean()
)

plt.figure(figsize=(9, 6))

plt.bar(
    hours_means.index.astype(str),
    hours_means.values
)

plt.title("Average Working Hours by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Hours per Week")

plt.tight_layout()

hours_file = (
    SCREENSHOT_DIR / "20_cluster_working_hours.png"
)

plt.savefig(
    hours_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 15. Income >50K percentage by cluster
# ------------------------------------------------------------

high_income = (
    analysis_df
    .assign(
        high_income=(
            analysis_df["income"] == ">50K"
        ).astype(int)
    )
    .groupby("cluster")["high_income"]
    .mean()
    * 100
)

high_income = high_income.round(2)

print("\nHIGH-INCOME PERCENTAGE BY CLUSTER")
print("-" * 75)

print(high_income)


plt.figure(figsize=(9, 6))

plt.bar(
    high_income.index.astype(str),
    high_income.values
)

plt.title("Percentage of >50K Income Records by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Percentage (%)")

plt.tight_layout()

income_file = (
    SCREENSHOT_DIR / "21_cluster_income_profile.png"
)

plt.savefig(
    income_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("WEEK 3 CLUSTER PROFILING SUMMARY")
print("=" * 75)

print("\nCluster sizes:")
print(cluster_sizes)

print("\nAverage age:")
print(age_means.round(2))

print("\nAverage hours per week:")
print(hours_means.round(2))

print("\n>50K income percentage:")
print(high_income)

print("\nCharts created:")
print("- 18_cluster_sizes.png")
print("- 19_cluster_average_age.png")
print("- 20_cluster_working_hours.png")
print("- 21_cluster_income_profile.png")

print("\nWeek 3 cluster profiling completed successfully.")