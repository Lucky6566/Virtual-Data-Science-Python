from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# WEEK 2 - EXPLORATORY DATA ANALYSIS & VISUALIZATION
# UCI ADULT DATASET
# =========================================================

DATA_FILE = Path("Week-2/data/adult_eda_ready.csv")
SCREENSHOT_DIR = Path("Week-2/screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("WEEK 2 - EXPLORATORY DATA ANALYSIS & VISUALIZATION")
print("=" * 80)


# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("\nDATASET OVERVIEW")
print("-" * 80)
print(f"Shape: {df.shape}")

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ---------------------------------------------------------
# 2. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------

print("\nDESCRIPTIVE STATISTICS")
print("-" * 80)

numeric_cols = df.select_dtypes(include=np.number).columns

print(df[numeric_cols].describe().round(2))


# ---------------------------------------------------------
# 3. TARGET DISTRIBUTION
# ---------------------------------------------------------

income_counts = df["income"].value_counts()

print("\nINCOME DISTRIBUTION")
print("-" * 80)
print(income_counts)

income_pct = (
    df["income"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nIncome percentage:")
print(income_pct)


plt.figure(figsize=(8, 5))

sns.countplot(
    data=df,
    x="income"
)

plt.title("Income Distribution")
plt.xlabel("Income Category")
plt.ylabel("Number of Individuals")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "01_income_distribution.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 4. AGE DISTRIBUTION
# ---------------------------------------------------------

print("\nAGE ANALYSIS")
print("-" * 80)
print(df["age"].describe().round(2))

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="age",
    bins=30,
    kde=True
)

plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "02_age_distribution.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 5. EDUCATION DISTRIBUTION
# ---------------------------------------------------------

education_counts = df["education"].value_counts()

print("\nEDUCATION DISTRIBUTION")
print("-" * 80)
print(education_counts)

plt.figure(figsize=(11, 6))

education_counts.sort_values().plot(
    kind="barh"
)

plt.title("Education Level Distribution")
plt.xlabel("Number of Individuals")
plt.ylabel("Education Level")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "03_education_distribution.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 6. WORKCLASS DISTRIBUTION
# ---------------------------------------------------------

workclass_counts = df["workclass"].value_counts()

print("\nWORKCLASS DISTRIBUTION")
print("-" * 80)
print(workclass_counts)

plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    y="workclass",
    order=workclass_counts.index
)

plt.title("Workclass Distribution")
plt.xlabel("Number of Individuals")
plt.ylabel("Workclass")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "04_workclass_distribution.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 7. HOURS PER WEEK
# ---------------------------------------------------------

print("\nHOURS PER WEEK")
print("-" * 80)
print(df["hours-per-week"].describe().round(2))

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="hours-per-week",
    bins=30,
    kde=True
)

plt.title("Hours Worked Per Week")
plt.xlabel("Hours per Week")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "05_hours_per_week.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 8. AGE VS INCOME
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="income",
    y="age"
)

plt.title("Age Distribution by Income Category")
plt.xlabel("Income")
plt.ylabel("Age")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "06_age_vs_income.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 9. EDUCATION VS INCOME
# ---------------------------------------------------------

education_income = pd.crosstab(
    df["education"],
    df["income"],
    normalize="index"
) * 100

print("\nEDUCATION VS INCOME (%)")
print("-" * 80)
print(education_income.round(2))

education_income = education_income.sort_values(
    ">50K"
)

education_income.plot(
    kind="barh",
    figsize=(11, 8)
)

plt.title("Income Distribution by Education Level")
plt.xlabel("Percentage")
plt.ylabel("Education Level")
plt.legend(title="Income")

plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "07_education_vs_income.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 10. HOURS VS INCOME
# ---------------------------------------------------------

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="income",
    y="hours-per-week"
)

plt.title("Hours Worked Per Week by Income")
plt.xlabel("Income")
plt.ylabel("Hours per Week")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "08_hours_vs_income.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 11. SEX VS INCOME
# ---------------------------------------------------------

sex_income = pd.crosstab(
    df["sex"],
    df["income"],
    normalize="index"
) * 100

print("\nSEX VS INCOME (%)")
print("-" * 80)
print(sex_income.round(2))

sex_income.plot(
    kind="bar",
    figsize=(9, 6)
)

plt.title("Income Distribution by Sex")
plt.xlabel("Sex")
plt.ylabel("Percentage")
plt.xticks(rotation=0)
plt.legend(title="Income")

plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "09_sex_vs_income.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 12. MARITAL STATUS VS INCOME
# ---------------------------------------------------------

marital_income = pd.crosstab(
    df["marital-status"],
    df["income"],
    normalize="index"
) * 100

print("\nMARITAL STATUS VS INCOME (%)")
print("-" * 80)
print(marital_income.round(2))

marital_income = marital_income.sort_values(
    ">50K"
)

marital_income.plot(
    kind="barh",
    figsize=(11, 7)
)

plt.title("Income Distribution by Marital Status")
plt.xlabel("Percentage")
plt.ylabel("Marital Status")
plt.legend(title="Income")

plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "10_marital_status_vs_income.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 13. CORRELATION MATRIX
# ---------------------------------------------------------

correlation = df[numeric_cols].corr()

print("\nCORRELATION MATRIX")
print("-" * 80)
print(correlation.round(3))

plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "11_correlation_heatmap.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 14. CAPITAL GAIN DISTRIBUTION
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="capital-gain",
    bins=40
)

plt.title("Capital Gain Distribution")
plt.xlabel("Capital Gain")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "12_capital_gain_distribution.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 15. OUTLIER ANALYSIS
# ---------------------------------------------------------

outlier_columns = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week"
]

print("\nOUTLIER ANALYSIS")
print("-" * 80)

for col in outlier_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    print(
        f"{col}: {len(outliers)} outliers "
        f"(lower={lower:.2f}, upper={upper:.2f})"
    )


# ---------------------------------------------------------
# 16. AGE OUTLIERS
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="age"
)

plt.title("Age Outlier Analysis")
plt.xlabel("Age")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "13_age_outlier_analysis.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 17. HOURS OUTLIERS
# ---------------------------------------------------------

plt.figure(figsize=(9, 5))

sns.boxplot(
    data=df,
    x="hours-per-week"
)

plt.title("Hours Worked Outlier Analysis")
plt.xlabel("Hours per Week")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "14_hours_outlier_analysis.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 18. TOP OCCUPATIONS
# ---------------------------------------------------------

occupation_counts = df["occupation"].value_counts().head(10)

print("\nTOP 10 OCCUPATIONS")
print("-" * 80)
print(occupation_counts)

plt.figure(figsize=(10, 6))

occupation_counts.sort_values().plot(
    kind="barh"
)

plt.title("Top 10 Occupations")
plt.xlabel("Number of Individuals")
plt.ylabel("Occupation")
plt.tight_layout()

plt.savefig(
    SCREENSHOT_DIR / "15_top_occupations.png",
    dpi=200
)

plt.close()


# ---------------------------------------------------------
# 19. SUMMARY INSIGHTS
# ---------------------------------------------------------

print("\n" + "=" * 80)
print("WEEK 2 EDA SUMMARY")
print("=" * 80)

print(f"""
Dataset:
- Records: {len(df):,}
- Features + target: {df.shape[1]}
- Missing values: {df.isnull().sum().sum()}
- Duplicate rows: {df.duplicated().sum()}

Income:
- <=50K: {income_counts.get('<=50K', 0):,}
- >50K : {income_counts.get('>50K', 0):,}

Average age:
- {df['age'].mean():.2f} years

Average working hours:
- {df['hours-per-week'].mean():.2f} hours/week

Most common education:
- {df['education'].mode()[0]}

Most common workclass:
- {df['workclass'].mode()[0]}

Most common occupation:
- {df['occupation'].mode()[0]}

Most common native country:
- {df['native-country'].mode()[0]}
""")

print("\nCharts created successfully.")

for file in sorted(SCREENSHOT_DIR.glob("*.png")):
    print(" -", file.name)

print("\nWeek 2 EDA completed successfully.")