from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ============================================================
# WEEK 3 - HIERARCHICAL CLUSTERING ANALYSIS
# ============================================================

DATA_FILE = Path("Week-3/data/adult_clustering_ready.csv")
SCREENSHOT_DIR = Path("Week-3/screenshots")

SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("WEEK 3 - HIERARCHICAL CLUSTERING ANALYSIS")
print("=" * 80)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(DATA_FILE)

print("\nDATASET")
print("-" * 80)
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")


# ------------------------------------------------------------
# 2. REMOVE DUPLICATES
# ------------------------------------------------------------

X = df.drop_duplicates().copy()

print("\nDUPLICATE HANDLING")
print("-" * 80)
print(f"Original rows: {len(df)}")
print(f"Rows after duplicate removal: {len(X)}")
print(f"Duplicates removed: {len(df) - len(X)}")


# ------------------------------------------------------------
# 3. SAMPLE DATA
# ------------------------------------------------------------

SAMPLE_SIZE = 5000

if len(X) > SAMPLE_SIZE:

    X_sample = X.sample(
        n=SAMPLE_SIZE,
        random_state=42
    ).reset_index(drop=True)

else:

    X_sample = X.copy().reset_index(drop=True)


print("\nSAMPLING")
print("-" * 80)
print(f"Hierarchical clustering sample size: {len(X_sample)}")
print("Random state: 42")


# ------------------------------------------------------------
# 4. TEST DIFFERENT K VALUES
# ------------------------------------------------------------

print("\nHIERARCHICAL CLUSTERING EVALUATION")
print("-" * 80)

results = []

for k in range(2, 7):

    print(f"Running K={k}...")

    model = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward"
    )

    labels = model.fit_predict(X_sample)

    score = silhouette_score(
        X_sample,
        labels
    )

    results.append({
        "k": k,
        "silhouette": score
    })

    print(
        f"K={k} | "
        f"Silhouette={score:.4f}"
    )


results_df = pd.DataFrame(results)


# ------------------------------------------------------------
# 5. BEST K
# ------------------------------------------------------------

best_row = results_df.loc[
    results_df["silhouette"].idxmax()
]

best_k = int(best_row["k"])
best_score = float(best_row["silhouette"])


print("\nCLUSTER SELECTION")
print("-" * 80)
print(f"Best K: {best_k}")
print(f"Best silhouette score: {best_score:.4f}")


# ------------------------------------------------------------
# 6. FINAL MODEL
# ------------------------------------------------------------

print("\nFINAL HIERARCHICAL MODEL")
print("-" * 80)

final_model = AgglomerativeClustering(
    n_clusters=best_k,
    linkage="ward"
)

final_labels = final_model.fit_predict(
    X_sample
)

X_sample["cluster"] = final_labels


# ------------------------------------------------------------
# 7. CLUSTER SIZES
# ------------------------------------------------------------

cluster_sizes = (
    X_sample["cluster"]
    .value_counts()
    .sort_index()
)

print("\nCLUSTER SIZES")
print("-" * 80)
print(cluster_sizes)

print("\nCluster percentages:")

cluster_percentages = (
    cluster_sizes / len(X_sample) * 100
).round(2)

print(cluster_percentages)


# ------------------------------------------------------------
# 8. PCA VISUALIZATION
# ------------------------------------------------------------

print("\nCREATING PCA VISUALIZATION")
print("-" * 80)

features = X_sample.drop(
    columns=["cluster"]
)

pca = PCA(
    n_components=2,
    random_state=42
)

components = pca.fit_transform(
    features
)

plot_df = pd.DataFrame(
    components,
    columns=["PC1", "PC2"]
)

plot_df["cluster"] = (
    X_sample["cluster"].values
)


plt.figure(figsize=(10, 7))

for cluster in sorted(
    plot_df["cluster"].unique()
):

    subset = plot_df[
        plot_df["cluster"] == cluster
    ]

    plt.scatter(
        subset["PC1"],
        subset["PC2"],
        s=10,
        alpha=0.5,
        label=f"Cluster {cluster}"
    )


plt.title(
    f"Hierarchical Clustering PCA "
    f"(K={best_k})"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.legend()
plt.tight_layout()

pca_file = (
    SCREENSHOT_DIR /
    "22_hierarchical_clustering_pca.png"
)

plt.savefig(
    pca_file,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 9. SILHOUETTE SCORE CHART
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    results_df["k"],
    results_df["silhouette"],
    marker="o"
)

plt.title(
    "Hierarchical Clustering Silhouette Scores"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")

plt.xticks(
    results_df["k"]
)

plt.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

silhouette_file = (
    SCREENSHOT_DIR /
    "23_hierarchical_silhouette_scores.png"
)

plt.savefig(
    silhouette_file,
    dpi=150
)

plt.close()


# ------------------------------------------------------------
# 10. SAVE RESULTS
# ------------------------------------------------------------

results_file = (
    Path("Week-3/data") /
    "hierarchical_clustering_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)


cluster_file = (
    Path("Week-3/data") /
    "hierarchical_cluster_sample.csv"
)

X_sample.to_csv(
    cluster_file,
    index=False
)


# ------------------------------------------------------------
# 11. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 80)
print("HIERARCHICAL CLUSTERING SUMMARY")
print("=" * 80)

print(f"\nSample size: {len(X_sample)}")
print(f"Best K: {best_k}")
print(f"Best silhouette score: {best_score:.4f}")

print("\nCluster sizes:")
print(cluster_sizes)

print("\nCluster percentages:")
print(cluster_percentages)

print("\nResults saved:")
print(f"- {results_file}")
print(f"- {cluster_file}")

print("\nCharts created:")
print("- 22_hierarchical_clustering_pca.png")
print("- 23_hierarchical_silhouette_scores.png")

print("\nHierarchical clustering analysis completed successfully.")