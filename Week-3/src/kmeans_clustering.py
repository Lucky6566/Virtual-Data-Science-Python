from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ============================================================
# WEEK 3 - K-MEANS CLUSTERING ANALYSIS
# ============================================================

INPUT_FILE = Path("Week-3/data/adult_clustering_ready.csv")
SCREENSHOT_DIR = Path("Week-3/screenshots")
RESULT_FILE = Path("Week-3/data/kmeans_cluster_results.csv")

SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 75)
print("WEEK 3 - K-MEANS CLUSTERING ANALYSIS")
print("=" * 75)


# ------------------------------------------------------------
# 1. Load clustering-ready data
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("\nDATASET")
print("-" * 75)
print(f"Shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")


# ------------------------------------------------------------
# 2. Test different numbers of clusters
# ------------------------------------------------------------

X = df.copy()

k_values = range(2, 11)

inertia_values = []
silhouette_values = []


print("\nK-MEANS EVALUATION")
print("-" * 75)

for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    inertia = model.inertia_
    silhouette = silhouette_score(X, labels)

    inertia_values.append(inertia)
    silhouette_values.append(silhouette)

    print(
        f"K={k:2d} | "
        f"Inertia={inertia:,.2f} | "
        f"Silhouette={silhouette:.4f}"
    )


# ------------------------------------------------------------
# 3. Elbow Method
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    list(k_values),
    inertia_values,
    marker="o"
)

plt.title("Elbow Method for K-Means Clustering")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Within-Cluster Sum of Squares (Inertia)")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)

elbow_file = SCREENSHOT_DIR / "16_elbow_method.png"

plt.savefig(
    elbow_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 4. Silhouette Score
# ------------------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    list(k_values),
    silhouette_values,
    marker="o"
)

plt.title("Silhouette Score by Number of Clusters")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.xticks(list(k_values))
plt.grid(True, alpha=0.3)

silhouette_file = SCREENSHOT_DIR / "17_silhouette_scores.png"

plt.savefig(
    silhouette_file,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 5. Select K using highest silhouette score
# ------------------------------------------------------------

best_index = silhouette_values.index(
    max(silhouette_values)
)

best_k = list(k_values)[best_index]

print("\nCLUSTER SELECTION")
print("-" * 75)
print(f"Best K based on silhouette score: {best_k}")
print(
    f"Best silhouette score: "
    f"{silhouette_values[best_index]:.4f}"
)


# ------------------------------------------------------------
# 6. Train final K-Means model
# ------------------------------------------------------------

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = final_model.fit_predict(X)

df["cluster"] = cluster_labels


# ------------------------------------------------------------
# 7. Cluster sizes
# ------------------------------------------------------------

cluster_counts = (
    df["cluster"]
    .value_counts()
    .sort_index()
)

print("\nCLUSTER SIZES")
print("-" * 75)

print(cluster_counts)

print("\nCluster percentages:")

cluster_percentages = (
    cluster_counts / len(df) * 100
).round(2)

print(cluster_percentages)


# ------------------------------------------------------------
# 8. Save results
# ------------------------------------------------------------

df.to_csv(
    RESULT_FILE,
    index=False
)

print("\nRESULTS SAVED")
print("-" * 75)
print(f"Result file: {RESULT_FILE}")

print("\nCharts created:")
print(f"- {elbow_file}")
print(f"- {silhouette_file}")

print("\nWeek 3 K-Means analysis completed successfully.")