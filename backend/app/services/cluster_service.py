import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
AGEING_RATIO_PATH = os.path.join(BASE_DIR, "data/processed/ageing_ratio_per_region.csv")

CLUSTER_COLORS = ["#4f8ef7", "#f7c14f", "#61d08b", "#f76f6f", "#b06ef7"]


def get_cluster_results(n_clusters: int = 3):
    df = pd.read_csv(AGEING_RATIO_PATH)
    df = df.rename(columns={"Country": "region", "Percent65plus": "percent65plus"})

    # Pivot: rows = regions, columns = years
    df_pivot = df.pivot_table(index="region", columns="Year", values="percent65plus")
    df_pivot = df_pivot.dropna(axis=1)  # drop years with any NaN

    # Normalise each region's time-series
    scaler = StandardScaler()
    X = scaler.fit_transform(df_pivot)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    results = []
    for i, region in enumerate(df_pivot.index):
        cluster_id = int(labels[i])
        trend = (
            df[df["region"] == region][["Year", "percent65plus"]]
            .rename(columns={"Year": "year", "percent65plus": "value"})
            .assign(value=lambda d: d["value"].round(4))
            .to_dict(orient="records")
        )
        results.append(
            {
                "region": region,
                "cluster": cluster_id,
                "color": CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)],
                "trend": trend,
            }
        )

    return results
