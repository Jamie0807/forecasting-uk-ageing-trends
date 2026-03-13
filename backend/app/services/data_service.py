import os
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
AGEING_RATIO_PATH = os.path.join(BASE_DIR, "data/processed/ageing_ratio_per_region.csv")


def _load_ageing_df() -> pd.DataFrame:
    df = pd.read_csv(AGEING_RATIO_PATH)
    df = df.rename(
        columns={"Country": "region", "Percent65plus": "percent65plus"}
    )
    df["percent65plus"] = df["percent65plus"].round(4)
    return df


def get_ageing_ratio():
    """Return all historical ageing ratio records as a list of dicts."""
    df = _load_ageing_df()
    return df[["Year", "region", "percent65plus"]].to_dict(orient="records")


def get_regions():
    """Return distinct region names."""
    df = _load_ageing_df()
    return sorted(df["region"].unique().tolist())


def get_overview_stats():
    """Return per-region summary: earliest / latest year and percent65plus."""
    df = _load_ageing_df()
    stats = []
    for region in sorted(df["region"].unique()):
        region_df = df[df["region"] == region].sort_values("Year")
        earliest = region_df.iloc[0]
        latest = region_df.iloc[-1]
        stats.append(
            {
                "region": region,
                "earliest_year": int(earliest["Year"]),
                "earliest_percent": round(float(earliest["percent65plus"]), 2),
                "latest_year": int(latest["Year"]),
                "latest_percent": round(float(latest["percent65plus"]), 2),
                "total_change": round(
                    float(latest["percent65plus"]) - float(earliest["percent65plus"]), 2
                ),
            }
        )
    return stats
