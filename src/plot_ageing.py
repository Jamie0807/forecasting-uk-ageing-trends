import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_65plus_trend(input_path, output_path, save_cluster_data_path=None):
    print("Plotting actual data chart: plot_65plus_trend")
    """
    Plot the trend of the proportion of the population aged 65 and above across UK regions (2020–2070),
    and save the figure as a PNG file.

    Parameters:
    - input_path : str, path to the CSV file containing population by age group 
                   (must include fields: Age, Year, Country, Population)
    - output_path : str, path to save the image (PNG)
    - save_cluster_data_path : str, optional path to save processed data for clustering

    Figure description:
    - X-axis: Year
    - Y-axis: Percentage of population aged 65+
    - Multiple lines: regions such as England, Wales, Scotland
    """

    # Step 1: Load data and keep "Total" sex records
    df = pd.read_csv(input_path)
    df = df[df["Sex"] == "Total"]

    # Step 2: Parse the Age field (extract starting age)
    def parse_age(age_str):
        try:
            if isinstance(age_str, str):
                clean = age_str.replace("–", "-").replace("—", "-").replace("−", "-")
                start = clean.split("-")[0].replace("+", "").strip()
                return int(start)
        except:
            return None

    df["AgeStart"] = df["Age"].apply(parse_age)

    # Step 3: Calculate population aged 65+ (by year and country)
    df_65plus = df[df["AgeStart"] >= 65]
    group_65plus = df_65plus.groupby(["Year", "Country"])["Population"].sum().reset_index(name="Pop65plus")

    # Step 4: Calculate total population per year
    group_total = df.groupby(["Year", "Country"])["Population"].sum().reset_index(name="PopTotal")

    # Step 5: Merge and compute ageing ratio
    merged = pd.merge(group_65plus, group_total, on=["Year", "Country"])
    merged["Percent65plus"] = merged["Pop65plus"] / merged["PopTotal"] * 100

    # Step 6: Print data check info
    print(f"Number of rows: {merged.shape[0]}")
    print(f"Regions: {merged['Country'].unique().tolist()}")
    print(f"Year column type: {merged['Year'].dtype}")
    print(merged.head())

    # Step 7: Plot figure
    plt.figure(figsize=(10, 6))
    for country in merged["Country"].dropna().unique():
        subset = merged[merged["Country"] == country]
        plt.plot(subset["Year"], subset["Percent65plus"], label=country)

    plt.title("Proportion of Population Aged 65+ by UK Region (2020–2070)")
    plt.xlabel("Year")
    plt.ylabel("Percentage (%)")
    plt.legend(title="Country")
    plt.grid(True)
    plt.tight_layout()

    # Step 8: Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Image saved to: {output_path}")

    # Step 9: Optionally export data for clustering analysis
    if save_cluster_data_path:
        merged.to_csv(save_cluster_data_path, index=False)
        print(f"Cluster input data saved to: {save_cluster_data_path}")
        merged.to_csv("data/processed/ageing_ratio_per_region.csv", index=False)
        print("Cluster input data saved to: data/processed/ageing_ratio_per_region.csv")