import pandas as pd

def generate_cluster_input(input_path: str, output_path: str):
    """
    Generate input data for clustering analysis based on ageing ratio.

    Parameters
    ----------
    input_path : str
        Path to the input CSV file (must contain 'Pop65plus' and 'PopTotal').
    output_path : str
        Path to save the processed clustering input CSV file.

    Raises
    ------
    ValueError
        If the input file does not contain required columns 'Pop65plus' or 'PopTotal'.
    """
    print("Generating clustering analysis input file...")
    df = pd.read_csv(input_path)

    # Ensure required columns exist
    if "Pop65plus" not in df.columns or "PopTotal" not in df.columns:
        raise ValueError("Missing 'Pop65plus' or 'PopTotal' column. Cannot calculate Percent65plus.")

    # Calculate ageing ratio (percentage of population aged 65+)
    df["Percent65plus"] = df["Pop65plus"] / df["PopTotal"] * 100

    # Keep only the required columns for clustering
    df_cluster = df[["Year", "Country", "Percent65plus"]]

    # Create pivot table:
    # - Rows: Country
    # - Columns: Year
    # - Values: Percent65plus
    pivot_df = df_cluster.pivot(index="Country", columns="Year", values="Percent65plus")

    # Handle missing values by forward-fill, then backward-fill across columns
    pivot_df = pivot_df.fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)

    # Save processed data to CSV
    pivot_df.to_csv(output_path)
    print(f"Clustering input data saved to: {output_path}")