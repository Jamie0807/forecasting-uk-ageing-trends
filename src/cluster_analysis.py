import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def cluster_ageing_trends(input_path, output_path, n_clusters):
    """
    Perform KMeans clustering on regional ageing trends.

    Parameters
    ----------
    input_path : str
        Path to the CSV file containing ageing ratio data per region and year.
    output_path : str
        Path to save the output cluster visualization plot.
    n_clusters : int
        Number of clusters to form.

    Notes
    -----
    - The function automatically detects the column containing ageing ratio 
      (either 'Percent65plus' or 'Ageing Ratio').
    - Data is pivoted so that rows = years, columns = regions.
    - StandardScaler is used to normalize each region's trend before clustering.
    - The final plot shows the ageing ratio trends for each region, labeled by cluster.
    """
    print("Reading cluster input data...")
    df = pd.read_csv(input_path)

    # Print column names to confirm structure
    print("Cluster input file columns:", df.columns.tolist())

    # Automatically detect which column to use for ageing ratio
    if "Percent65plus" in df.columns:
        value_column = "Percent65plus"
    elif "Ageing Ratio" in df.columns:
        value_column = "Ageing Ratio"
    else:
        raise ValueError("No 'Percent65plus' or 'Ageing Ratio' column found. Please check input file columns.")

    # Create pivot table: rows = Year, columns = Country, values = ageing ratio
    df_pivot = df.pivot(index="Year", columns="Country", values=value_column)
    df_pivot = df_pivot.dropna(axis=1)  # Remove any columns (regions) with missing values

    print("Successfully built clustering input matrix, shape:", df_pivot.shape)

    # Standardize features (regions are treated as samples, so transpose the matrix)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pivot.T)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Plot results
    plt.figure(figsize=(10, 6))
    for i, region in enumerate(df_pivot.columns):
        plt.plot(df_pivot.index, df_pivot[region], label=f"{region} (Cluster {labels[i]})")
    
    plt.title("Regional Ageing Trends Clustered")
    plt.xlabel("Year")
    plt.ylabel("Ageing Ratio (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Cluster plot saved to: {output_path}")