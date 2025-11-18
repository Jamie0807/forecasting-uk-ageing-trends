import matplotlib.pyplot as plt
import os

def plot_forecast_comparison(results, output_path):
    print("Comparing forecast results (forecast_comparison)")
    print(f"Number of regions included: {len(results)}")
    for region in results:
        print(f"  - Included: {region}, forecast length = {len(results[region])}")

    """
    Compare Prophet forecast results across multiple regions and generate a unified plot.

    Parameters:
    - results: dict, keys are region names, values are Prophet forecast DataFrames (must contain ds and yhat)
    - output_path: str, path to save the image (.png)
    """

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Plot forecasts
    plt.figure(figsize=(12, 6))
    for region, forecast_df in results.items():
        plt.plot(forecast_df["ds"], forecast_df["yhat"], label=region)

    plt.title("Comparison of 65+ Forecasts Across Regions")
    plt.xlabel("Year")
    plt.ylabel("Proportion of Population Aged 65+ (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Regional comparison figure saved: {output_path}")