import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

# Set default font (English)
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['axes.unicode_minus'] = True  # Support display of minus signs

def plot_england_forecast(input_path, output_path):
    """
    Visualize Prophet forecast results: ageing ratio (65+) for England.

    Parameters:
    - input_path: str, path to the forecast result CSV file (should contain ds, yhat, yhat_lower, yhat_upper)
    - output_path: str, path to save the generated image (e.g. output/england_forecast.png)

    Output:
    - A PNG image containing the forecast curve and confidence interval
    """
    # Read forecast results
    df = pd.read_csv(input_path)

    # Create figure
    plt.figure(figsize=(10, 5))
    plt.plot(df["ds"], df["yhat"], label="Predicted ratio (65+)", linewidth=2)
    plt.fill_between(df["ds"], df["yhat_lower"], df["yhat_upper"], alpha=0.2, label="Confidence interval")

    # Configure plot elements
    plt.xlabel("Year")
    plt.ylabel("Proportion of population aged 65+")
    plt.title("Forecast of Ageing Population Ratio")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Image saved: {output_path}")