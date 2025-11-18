import pandas as pd
import re

def extract_age_lower(age_str):
    """
    Extract the lower bound of an age range string.

    Examples:
        '65+'   -> 65
        '70–74' -> 70

    Parameters
    ----------
    age_str : str
        Age range string (e.g., '65+', '70–74').

    Returns
    -------
    int or None
        Lower bound of the age range as an integer, or None if parsing fails.
    """
    age_str = str(age_str).strip()  # Remove leading/trailing spaces
    match = re.match(r"(\d+)", age_str)
    return int(match.group(1)) if match else None


def generate_england_timeseries(input_path, output_path):
    """
    Generate a time series of the proportion of population aged 65+ for England,
    formatted for Prophet forecasting.

    This function:
    1. Reads the cleaned England population dataset.
    2. Filters for total population (all sexes).
    3. Calculates the proportion of people aged 65 and above for each year.
    4. Saves the result in Prophet's required format (columns: ds, y).

    Parameters
    ----------
    input_path : str
        Path to the cleaned England CSV file containing columns:
        Year, Age, Sex, Population.

    output_path : str
        Path to save the output CSV file in Prophet format (ds, y).

    Output
    ------
    CSV file with columns:
        ds : datetime (year)
        y  : ageing ratio (proportion of population aged 65+)
    """

    # Step 1: Read input data
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.strip()  # Remove any extra spaces

    print("Original columns:", df.columns.tolist())
    print("Available 'Sex' values:", df["Sex"].unique())

    # Step 2: Filter only total population (Sex == 'Total')
    df = df[df["Sex"] == "Total"]
    print("Number of rows after filtering by 'Total' sex:", len(df))

    # Step 3: Extract the lower bound of age ranges
    df["Age_lower"] = df["Age"].apply(extract_age_lower)
    print("Example of converted 'Age_lower' values:", df["Age_lower"].unique()[:10])

    # Drop rows where age parsing failed
    df = df[df["Age_lower"].notnull()]

    # Step 4: Calculate total population per year and 65+ population per year
    total_pop = df.groupby("Year")["Population"].sum()
    pop_65plus = df[df["Age_lower"] >= 65].groupby("Year")["Population"].sum()

    # Step 5: Calculate the ageing ratio (65+ population / total population)
    age_ratio = (pop_65plus / total_pop).reset_index()
    age_ratio.columns = ["ds", "y"]
    age_ratio["ds"] = pd.to_datetime(age_ratio["ds"], format="%Y")

    # Step 6: Save as Prophet input format
    age_ratio.to_csv(output_path, index=False)
    print(f"Successfully generated Prophet input file: {output_path}")
    print(age_ratio.head())