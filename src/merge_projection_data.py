import pandas as pd
import os

def merge_population_data(england_path, wales_path, scotland_path, output_path):
    """
    Merge population projection data for England, Wales, and Scotland 
    into a unified long-format CSV file.

    Parameters
    ----------
    england_path : str
        File path to England's projection data CSV.
    wales_path : str
        File path to Wales's projection data CSV.
    scotland_path : str
        File path to Scotland's projection data CSV.
    output_path : str
        File path for saving the merged CSV.

    Expected Input Format
    ---------------------
    The input CSVs must contain the following columns:
        ['Year', 'Age', 'Sex', 'Country', 'Population']

    Returns
    -------
    df_all : pandas.DataFrame
        Combined DataFrame containing all regions' data.
    """

    # Step 1: Read projection data for the three regions
    df_england = pd.read_csv(england_path)
    df_wales = pd.read_csv(wales_path)
    df_scotland = pd.read_csv(scotland_path)

    # Step 2: Concatenate all regional datasets into one
    df_all = pd.concat([df_england, df_wales, df_scotland], ignore_index=True)

    # Step 3: Standardize data types
    df_all["Year"] = df_all["Year"].astype(int)
    df_all["Population"] = pd.to_numeric(df_all["Population"], errors="coerce")

    # Step 4: Save the merged file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_csv(output_path, index=False)

    # Step 5: Print summary log
    print(f"Merging complete: {output_path}, total {len(df_all)} records")

    return df_all