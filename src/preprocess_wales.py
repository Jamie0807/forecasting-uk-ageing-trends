import pandas as pd
import os

def clean_wales_projection_xls(input_path, output_path):
    """
    Clean Wales population projection data (2020-based Projection for Wales),
    read the PERSONS sheet, and convert it into a long-format CSV.

    Parameters:
    - input_path  : str, path to the original .xls file
    - output_path : str, path to save the output CSV file

    Returns:
    - df_clean : pandas.DataFrame, the cleaned long-format dataset
    """

    # Step 1: Read Excel data, skip the first 6 rows of metadata/description
    df = pd.read_excel(input_path, sheet_name="PERSONS", skiprows=6, engine="xlrd")

    # Step 2: Rename the first column to Age, the rest are year columns
    df.rename(columns={df.columns[0]: "Age"}, inplace=True)
    year_columns = df.columns[1:]

    # Step 3: Convert from wide to long format (Year, Population)
    df_melted = df.melt(id_vars=["Age"], value_vars=year_columns,
                        var_name="Year", value_name="Population")

    # Step 4: Add fixed columns (Sex + Country)
    df_melted["Sex"] = "Total"
    df_melted["Country"] = "Wales"

    # Step 5: Convert column types
    df_melted["Year"] = pd.to_numeric(df_melted["Year"], errors="coerce").astype(int)
    df_melted["Population"] = pd.to_numeric(df_melted["Population"], errors="coerce") * 1000

    # Step 6: Clean and reorder selected columns
    df_clean = df_melted[["Year", "Age", "Sex", "Country", "Population"]].dropna()

    # Step 7: Save as CSV file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    # Step 8: Print processing result log
    print(f"Wales projection data cleaned: {output_path}, total {len(df_clean)} records")

    return df_clean