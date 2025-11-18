import pandas as pd
import os

def clean_uk_projection_xls(input_path, output_path):
    """
    Clean UK national-level population projection data (2020-based Projection for the United Kingdom),
    read the PERSONS sheet, and output a long-format CSV file.

    Parameters:
    - input_path  : str, path to the original Excel file (.xls)
    - output_path : str, path to save the processed CSV file

    Returns:
    - df_clean : pandas.DataFrame, the cleaned long-format dataset
    """

    # Step 1: Read the data (skip the first 6 rows of metadata)
    df = pd.read_excel(input_path, sheet_name="PERSONS", skiprows=6, engine="xlrd")

    # Step 2: Rename the first column as Age, the rest as year columns
    df.rename(columns={df.columns[0]: "Age"}, inplace=True)
    year_columns = df.columns[1:]

    # Step 3: Convert from wide to long format (Age, Year → Population)
    df_melted = df.melt(
        id_vars=["Age"],
        value_vars=year_columns,
        var_name="Year",
        value_name="Population"
    )

    # Step 4: Add country and sex fields
    df_melted["Country"] = "United Kingdom"
    df_melted["Sex"] = "Total"

    # Step 5: Type conversion
    df_melted["Year"] = pd.to_numeric(df_melted["Year"], errors="coerce").astype(int)
    df_melted["Population"] = pd.to_numeric(df_melted["Population"], errors="coerce") * 1000

    # Step 6: Reorder columns and drop missing values
    df_clean = df_melted[["Year", "Age", "Sex", "Country", "Population"]].dropna()

    # Step 7: Save as CSV file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)

    # Step 8: Print log
    print(f"UK projection data cleaned: {output_path}, total {len(df_clean)} records")

    return df_clean