import pandas as pd
import os

def clean_projected_population_data(input_xlsx_path, output_csv_path, sheet_name="Persons"):
    """
    Clean the ONS 2018-based Subnational Population Projections data,
    and convert age-group population projections into a standard long-format CSV table.

    Parameters:
    - input_xlsx_path : str, path to the input Excel file (e.g. SNPP18dt2.xlsx)
    - output_csv_path : str, path to save the output CSV file
    - sheet_name      : str, name of the sheet to read, default is "Persons"

    Output fields:
    - Region_Code            : region code
    - Region                 : region name
    - Age_Group              : age group (e.g. "0-4", "5-9", ...)
    - Year                   : projection year (2018–2043)
    - Projected_Population   : projected population count

    Returns:
    - df_long : pandas.DataFrame, cleaned long-format table
    """

    # Step 1: Define column names (first 3 columns + year columns)
    columns = ["Region_Code", "Region", "Age_Group"] + list(range(2018, 2044))

    # Step 2: Skip the first 6 rows, read data starting from row 7
    df = pd.read_excel(input_xlsx_path, sheet_name=sheet_name, skiprows=6, names=columns)

    # Step 3: Drop rows with missing values in basic fields (region, age group)
    df.dropna(subset=["Region_Code", "Region", "Age_Group"], inplace=True)

    # Step 4: Convert wide format to long format (one row per year)
    df_long = df.melt(
        id_vars=["Region_Code", "Region", "Age_Group"],
        var_name="Year",
        value_name="Projected_Population"
    )

    # Step 5: Convert data types
    df_long["Year"] = df_long["Year"].astype(int)
    df_long["Projected_Population"] = pd.to_numeric(df_long["Projected_Population"], errors="coerce")

    # Step 6: Save as CSV file
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_long.to_csv(output_csv_path, index=False)

    # Step 7: Print completion log
    print(f"Projected population data cleaned: {output_csv_path}, total {len(df_long)} records")

    return df_long