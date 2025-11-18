import pandas as pd
import os
import re


# Extract population data by age and year from ONS Excel raw files (expanded by age and year),
# and clean it into a long-format CSV.

def clean_population_data(input_excel_path, output_csv_path, sheet_name="Table 6"):
    """
    Extracts UK population data by age and year from the original ONS Excel file,
    cleans and reshapes it into long format (Age, Year, Population), and saves as a CSV.

    Parameters:
    - input_excel_path : str, path to the input Excel file
    - output_csv_path  : str, path to save the output CSV file
    - sheet_name       : str, sheet name in the Excel file (default "Table 6")

    Returns:
    - df_melted : pandas.DataFrame, cleaned long-format dataset
    """

    # Step 1: Read the header row (second row) to extract column names (age, sex, year info)
    df_header = pd.read_excel(input_excel_path, sheet_name=sheet_name, nrows=1, skiprows=1)
    raw_cols = df_header.columns.tolist()

    raw_cols[0] = "Age"   # The first column should be age
    raw_cols[1] = "Sex"   # The second column should be sex

    # Step 2: Read the actual data (starting from the third row, no header)
    df = pd.read_excel(input_excel_path, sheet_name=sheet_name, skiprows=2, header=None)
    df.columns = raw_cols

    # Step 3: Keep only total population (Sex == "Persons"), drop male/female breakdown
    df = df[df["Sex"] == "Persons"].copy()
    df.drop(columns=["Sex"], inplace=True)

    # Step 4: Extract year values from column names, e.g. "Mid-2022" → "2022"
    def extract_year(col):
        try:
            year = int(str(col)[-4:])
            if 1900 <= year <= 2100:
                return str(year)
        except:
            return None
        return None

    renamed_cols = {}
    for col in df.columns:
        if col == "Age":
            renamed_cols[col] = "Age"
        else:
            year = extract_year(col)
            if year:
                renamed_cols[col] = year

    df.rename(columns=renamed_cols, inplace=True)

    # Step 5: Keep only valid year columns (4-digit format)
    year_columns = [col for col in df.columns if col != "Age" and re.fullmatch(r"\d{4}", col)]
    df = df[["Age"] + year_columns]

    # Step 6: Convert wide format to long format (melt)
    df_melted = df.melt(id_vars=["Age"], var_name="Year", value_name="Population")

    # Step 7: Clean Age column (remove "All Ages", unify as integers)
    df_melted = df_melted[df_melted["Age"] != "All Ages"]
    df_melted["Age"] = df_melted["Age"].astype(str).str.replace("+", "", regex=False)
    df_melted["Age"] = pd.to_numeric(df_melted["Age"], errors="coerce")
    df_melted.dropna(subset=["Age"], inplace=True)

    # Step 8: Convert column types (Year → int, Population → float)
    df_melted["Year"] = df_melted["Year"].astype(int)
    df_melted["Population"] = pd.to_numeric(df_melted["Population"], errors="coerce")

    # Step 9: Save as CSV (create directories automatically if needed)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_melted.to_csv(output_csv_path, index=False)

    # Step 10: Output a completion message
    print(f"Processed: {output_csv_path}, total {len(df_melted)} records")

    return df_melted