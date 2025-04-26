# snt/core.py
import pandas as pd
from pathlib import Path
import numpy as np

def concatenate(folder_path):
    files = Path(folder_path).glob("*.xlsx")
    df_list = [pd.read_excel(file) for file in files]
    return pd.concat(df_list, ignore_index=True)

def rename(df, dict_path):
    name_map = pd.read_excel(dict_path)
    for i in range(len(name_map)):
        old = name_map.iloc[i, 0]
        new = name_map.iloc[i, 1]
        df.rename(columns={old: new}, inplace=True)
    return df

def compute(df, compute_path):
    comp = pd.read_excel(compute_path)
    for i in range(len(comp)):
        new_var = comp['new_variable'][i]
        op = comp['operation'][i]
        components = [x.strip() for x in comp['components'][i].split(',')]

        if op == "rowsum":
            df[new_var] = df[components].sum(axis=1, skipna=True, min_count=1)
        elif op == "subtract":
            df[new_var] = df[components[0]] - df[components[1]]
            df[new_var] = df[new_var].clip(lower=0)
    return df

def sort(df, compute_path):
    # Read the compute instructions
    comp = pd.read_excel(compute_path)

    sorted_columns = []
    
    # For each row in compute_path
    for i in range(len(comp)):
        components = [x.strip() for x in comp['components'][i].split(',')]
        new_var = comp['new_variable'][i]
        
        sorted_columns.extend(components)
        sorted_columns.append(new_var)

    # Add any remaining columns that were not mentioned
    remaining_columns = [col for col in df.columns if col not in sorted_columns]
    final_order = remaining_columns + sorted_columns

    # Reorder the DataFrame
    df_sorted = df[final_order]

    return df_sorted

def split(df, split_path):
    # Read the mapping Excel file
    mapping = pd.read_excel(split_path)

    # Get names
    original_col = mapping['original_col'].iloc[0]
    new_col_month = mapping['new_col_month'].iloc[0]
    new_col_year = mapping['new_col_year'].iloc[0]

    # Split the original column
    split_data = df[original_col].str.split(' ', expand=True)

    # Assign to new columns
    df[new_col_month] = split_data[0]
    df[new_col_year] = split_data[1]

    # Define month mappings (English and French)
    month_map = {
        'January': '01', 'Janvier': '01',
        'February': '02', 'Février': '02', 'Fevrier': '02',
        'March': '03', 'Mars': '03',
        'April': '04', 'Avril': '04',
        'May': '05', 'Mai': '05',
        'June': '06', 'Juin': '06',
        'July': '07', 'Juillet': '07',
        'August': '08', 'Août': '08', 'Aout': '08',
        'September': '09', 'Septembre': '09',
        'October': '10', 'Octobre': '10',
        'November': '11', 'Novembre': '11',
        'December': '12', 'Décembre': '12', 'Decembre': '12'
    }

    # Replace month names
    df[new_col_month] = df[new_col_month].map(lambda x: month_map.get(x, x))

    return df

# Outlier
import pandas as pd
import numpy as np

# Function to detect outliers using Scatterplot with Q1 and Q3 lines
def detect_outliers_scatterplot(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

# Function to apply winsorization to a column
def winsorize_series(series, lower_bound, upper_bound):
    return series.clip(lower=lower_bound, upper=upper_bound)

# Function to process and export the results for all numeric columns using Winsorization
def outliers(df, grouped_columns_path):
    # Load the grouped columns from Excel
    grouped_columns_df = pd.read_excel(grouped_columns_path)

    # Strip any leading/trailing spaces in column names
    grouped_columns_df.columns = grouped_columns_df.columns.str.strip()

    # Extract the column names that should be used for grouping from the Excel file
    group_columns = list(grouped_columns_df.columns)

    # Ensure 'year' is included in the group columns
    if 'year' not in group_columns:
        raise ValueError("'year' must be part of the group columns. Columns available: " + ", ".join(group_columns))

    # Group by the columns extracted from the Excel file (including 'year')
    grouped = df.groupby(group_columns)

    results = []
    
    # Auto-detect numeric columns
    numeric_columns = df.select_dtypes(include=np.number).columns

    for numeric_column in numeric_columns:
        for group_keys, group in grouped:
            # Perform outlier detection and Winsorization for the numeric column
            lower_bound, upper_bound = detect_outliers_scatterplot(group, numeric_column)
            group[f'{numeric_column}_lower_bound'] = lower_bound
            group[f'{numeric_column}_upper_bound'] = upper_bound
            group[f'{numeric_column}_category'] = np.where(
                (group[numeric_column] < lower_bound) | (group[numeric_column] > upper_bound), 'Outlier', 'Non-Outlier'
            )
            group[f'{numeric_column}_winsorized'] = winsorize_series(group[numeric_column], lower_bound, upper_bound)
            results.append(group)

    final_df = pd.concat(results)

    # Prepare the columns to export
    export_columns = [
        'adm1', 'adm2', 'adm3', 'hf', 'hf_uid', 'year', 'month', 'date'
    ]
    for numeric_column in numeric_columns:
        export_columns.extend([
            numeric_column,
            f'{numeric_column}_category',
            f'{numeric_column}_lower_bound',
            f'{numeric_column}_upper_bound',
            f'{numeric_column}_winsorized'
        ])

    export_columns = [col for col in export_columns if col in final_df.columns]

    return final_df[export_columns]

# result = process_all_numeric_columns(df, grouped_columns_path)


