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

### Outlier
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

# Function to process and analyze the dataset for numeric columns using Winsorization
def detect_outliers(df):
    # Filter for numeric columns only
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    processed_dfs = []

    for column in numeric_columns:
        # Skip if the column contains only missing values
        if df[column].isnull().all():
            print(f"Skipping column {column} as it contains only missing values.")
            continue

        print(f"Processing numeric column: {column}")
        # Apply the outlier detection and winsorization
        lower_bound, upper_bound = detect_outliers_scatterplot(df, column)
        df[f'{column}_lower_bound'] = lower_bound
        df[f'{column}_upper_bound'] = upper_bound
        df[f'{column}_category'] = np.where(
            (df[column] < lower_bound) | (df[column] > upper_bound), 'Outlier', 'Non-Outlier'
        )
        df[f'{column}_winsorized'] = winsorize_series(df[column], lower_bound, upper_bound)

        processed_dfs.append(df[[column, f'{column}_category', f'{column}_lower_bound', f'{column}_upper_bound', f'{column}_winsorized']])

        print(f"Processed Data for {column}:")
        print(df[[column, f'{column}_category', f'{column}_lower_bound', f'{column}_upper_bound', f'{column}_winsorized']].head())

    # Combine the results for all numeric columns
    if processed_dfs:
        final_combined_df = pd.concat(processed_dfs, axis=1)
        return final_combined_df
    else:
        print("No numeric columns found to process.")
        return df  # Return the original DataFrame if no numeric columns were found


