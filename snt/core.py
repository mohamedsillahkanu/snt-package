# snt/core.py
import pandas as pd
from pathlib import Path

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

def detect_bounds(series):
    """Detect lower and upper bounds using IQR method."""
    series_nonan = series.dropna()
    if series_nonan.empty:
        return np.nan, np.nan
    Q1 = series_nonan.quantile(0.25)
    Q3 = series_nonan.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return lower_bound, upper_bound

def outliers(df, group_column_path):
    # Read group columns
    group_columns = pd.read_excel(group_column_path)['grouped_columns'].dropna().tolist()

    # Get all numeric columns except 'month'
    scale_cols = df.select_dtypes(include='number').columns.tolist()
    scale_cols = [col for col in scale_cols if col.lower() != 'month']

    # Copy to modify
    df_corrected = df.copy()

    # Initialize summary dict
    outlier_counts_before = {col: 0 for col in scale_cols}
    outlier_counts_after = {col: 0 for col in scale_cols}

    # Group and process
    grouped = df.groupby(group_columns)

    for _, group_data in grouped:
        idx = group_data.index

        for col in scale_cols:
            if col in group_data.columns:
                lower, upper = detect_bounds(group_data[col])

                # 🛠 Correct safe check
                if np.isnan(lower) or np.isnan(upper):
                    continue

                # Count before correction
                before_mask = (df_corrected.loc[idx, col] < lower) | (df_corrected.loc[idx, col] > upper)
                outliers_before = before_mask.sum()

                # Clip values
                df_corrected.loc[idx, col] = df_corrected.loc[idx, col].clip(lower=lower, upper=upper)

                # Count after correction
                after_mask = (df_corrected.loc[idx, col] < lower) | (df_corrected.loc[idx, col] > upper)
                outliers_after = after_mask.sum()

                # Track
                outlier_counts_before[col] += outliers_before
                outlier_counts_after[col] += outliers_after

    # Create summary DataFrame
    summary_df = pd.DataFrame({
        'Variable': list(outlier_counts_before.keys()),
        'Outliers Detected': list(outlier_counts_before.values()),
        'Outliers Corrected': [outlier_counts_before[col] - outlier_counts_after[col] for col in scale_cols]
    })

    return df_corrected, summary_df
