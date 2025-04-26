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
def outliers(df, group_column_path):
    # Read group columns
    group_columns = pd.read_excel(group_column_path)['grouped_columns'].dropna().tolist()
    
    # Key variables for summary reporting only
    key_variables = ['allout', 'susp', 'test', 'conf', 'maltreat', 'pres']
    
    # Filter to only include existing key variables
    key_variables = [col for col in key_variables if col in df.columns]
    
    # Get all numeric columns for outlier detection
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    
    # Exclude 'month' if present
    numeric_cols = [col for col in numeric_cols if col.lower() != 'month']

    # Initialize outlier statistics for key variables
    variable_stats = {
        'Year': [],
        'Variable': [],
        'Outliers Before': [],
        'Outliers After': []
    }

    # Create a copy of df to modify
    df_corrected = df.copy()
    
    # Assume 'year' is one of the columns in your dataframe
    years = df['year'].unique() if 'year' in df.columns else [None]
    
    # Process each year
    for year in years:
        # Filter data for this year if year column exists
        if year is not None:
            year_data = df[df['year'] == year]
        else:
            year_data = df
        
        # Group the year data
        grouped = year_data.groupby(group_columns)
        
        # Track outliers for key variables in this year
        year_var_outliers_before = {var: 0 for var in key_variables}
        year_var_outliers_after = {var: 0 for var in key_variables}
        
        # Process each group
        for group_name, group_data in grouped:
            for col in numeric_cols:  # Process ALL numeric columns
                if col in group_data.columns:
                    series_nonan = group_data[col].dropna()
                    
                    if not series_nonan.empty:
                        Q1 = series_nonan.quantile(0.25)
                        Q3 = series_nonan.quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Get the indices of this group
                        idx = group_data.index
                        
                        # Count outliers before correction
                        outliers_before = ((df_corrected.loc[idx, col] < lower_bound) | 
                                          (df_corrected.loc[idx, col] > upper_bound)).sum()
                        
                        # Clip values for this group
                        df_corrected.loc[idx, col] = df_corrected.loc[idx, col].clip(lower=lower_bound, upper=upper_bound)
                        
                        # Count outliers after correction
                        outliers_after = ((df_corrected.loc[idx, col] < lower_bound) | 
                                         (df_corrected.loc[idx, col] > upper_bound)).sum()
                        
                        # Track statistics only for key variables
                        if col in key_variables:
                            year_var_outliers_before[col] += outliers_before
                            year_var_outliers_after[col] += outliers_after
        
        # Record statistics for key variables in this year
        for key_var in key_variables:
            if year_var_outliers_before[key_var] > 0:  # Only include if outliers were found
                variable_stats['Year'].append(year if year is not None else 'All')
                variable_stats['Variable'].append(key_var)
                variable_stats['Outliers Before'].append(year_var_outliers_before[key_var])
                variable_stats['Outliers After'].append(year_var_outliers_after[key_var])

    # Create summary DataFrame
    summary_df = pd.DataFrame(variable_stats)
    print("\nOutlier Summary for Key Variables by Year:")
    print(summary_df)

    return df_corrected




