# snt/core.py
import pandas as pd
from pathlib import Path
import numpy as np
from tabulate import tabulate

from pathlib import Path
import pandas as pd
from tabulate import tabulate
import math

def concatenate():
    # Combine the files
    files = Path("input_files/routine").glob("*.xls")
    df_list = [pd.read_excel(file) for file in files]
    combined_df = pd.concat(df_list, ignore_index=True)

    # Print head
    print("\n=== Preview of Combined Data ===")
    print(tabulate(combined_df.tail(), headers='keys', tablefmt='grid'))

    # Format column names into 3 columns
    print("\n=== Column Names (3 per row) ===")
    columns = list(combined_df.columns)
    padded_cols = columns + [""] * ((3 - len(columns) % 3) % 3)  # pad to multiple of 3
    col_table = [padded_cols[i:i+3] for i in range(0, len(padded_cols), 3)]
    print(tabulate(col_table, headers=["Column 1", "Column 2", "Column 3"], tablefmt="grid"))

    return combined_df

def rename(df):
    name_map = pd.read_excel("input_files/others/old_new_rename.xlsx")
    rename_dict = dict(zip(name_map.iloc[:, 0], name_map.iloc[:, 1]))
    return df.rename(columns=rename_dict)
    
def compute(df):
    try:
        comp = pd.read_excel("input_files/others/compute new variables_python.xlsx")
    except Exception as e:
        raise FileNotFoundError(f"Error reading compute file: {e}")

    for i in range(len(comp)):
        new_var = comp.at[i, 'new_variable']
        op = comp.at[i, 'operation']
        components = [x.strip() for x in comp.at[i, 'components'].split(',')]

        # Check if all required columns exist in df
        missing_cols = [col for col in components if col not in df.columns]
        if missing_cols:
            print(f"Skipping '{new_var}' — missing columns: {missing_cols}")
            continue

        if op == "rowsum":
            df[new_var] = df[components].sum(axis=1, skipna=True, min_count=1)
        elif op == "subtract" and len(components) >= 2:
            df[new_var] = df[components[0]] - df[components[1]]
            df[new_var] = df[new_var].clip(lower=0)
        else:
            print(f"Skipping '{new_var}' — unsupported operation or insufficient components.")
    
    return df

def sort(df):
    try:
        comp = pd.read_excel("input_files/others/compute new variables_python.xlsx")
    except Exception as e:
        raise FileNotFoundError(f"Could not read compute file: {e}")

    sorted_columns = []

    # Collect components and new variables in order
    for i in range(len(comp)):
        components = [x.strip() for x in str(comp.at[i, 'components']).split(',')]
        new_var = comp.at[i, 'new_variable']
        sorted_columns.extend(components)
        sorted_columns.append(new_var)

    # Ensure uniqueness and keep only existing columns
    sorted_columns = [col for col in dict.fromkeys(sorted_columns) if col in df.columns]

    # Add any remaining columns not in the sort list
    remaining_columns = [col for col in df.columns if col not in sorted_columns]

    # Final column order
    final_order = remaining_columns + sorted_columns

    # Reorder DataFrame
    return df[final_order]


def split(df):
    try:
        # Read the mapping file
        mapping = pd.read_excel("input_files/others/split colums.xlsx")
    except Exception as e:
        raise FileNotFoundError(f"Could not read split columns file: {e}")

    # Validate expected columns
    required_cols = {'original_col', 'new_col_month', 'new_col_year'}
    if not required_cols.issubset(mapping.columns):
        raise ValueError(f"Missing expected columns in mapping file: {required_cols - set(mapping.columns)}")

    original_col = mapping.at[0, 'original_col']
    new_col_month = mapping.at[0, 'new_col_month']
    new_col_year = mapping.at[0, 'new_col_year']

    # Ensure original column exists in df
    if original_col not in df.columns:
        raise KeyError(f"Column '{original_col}' not found in DataFrame.")

    # Split the column
    split_data = df[original_col].astype(str).str.strip().str.split(' ', n=1, expand=True)

    # Assign new columns
    df[new_col_month] = split_data[0].str.strip()
    df[new_col_year] = split_data[1].str.strip() if split_data.shape[1] > 1 else None

    # Month name to number mapping (English and French)
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

    # Standardize month
    df[new_col_month] = df[new_col_month].map(lambda x: month_map.get(x, x))

    return df


### Outlier detection and correctio with winsorized method
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

# Function to process a single column (grouped by adm1, adm2, adm3, hf, year)
def process_column_winsorization(df, column):
    grouped = df.groupby(['adm1', 'adm2', 'adm3', 'hf', 'year'])
    results = []

    for (adm1, adm2, adm3, hf, year), group in grouped:
        group = group.copy()
        lower_bound, upper_bound = detect_outliers_scatterplot(group, column)

        group[f'{column}_lower_bound'] = lower_bound
        group[f'{column}_upper_bound'] = upper_bound
        group[f'{column}_category'] = np.where(
            (group[column] < lower_bound) | (group[column] > upper_bound),
            'Outlier',
            'Non-Outlier'
        )
        group[f'{column}_winsorized'] = winsorize_series(group[column], lower_bound, upper_bound)

        results.append(group)

    final_df = pd.concat(results)

    export_columns = [
        'adm1', 'adm2', 'adm3', 'hf', 'year', 'month', column,
        f'{column}_category', f'{column}_lower_bound', f'{column}_upper_bound',
        f'{column}_winsorized'
    ]
    export_columns = [col for col in export_columns if col in final_df.columns]

    return final_df[export_columns]

# Main function to process multiple columns and merge the results
def detect_outliers(df):
    columns_to_process = ['allout', 'susp', 'test', 'conf', 'maltreat', 'pres', 'maladm', 'maldth']
    processed_dfs = []

    for column in columns_to_process:
        if column not in df.columns:
            continue
        if df[column].isnull().all():
            continue

        processed_df = process_column_winsorization(df, column)
        processed_dfs.append(processed_df)

    if processed_dfs:
        merge_keys = ['adm1', 'adm2', 'adm3', 'hf', 'year', 'month']
        final_combined_df = processed_dfs[0]

        for df_to_merge in processed_dfs[1:]:
            final_combined_df = final_combined_df.merge(
                df_to_merge, on=merge_keys, how='outer', suffixes=('', '_dup')
            )
            final_combined_df = final_combined_df[[col for col in final_combined_df.columns if not col.endswith('_dup')]]

        return final_combined_df
    else:
        return None

import pandas as pd
from tabulate import tabulate

def outlier_summary(df):
    # Automatically detect columns ending with '_category'
    category_columns = [col for col in df.columns if col.endswith('_category')]
    
    summary_stats = {}

    for col in category_columns:
        total_outliers = (df[col] == 'Outlier').sum()
        total_non_outliers = (df[col] == 'Non-Outlier').sum()
        total = total_outliers + total_non_outliers

        if total > 0:
            outlier_percentage = (total_outliers / total) * 100
            non_outlier_percentage = (total_non_outliers / total) * 100
        else:
            outlier_percentage = 0
            non_outlier_percentage = 0

        summary_stats[col] = {
            'Total Outliers': total_outliers,
            'Total Non-Outliers': total_non_outliers,
            'Total Records': total,
            'Outlier Percentage': f"{outlier_percentage:.2f}%",
            'Non-Outlier Percentage': f"{non_outlier_percentage:.2f}%"
        }

    summary_df = pd.DataFrame(summary_stats).T

    # Print in a pretty table
    print(tabulate(summary_df, headers='keys', tablefmt='pretty'))

    return summary_df

### Outlier detection after correction with winsorized method
import pandas as pd
import numpy as np

# Function to detect outliers using Scatterplot with Q1 and Q3 lines
def detect_outliers_scatterplot(df, col):
    
    # Calculate Q1 and Q3
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    
    # Calculate the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    return lower_bound, upper_bound

# Function to apply winsorization to a column
def winsorize_series(series, lower_bound, upper_bound):
  
    # Clip the values that are outside the bounds
    return series.clip(lower=lower_bound, upper=upper_bound)

# Function to process a column and return a DataFrame with winsorized data
def process_column_winsorization(df, column):
 
    # Group by 'adm1', 'adm2', 'adm3', 'hf', 'year' for processing each group separately
    grouped = df.groupby(['adm1', 'adm2', 'adm3', 'hf', 'year'])
    results = []

    # Process each group
    for (adm1, adm2, adm3, hf, year), group in grouped:
        # Detect outliers
        lower_bound, upper_bound = detect_outliers_scatterplot(group, column)
        
        # Add new columns for outlier boundaries, category, and winsorized data
        group[f'{column}_lower_bound'] = lower_bound
        group[f'{column}_upper_bound'] = upper_bound
        group[f'{column}_category'] = np.where(
            (group[column] < lower_bound) | (group[column] > upper_bound), 'Outlier', 'Non-Outlier'
        )

        
        # Append the processed group to the results list
        results.append(group)

    # Concatenate all the processed groups
    final_df = pd.concat(results)
    
    # Define the columns to export
    export_columns = [
        'adm1', 'adm2', 'adm3', 'hf', 'year', 'month', column,
        f'{column}_category', f'{column}_lower_bound', f'{column}_upper_bound',
       
    ]
    
    # Filter to include only the existing columns in the DataFrame
    export_columns = [col for col in export_columns if col in final_df.columns]
    
    return final_df[export_columns]

# Main function to process multiple columns and merge the results
def detect_outliers_after_correction(df):
    # List of columns to process
    columns_to_process = ['allout_winsorized', 'susp_winsorized', 'test_winsorized', 'conf_winsorized', 'maltreat_winsorized', 'pres_winsorized', 'maladm_winsorized', 'maldth_winsorized']
    processed_dfs = []

    # Loop through each column and process it
    for column in columns_to_process:
        if column not in df.columns:
            print(f"Skipping column {column} as it does not exist in the dataset.")
            continue
        if df[column].isnull().all():
            print(f"Skipping column {column} as it contains only missing values.")
            continue

        print(f"Processing column: {column}")
        processed_df = process_column_winsorization(df, column)
        processed_dfs.append(processed_df)

    # Merge the processed DataFrames
    if processed_dfs:
        merge_keys = ['adm1', 'adm2', 'adm3', 'hf', 'year', 'month']
        final_combined_df = processed_dfs[0]
        for df_to_merge in processed_dfs[1:]:
            final_combined_df = final_combined_df.merge(df_to_merge, on=merge_keys, how='outer')
        
        return final_combined_df
    else:
        print("No valid columns were processed.")
        return None

import pandas as pd
from tabulate import tabulate

def outlier_summary_after_correction(df):
    # Automatically detect columns ending with '_category'
    category_columns = [col for col in df.columns if col.endswith('_category')]
    
    summary_stats = {}

    for col in category_columns:
        total_outliers = (df[col] == 'Outlier').sum()
        total_non_outliers = (df[col] == 'Non-Outlier').sum()
        total = total_outliers + total_non_outliers

        if total > 0:
            outlier_percentage = (total_outliers / total) * 100
            non_outlier_percentage = (total_non_outliers / total) * 100
        else:
            outlier_percentage = 0
            non_outlier_percentage = 0

        summary_stats[col] = {
            'Total Outliers': total_outliers,
            'Total Non-Outliers': total_non_outliers,
            'Total Records': total,
            'Outlier Percentage': f"{outlier_percentage:.2f}%",
            'Non-Outlier Percentage': f"{non_outlier_percentage:.2f}%"
        }

    summary_df = pd.DataFrame(summary_stats).T

    # Print in a pretty table
    print(tabulate(summary_df, headers='keys', tablefmt='pretty'))

    return summary_df


# Epi stratification

import pandas as pd
import numpy as np
import geopandas as gpd
from functools import reduce
import os

def epi_stratification(
    output_folder='epi_output',
    output_filename='adjusted_incidence_with_mean_median.xlsx'
):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, output_filename)

    # Load input data
    routine_data = pd.read_excel("input_files/routine/clean_data/routine_data (1).xlsx")
    population_data = pd.read_excel("input_files/routine/population_data/population_data.xlsx")
    df = routine_data.copy()

    # Preprocess dates
    df['date'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m').dt.to_period('M')
    df['Total_Reports'] = df[['allout', 'susp', 'test', 'conf', 'maltreat']].sum(axis=1)

    # Dynamic year range
    start = df['date'].min().year
    end = df['date'].max().year
    years = range(start, end + 1)

    # First reporting date
    df_active = df[df['Total_Reports'] > 0]
    first_report_dates = df_active.groupby(['adm1', 'adm2', 'adm3', 'hf'])['date'].min().reset_index()
    first_report_dates.rename(columns={'date': 'First_Reported_Date'}, inplace=True)

    # Reporting stats
    reporting_stats_by_year = []
    for year in years:
        df_year = df[df['year'] == year]
        reported = (
            df_year[df_year['conf'] > 0]
            .groupby(['adm1', 'adm2', 'adm3'], as_index=False)['conf']
            .count()
            .rename(columns={'conf': f'Times_Reported_{year}'})
        )
        expected = (
            first_report_dates
            .assign(Times_Expected=lambda x: np.where(
                x['First_Reported_Date'].dt.year == year,
                12 - x['First_Reported_Date'].dt.month + 1,
                np.where(year > x['First_Reported_Date'].dt.year, 12, 0)
            ))
            .groupby(['adm1', 'adm2', 'adm3'], as_index=False)['Times_Expected']
            .sum()
            .rename(columns={'Times_Expected': f'Times_Expected_To_Report_{year}'})
        )
        stats = pd.merge(expected, reported, on=['adm1', 'adm2', 'adm3'], how='outer')
        stats[f'Times_Reported_{year}'] = stats[f'Times_Reported_{year}'].fillna(0)
        stats[f'Times_Expected_To_Report_{year}'] = stats[f'Times_Expected_To_Report_{year}'].fillna(0)
        stats[f'conf_RR_{year}'] = (
            stats[f'Times_Reported_{year}']
            .div(stats[f'Times_Expected_To_Report_{year}'])
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
            .round(2)
            .mul(100)
        )
        reporting_stats_by_year.append(stats)

    confirmed_data = reduce(
        lambda left, right: pd.merge(left, right, on=['adm1', 'adm2', 'adm3'], how='outer'),
        reporting_stats_by_year
    )

    # Aggregated routine data by year
    dfs = []
    for year in years:
        df_year = df[df['year'] == year]
        grouped = df_year.groupby(['adm1', 'adm2', 'adm3'], as_index=False)[['conf', 'test', 'pres']].sum()
        grouped = grouped.rename(columns={
            'conf': f'conf_{year}', 'test': f'test_{year}', 'pres': f'pres_{year}'
        })
        dfs.append(grouped)

    df_merge = reduce(lambda left, right: pd.merge(left, right, on=['adm1', 'adm2', 'adm3'], how='outer'), dfs)

    # Merge all
    df1 = df_merge.merge(confirmed_data, on=['adm1', 'adm2', 'adm3'], how='left', validate='1:1')
    data = df1.merge(population_data, on='adm3', how='left', validate='1:1')
  
    # Compute metrics
    for year in years:
        conf_col = f"conf_{year}"
        test_col = f"test_{year}"
        pop_col = f"pop{year}"
        pres_col = f"pres_{year}"
        conf_RR_col = f"conf_RR_{year}"

        if not all(col in data.columns for col in [conf_col, test_col, pop_col, pres_col, conf_RR_col]):
            continue

        data[f'TPR_{year}'] = data[conf_col].div(data[test_col])
        data[f'crude_incidence_{year}'] = data[conf_col].div(data[pop_col]).mul(1000)
        data[f'presumed_adjusted_case_{year}'] = data[conf_col].add(data[pres_col].mul(data[f'TPR_{year}']))
        data[f'adjusted1_{year}'] = data[f'presumed_adjusted_case_{year}'].div(data[pop_col]).mul(1000)
        data[f'presumed_adjusted_case_RR_{year}'] = data[f'presumed_adjusted_case_{year}'].div(data[conf_RR_col])
        data[f'adjusted2_{year}'] = data[f'presumed_adjusted_case_RR_{year}'].div(data[pop_col]).mul(1000)
        data[f'adjusted3_{year}'] = (
            data[f'adjusted2_{year}']
            .add(data[f'adjusted2_{year}'].mul(data['CSpr']).div(data['CSpu']))
            .add(data[f'adjusted2_{year}'].mul(data['CSn']).div(data['CSpu']))
            .div(data[pop_col])
            .mul(1000)
        )

    # Summary stats
    for prefix in ['adjusted1', 'adjusted2', 'adjusted3']:
        cols = [f'{prefix}_{year}' for year in years if f'{prefix}_{year}' in data.columns]
        data[f'{prefix}_mean'] = data[cols].mean(axis=1)
        data[f'{prefix}_median'] = data[cols].median(axis=1)

    # Save
    data.to_excel(output_file, index=False)
    print(f"Data has been successfully saved to {output_folder}")
    return data

# Epi plots (individual)
import os
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

def merge_data_with_shapefile(df1, shapefile):
    gdf = shapefile.merge(df1, on=['FIRST_DNAM', 'FIRST_CHIE'], how='left', validate='1:1')
    return gdf

def individual_plots(epi_data_path,
                     shapefile_path,
                     prefixes=['crude_incidence_', 'adjusted1_', 'adjusted2_', 'adjusted3_'],
                     colormap='RdYlBu_r',
                     edge_color='gray',
                     bins=[0, 50, 100, 250, 450, 700, 1000, float('inf')],
                     bin_labels=['<50', '50-100', '100-250', '250-450', '450-700', '700-1000', '>1000'],
                     output_root='epi_maps'):
    """
    Creates individual maps for each column with a valid prefix and 4-digit year.
    Saves each map in a subfolder named after the prefix inside the 'epi_maps' folder.
    """

    # Load input data
    df1 = pd.read_excel(epi_data_path)
    shapefile = gpd.read_file(shapefile_path)
    os.makedirs(output_root, exist_ok=True)

    # Merge data
    gdf = merge_data_with_shapefile(df1, shapefile)

    # Detect valid columns
    pattern = re.compile(r'_(\d{4})$')
    columns_to_plot = []
    for col in gdf.columns:
        for prefix in prefixes:
            if col.startswith(prefix) and pattern.search(col):
                columns_to_plot.append((col, prefix))
                break

    if not columns_to_plot:
        print("No valid columns found.")
        return

    # Setup color map
    cmap = plt.cm.get_cmap(colormap, len(bins) - 1)
    norm = BoundaryNorm(bins, ncolors=cmap.N)

    for column_name, prefix in columns_to_plot:
        fig, ax = plt.subplots(figsize=(10, 10))

        valid_data = gdf[column_name].dropna()
        counts, _ = np.histogram(valid_data, bins=bins)
        bin_labels_with_counts = [f"{label} ({count})" for label, count in zip(bin_labels, counts)]

        gdf.plot(
            column=column_name,
            cmap=cmap,
            norm=norm,
            edgecolor=edge_color,
            linewidth=0.5,
            legend=False,
            ax=ax,
            missing_kwds={'color': 'lightgrey', 'edgecolor': 'white', 'linewidth': 0.3}
        )

        district_boundaries = gdf.dissolve(by='FIRST_DNAM')
        district_boundaries.boundary.plot(ax=ax, color='gray', linewidth=1.0, zorder=2)

        legend_elements = [
            Patch(facecolor=cmap(norm(bin_start)), edgecolor='black', label=label)
            for bin_start, label in zip(bins[:-1], bin_labels_with_counts)
        ]

        ax.legend(
            handles=legend_elements,
            loc='lower right',
            title="Cases per 1000",
            fontsize=9,
            title_fontsize=10,
            frameon=True,
            framealpha=1.0,
            ncol=1
        )

        ax.set_title(column_name.replace("_", " "), fontsize=14, pad=10)
        ax.axis("off")

        # Subfolder for each prefix
        prefix_folder = os.path.join(output_root, prefix.rstrip("_"))
        os.makedirs(prefix_folder, exist_ok=True)

        output_file = os.path.join(prefix_folder, f"{column_name}.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved: {column_name}.png to {prefix_folder}")
        plt.close()


# Subplots
import os
import re
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.patches import Patch
import numpy as np

def subplots(epi_data_path, shapefile_path):
    prefixes = ['crude_incidence_', 'adjusted1_', 'adjusted2_', 'adjusted3_']
    os.makedirs("subplots", exist_ok=True)
    
    df1 = pd.read_excel(epi_data_path)
    gdf_shape = gpd.read_file(shapefile_path)
    gdf = gdf_shape.merge(df1, on=["FIRST_DNAM", "FIRST_CHIE"], how="left", validate="1:1")
    
    bins = [0, 50, 100, 250, 450, 700, 1000, float("inf")]
    labels = ['<50', '50-100', '100-250', '250-450', '450-700', '700-1000', '>1000']
    cmap = plt.cm.get_cmap("RdYlBu_r", len(bins)-1)
    norm = BoundaryNorm(bins, cmap.N)
    
    for prefix in prefixes:
        pattern = re.compile(f"^{re.escape(prefix)}(\\d{{4}})$")
        columns = [(col, pattern.match(col).group(1)) for col in gdf.columns if pattern.match(col)]
        
        if not columns:
            print(f"[Skipped] No columns found for prefix '{prefix}'")
            continue
            
        columns.sort(key=lambda x: x[1])
        
        # Create a 3x3 grid
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        
        # Hide any unused axes
        for i in range(len(columns), 9):
            axes[i].set_visible(False)
        
        for i, ((col, year), ax) in enumerate(zip(columns, axes)):
            gdf.plot(column=col, cmap=cmap, norm=norm, edgecolor='gray', linewidth=0.5, ax=ax, legend=False, missing_kwds={"color": "lightgrey"})
            gdf.dissolve(by="FIRST_DNAM").boundary.plot(ax=ax, color="black", linewidth=1)
            
            data = gdf[col].dropna()
            counts, _ = np.histogram(data, bins=bins)
            legend_labels = [f"{label} ({count})" for label, count in zip(labels, counts)]
            ax.set_title(year, fontsize=14)
            ax.axis("off")
        
        # Create legend on the right side center
        legend_items = [Patch(facecolor=cmap(norm(b)), edgecolor='black', label=lab) for b, lab in zip(bins[:-1], legend_labels)]
        fig.legend(handles=legend_items, loc='center right', bbox_to_anchor=(1.15, 0.5), fontsize=10, title="Cases per 1000")
        
        plt.tight_layout()
        output_path = f"subplots/{prefix.rstrip('_')}_maps.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Saved] {output_path}")




## Line plots
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import os

def epi_trends(path, output_folder='epi_lineplots'):
    os.makedirs(output_folder, exist_ok=True)

    # Read the Excel file
    df = pd.read_excel(path)

    # Define prefixes and colors
    prefixes = ['crude_incidence', 'adjusted1', 'adjusted2', 'adjusted3']
    colors = ['blue', 'green', 'orange', 'red']

    # Get list of years from column names
    pattern = re.compile(r'^crude_incidence_(\d{4})$')
    years = sorted(int(pattern.match(col).group(1)) for col in df.columns if pattern.match(col))

    # Loop through each district (adm1 = FIRST_DNAM)
    for district in df['FIRST_DNAM'].dropna().unique():
        df_district = df[df['FIRST_DNAM'] == district]
        chiefdoms = df_district['FIRST_CHIE'].dropna().unique()
        n = len(chiefdoms)

        n_cols = 3
        n_rows = int(np.ceil(n / n_cols))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4), sharex=True, sharey=True)
        axes = axes.flatten()

        for i, chiefdom in enumerate(chiefdoms):
            ax = axes[i]
            row = df_district[df_district['FIRST_CHIE'] == chiefdom]

            if row.empty:
                ax.set_title(f"{chiefdom} (No data)")
                ax.axis("off")
                continue

            for prefix, color in zip(prefixes, colors):
                cols = [f"{prefix}_{year}" for year in years if f"{prefix}_{year}" in row.columns]
                values = row[cols].values.flatten()

                if len(values) != len(years) or all(pd.isna(values)):
                    continue  # Skip if no valid data

                ax.plot(years, values, marker='o', label=prefix.replace('_', ' ').title(), color=color)

            ax.set_title(chiefdom, fontsize=10)
            ax.grid(True)
            ax.tick_params(axis='x', rotation=45)

        # Turn off unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        # Shared legend
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title="Indicator", loc="lower center", ncol=4)

        fig.suptitle(f"Incidence Trends by Chiefdom - {district}", fontsize=14)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        filename = os.path.join(output_folder, f"{district}_trends.png")
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"[Saved] {filename}")

    return df
## 
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from docx import Document
from docx.shared import Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
import datetime

def export_and_interpret(path,
    report_folder="final_report",
    report_title="Malaria Epidemiological Analysis Report",
    author="Malaria Surveillance Team"
):
    os.makedirs(report_folder, exist_ok=True)
    epi_data = pd.read_excel(path)
    doc = Document()

    # Title and author
    doc.add_heading(report_title, level=0)
    p = doc.add_paragraph()
    p.add_run(f"Prepared by: {author}").bold = True
    p.add_run(f"\nDate: {datetime.datetime.now().strftime('%B %d, %Y')}")

    # Introduction
    doc.add_heading("Introduction", level=1)
    doc.add_paragraph(
        "This report presents the results of malaria epidemiological analysis using routine "
        "surveillance data. The analysis includes data cleaning, outlier detection, "
        "incidence calculation with various adjustment methods, and geographic distribution "
        "visualization. This document provides interpretations and recommendations based on "
        "the findings."
    )

    # Methods
    doc.add_heading("Methods", level=1)
    doc.add_paragraph(
        "The analysis workflow involved several steps:\n"
        "1. Data concatenation and cleaning from routine surveillance files\n"
        "2. Outlier detection using IQR method and winsorization for correction\n"
        "3. Calculation of crude and adjusted incidence rates\n"
        "4. Visualization of geographic distribution of malaria burden\n"
        "5. Statistical summary and interpretation of findings"
    )

    # Dynamic interpretation function
    def generate_dynamic_interpretations(df):
        interpretations = {}
        summary_stats = {}

        for prefix in ["crude_incidence_", "adjusted1_", "adjusted2_", "adjusted3_"]:
            incidence_cols = [col for col in df.columns if col.startswith(prefix) and col.replace(prefix, "").isdigit()]
            if not incidence_cols:
                continue

            max_vals = df[incidence_cols].max(axis=1)
            low = (max_vals < 250).sum()
            moderate = ((max_vals >= 250) & (max_vals <= 450)).sum()
            high = (max_vals > 450).sum()
            total = len(max_vals)

            pct_low = low / total * 100
            pct_moderate = moderate / total * 100
            pct_high = high / total * 100

            type_name = prefix.rstrip("_").replace("_", " ").title()
            summary_stats[type_name] = {
                "Low": pct_low,
                "Moderate": pct_moderate,
                "High": pct_high
            }

            interpretation = f"{type_name} analysis shows that approximately {pct_high:.1f}% of chiefdoms experience high transmission (>450), " \
                            f"{pct_moderate:.1f}% moderate (250–450), and {pct_low:.1f}% low (<250). "

            if pct_high > 40:
                interpretation += "This indicates a critical need for intensified malaria control interventions in a significant proportion of areas."
            elif pct_moderate > 50:
                interpretation += "Moderate burden dominates, suggesting targeted scale-up of prevention and treatment may yield high impact."
            else:
                interpretation += "Most areas are experiencing low or moderate transmission, providing an opportunity to push toward elimination in specific regions."

            interpretations[type_name] = interpretation

        return pd.DataFrame(summary_stats).T, interpretations

    # Generate interpretations
    df_summary, dynamic_interps = generate_dynamic_interpretations(epi_data)

    # Add summary to document
    doc.add_heading("Transmission Intensity Summary", level=1)
    doc.add_paragraph(
        "This section summarizes the percentage of chiefdoms with low, moderate, and high malaria transmission. "
        "These statistics are based on the highest annual incidence observed per chiefdom."
    )

    # Insert table
    table = doc.add_table(rows=1, cols=4)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    hdr[0].text = "Incidence Type"
    hdr[1].text = "Low (%)"
    hdr[2].text = "Moderate (%)"
    hdr[3].text = "High (%)"

    for index, row in df_summary.iterrows():
        cells = table.add_row().cells
        cells[0].text = index
        cells[1].text = f"{row['Low']:.1f}"
        cells[2].text = f"{row['Moderate']:.1f}"
        cells[3].text = f"{row['High']:.1f}"

    # Add dynamic interpretations
    doc.add_heading("Interpretation and Recommendations", level=1)
    for inc_type, text in dynamic_interps.items():
        doc.add_heading(inc_type, level=2)
        doc.add_paragraph(text)

    # Save document
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(report_folder, f"Malaria_Analysis_Report_{timestamp}.docx")
    doc.save(output_file)

    print(f"✅ Report saved to: {output_file}")
    return output_file


