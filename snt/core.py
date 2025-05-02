# snt/core.py
import pandas as pd
from pathlib import Path
import numpy as np

def concatenate():
    files = Path("input_files/routine").glob("*.xls")
    df_list = [pd.read_excel(file) for file in files]
    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df.head()
    return combined_df

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
    routine_data_path,
    population_data_path,
    output_folder='epi_output',
    output_filename='adjusted_incidence_with_mean_median.xlsx'
):
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, output_filename)

    # Load input data
    routine_data = pd.read_excel(routine_data_path)
    population_data = pd.read_excel(population_data_path)
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
    os.makedirs("epi_maps", exist_ok=True)

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
        fig, axes = plt.subplots(1, len(columns), figsize=(len(columns)*3.5, 5))
        if len(columns) == 1:
            axes = [axes]

        for (col, year), ax in zip(columns, axes):
            gdf.plot(column=col, cmap=cmap, norm=norm, edgecolor='gray', linewidth=0.5,
                     ax=ax, legend=False, missing_kwds={"color": "lightgrey"})
            gdf.dissolve(by="FIRST_DNAM").boundary.plot(ax=ax, color="black", linewidth=1)

            data = gdf[col].dropna()
            counts, _ = np.histogram(data, bins=bins)
            legend_labels = [f"{label} ({count})" for label, count in zip(labels, counts)]
            legend_items = [Patch(facecolor=cmap(norm(b)), edgecolor='black', label=lab)
                            for b, lab in zip(bins[:-1], legend_labels)]

            ax.legend(handles=legend_items, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=7, title="Cases per 1000")
            ax.set_title(year, fontsize=14)
            ax.axis("off")

        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        output_path = f"epi_maps/{prefix.rstrip('_')}_maps.png"
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"[Saved] {output_path}")

