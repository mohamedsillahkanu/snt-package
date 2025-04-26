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
    # Step 1: Read and sort compute instructions
    comp = pd.read_excel(compute_path)

    # Separate original variables (operation is NaN) and computed variables (operation is not NaN)
    originals = comp[comp['operation'].isna()]
    computed = comp[comp['operation'].notna()]

    # New list for sorted rows
    sorted_rows = []
    added = set()

    for idx, row in computed.iterrows():
        # Add components first
        components = [x.strip() for x in str(row['components']).split(',')]
        for comp_var in components:
            if comp_var not in added:
                sorted_rows.append({'new_variable': comp_var, 'operation': None, 'components': None})
                added.add(comp_var)
        
        # Add computed variable
        if row['new_variable'] not in added:
            sorted_rows.append(row)
            added.add(row['new_variable'])

    # Convert back to DataFrame
    sorted_comp = pd.DataFrame(sorted_rows)

    # Step 2: Now compute based on sorted instructions
    for i in range(len(sorted_comp)):
        new_var = sorted_comp['new_variable'][i]
        op = sorted_comp['operation'][i]
        components = sorted_comp['components'][i]

        if pd.isna(op) or pd.isna(components):
            continue  # skip if no operation

        components = [x.strip() for x in components.split(',')]

        if op == "rowsum":
            df[new_var] = df[components].sum(axis=1, skipna=True)
        elif op == "subtract":
            df[new_var] = df[components[0]] - df[components[1]]
            df[new_var] = df[new_var].clip(lower=0)

    return df


def split(df, split_path):
    split_cols = pd.read_excel(split_path)['columns'].tolist()
    id_vars = [col for col in df.columns if col not in split_cols]
    return pd.melt(df, id_vars=id_vars, value_vars=split_cols,
                   var_name='variable', value_name='value')
