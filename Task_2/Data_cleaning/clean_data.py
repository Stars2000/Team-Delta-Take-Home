import pandas as pd
import os

dirpath = "../Extra_data"

def get_file_names(dirpath):
    return [f for f in os.listdir(dirpath) if f.endswith(".csv")]

def load_and_prepare(filepath):
    file_name = os.path.basename(filepath)

    # Parse dates depending on file
    if file_name == "SPX.csv":
        df = pd.read_csv(filepath, parse_dates=["observation_date"], dayfirst=False)
        df["observation_date"] = pd.to_datetime(df["observation_date"], format="%m/%d/%Y")
        df = df.sort_values("observation_date")
    else:
        df = pd.read_csv(filepath, parse_dates=["observation_date"])

    df = df.set_index("observation_date")

    return df

def collect_raw_data(dirpath, file_list):
    df_list = []
    for file in file_list:
        filepath = os.path.join(dirpath, file)
        df_list.append(load_and_prepare(filepath))

    return df_list

def concact_and_clean(df_list):
    final_df = pd.concat(df_list, axis=1, join="outer")

    for col in final_df.columns:
        if final_df[col].dtype == object:
            # remove commas, dollar signs, spaces etc.
            final_df[col] = (
                final_df[col]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.strip()
            )
            # try convert to numeric (errors -> NaN)
            final_df[col] = pd.to_numeric(final_df[col], errors="coerce")


    final_df = final_df.sort_index()
    final_df = final_df.dropna(how="any")

    return final_df

def save_cleaned_data(df):
    df.to_csv("../cleaned_data.csv")

# Execution
file_list = get_file_names(dirpath)
df_list = collect_raw_data(dirpath, file_list)
df = concact_and_clean(df_list)
save_cleaned_data(df)