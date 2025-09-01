import pandas as pd

def load_df():
    df = pd.read_csv('../Results/monthly_returns.csv')
    return df

def calc_basic_stats(df):
    numeric_df = df.select_dtypes(include='number')
    for col in numeric_df.columns:
        mean = numeric_df[col].mean()
        std = numeric_df[col].std()
        print(f'{col}:\n - Mean: {mean:.2f}%\n - Std Dev: {std:.2f}%\n')

if __name__ == '__main__':
    df = load_df()
    calc_basic_stats(df)