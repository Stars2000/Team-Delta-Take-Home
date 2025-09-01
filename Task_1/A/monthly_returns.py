import pandas as pd

def load_df():
    df = pd.read_csv('../bitcoin_spx.csv')
    df.set_index('Date', inplace=True)

    # Convert all other columns to numeric (just in case)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def calculate_monthly_returns(df):
    monthly_returns = df.pct_change().dropna() * 100
    monthly_returns.columns = [f'{col} Monthly Returns' for col in monthly_returns.columns]
    return monthly_returns

def save_returns_df(df):
    df.to_csv('../Results/monthly_returns.csv', index=True)

if __name__ == "__main__":
    df = load_df()
    monthly_returns = calculate_monthly_returns(df)
    save_returns_df(monthly_returns)