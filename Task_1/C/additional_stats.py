import pandas as pd

def load_df():
    df = pd.read_csv('../Results/monthly_returns.csv')
    return df

def calc_sharpe_ratio(df):
    numeric_df = df.select_dtypes(include='number')
    for col in numeric_df.columns:
        monthly_mean = numeric_df[col].mean()
        monthly_std = numeric_df[col].std()
        yearly_mean = (pow(((monthly_mean/100) + 1),12) - 1) * 100
        yearly_std = monthly_std * pow(12,1/2)
        risk_free_rate = 4.34  # As of 25/06/2025
        sharpe_ratio = (yearly_mean - risk_free_rate) / yearly_std
        print(f'{col}:\n - Sharpe Ratio: {sharpe_ratio:.2f}\n')

if __name__ == '__main__':
    df = load_df()
    calc_sharpe_ratio(df)