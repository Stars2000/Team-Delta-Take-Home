import pandas as pd
import plotly.express as px
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

df = pd.read_csv("../../US_inflation.csv")

def seasonality_test():
    # Visual test
    plot_monthly_averages()

    # Statistical test - using ACF on differenced data
    acf_seasonality_test()


def plot_monthly_averages():
    stationary_series = df["USACPALTT01CTGYM"].diff().dropna()

    months = pd.to_datetime(df['observation_date'].iloc[1:]).dt.month

    temp_df = pd.DataFrame({
        'change': stationary_series.values,
        'month': months.values
    })

    monthly_averages = temp_df.groupby('month')['change'].mean().reset_index()

    fig = px.bar(monthly_averages, x='month', y='change',
                 title='Average Monthly Change in YoY Inflation',
                 labels={'change': 'Average Change', 'month': 'Month'})

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig.update_xaxes(tickvals=list(range(1, 13)), ticktext=month_names)

    fig.show()


def acf_seasonality_test(period=12, alpha=0.05):
    stationary_series = df["USACPALTT01CTGYM"].diff().dropna()

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_acf(stationary_series, lags=36, ax=ax, alpha=alpha)
    plt.title('Autocorrelation Function (ACF) of Stationary Series - Seasonal Lags Highlighted')
    for l in range(period, 37, period):
        ax.axvline(x=l, color='red', linestyle='--', alpha=0.3)
    plt.show()

if __name__ == '__main__':
    seasonality_test()