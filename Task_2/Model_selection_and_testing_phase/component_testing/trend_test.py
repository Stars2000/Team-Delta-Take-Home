import pandas as pd
import plotly.express as px
from statsmodels.tsa.stattools import adfuller

df = pd.read_csv("../../US_inflation.csv")

def trend_test():
    # Visual inspection
    plot_time_series()

    # Statistical test
    print("\nResults from ADF without differencing")
    adf_test(df["USACPALTT01CTGYM"])

    # Apply first differencing and retest
    print("\nResults from ADF after first differencing")
    stationary_series = df["USACPALTT01CTGYM"].diff().dropna()
    adf_test(stationary_series)

def plot_time_series():
    fig = px.line(df, x="observation_date", y="USACPALTT01CTGYM", title="US Inflation")
    fig.show()

def adf_test(series):
    result = adfuller(series)

    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

    # Interpretation
    if result[1] <= 0.05:
        print("\n Reject the null hypothesis - data is stationary (no significant trend)")
    else:
        print("\n Fail to reject the null hypothesis - data is non-stationary (has a trend)")

if __name__ == '__main__':
    trend_test()