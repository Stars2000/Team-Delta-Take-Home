import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

df = pd.read_csv("../../US_inflation.csv")

def cycle_test():
    # Utilizing Ljung-box test
    stationary_series = df["USACPALTT01CTGYM"].diff().dropna()

    lb_test = acorr_ljungbox(stationary_series, lags=[48], return_df=True)
    print("\nLjung-Box Test for Autocorrelation (H₀: No autocorrelation)")
    print(lb_test)

    # Simple interpretation
    p_value = lb_test['lb_pvalue'].iloc[0]
    if p_value <= 0.05:
        print(f"p-value = {p_value:.4f} -> Reject H₀. Significant autocorrelation remains.")
    else:
        print(f"p-value = {p_value:.4f} -> Fail to reject H₀. No significant autocorrelation.")

if __name__ == '__main__':
    cycle_test()