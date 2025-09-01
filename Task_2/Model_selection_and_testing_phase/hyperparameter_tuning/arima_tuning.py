import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import itertools
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("../../US_inflation.csv")
series = df["USACPALTT01CTGYM"]


def manual_arima_tuning(series, p_range, d_range, q_range):
    """
    Manually tune ARIMA parameters by grid searching over p, d, q values.
    Returns the model with the lowest AIC.
    """
    best_aic = np.inf
    best_order = None
    best_model = None

    # Generate all combinations of p, d, q
    pdq = list(itertools.product(p_range, d_range, q_range))

    print(f"Testing {len(pdq)} parameter combinations...")

    for order in pdq:
        try:
            # Fit ARIMA model
            model = ARIMA(series, order=order)
            results = model.fit()

            # Check if this model has better AIC
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = order
                best_model = results
                print(f"New best: ARIMA{order} - AIC: {results.aic:.2f}")

        except Exception as e:
            # Skip parameter combinations that don't work
            continue

    return best_model, best_order, best_aic


def main():
    p_range = range(0, 6)
    d_range = [1]
    q_range = range(0, 6)

    # Perform manual tuning
    best_model, best_order, best_aic = manual_arima_tuning(series, p_range, d_range, q_range)

    # Print results
    print("\n" + "=" * 50)
    print("MANUAL ARIMA TUNING RESULTS")
    print("=" * 50)
    print(f"Best model: ARIMA{best_order}")
    print(f"Best AIC: {best_aic:.2f}")
    print("\nModel Summary:")
    print(best_model.summary())


if __name__ == "__main__":
    main()
