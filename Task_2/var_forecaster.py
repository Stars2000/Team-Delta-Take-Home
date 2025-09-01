import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')


class InflationDerivativePricer:
    def __init__(self, data_path):
        self.data = self.load_data(data_path)
        self.original_data = self.data.copy()
        self.model = None
        self.results = None
        self.forecast = None
        self.forecast_dates = None
        self.yoy_forecast = None
        self.simulations = None
        self.yoy_simulations = None
        self.stationary_data = None
        self.differencing_applied = False

    def load_data(self, data_path):
        """Load and prepare the data"""
        data = pd.read_csv(data_path, index_col=0, parse_dates=True)
        print(f"Data loaded successfully. Shape: {data.shape}")
        print(f"Data range: {data.index.min()} to {data.index.max()}")
        return data

    def check_stationarity(self, series, alpha=0.05, variable_name="Variable"):
        """Check stationarity using ADF and KPSS tests"""
        print(f"\nStationarity analysis for {variable_name}:")
        print("=" * 50)

        # ADF Test
        adf_result = adfuller(series.dropna())
        print(f"ADF Test:")
        print(f"  Test Statistic: {adf_result[0]:.4f}")
        print(f"  p-value: {adf_result[1]:.4f}")
        adf_stationary = adf_result[1] < alpha

        # KPSS Test
        try:
            kpss_result = kpss(series.dropna(), regression='c')
            print(f"KPSS Test:")
            print(f"  Test Statistic: {kpss_result[0]:.4f}")
            print(f"  p-value: {kpss_result[1]:.4f}")
            kpss_stationary = kpss_result[1] > alpha
        except:
            print("KPSS Test: Could not be computed")
            kpss_stationary = False

        print(f"Conclusion:")
        print(f"  ADF suggests {'stationary' if adf_stationary else 'non-stationary'}")
        print(f"  KPSS suggests {'stationary' if kpss_stationary else 'non-stationary'}")

        return adf_stationary and kpss_stationary

    def apply_differencing(self):
        """Apply first-order differencing to make data stationary"""
        print("\n" + "=" * 50)
        print("APPLYING FIRST-ORDER DIFFERENCING")
        print("=" * 50)

        # Check stationarity for each variable
        needs_differencing = {}
        for col in self.data.columns:
            is_stationary = self.check_stationarity(self.data[col], variable_name=col)
            needs_differencing[col] = not is_stationary

        # Apply differencing only to non-stationary variables
        self.stationary_data = self.data.copy()
        for col, needs_diff in needs_differencing.items():
            if needs_diff:
                print(f"Applying differencing to {col}")
                self.stationary_data[col] = self.stationary_data[col].diff()
                self.differencing_applied = True

        # Drop NA values created by differencing
        self.stationary_data = self.stationary_data.dropna()  # Fixed the typo here

        return self.stationary_data

    def find_optimal_lag(self, max_lags=6):
        """Find the optimal lag order for the VAR model"""
        print("\n" + "=" * 50)
        print("FINDING OPTIMAL LAG ORDER")
        print("=" * 50)

        try:
            model = VAR(self.stationary_data)
            results = model.select_order(max_lags)
            print(results.summary())

            # Get the optimal lag based on AIC
            optimal_lag = results.aic
            print(f"Optimal lag order based on AIC: {optimal_lag}")

            return optimal_lag
        except Exception as e:
            print(f"Error in finding optimal lag: {e}")
            print("Using default lag order of 1")
            return 1

    def fit_model(self, lag_order=None):
        """Fit the VAR model with specified or optimal lag order"""
        if lag_order is None:
            lag_order = self.find_optimal_lag()

        try:
            self.model = VAR(self.stationary_data)
            self.results = self.model.fit(lag_order)
            print(f"VAR({lag_order}) model fitted successfully")
            return self.results
        except Exception as e:
            print(f"Error fitting VAR model: {e}")
            print("Trying with lag order 1")
            self.model = VAR(self.stationary_data)
            self.results = self.model.fit(1)
            return self.results

    def generate_forecast(self, steps=6):
        """Generate forecast for the specified number of steps"""
        if self.results is None:
            raise ValueError("Model must be fitted before forecasting")

        try:
            # Generate forecast
            lag_order = self.results.k_ar
            forecast_input = self.stationary_data.values[-lag_order:]
            forecast_values = self.results.forecast(forecast_input, steps=steps)

            # Create forecast dates as the first of each month (July to December)
            last_date = self.stationary_data.index[-1]
            self.forecast_dates = pd.date_range(
                start=last_date + pd.DateOffset(months=1),
                periods=steps,
                freq='MS'  # Month Start frequency
            )

            # Convert to DataFrame
            self.forecast = pd.DataFrame(
                forecast_values,
                index=self.forecast_dates,
                columns=self.stationary_data.columns
            )

            # Reverse differencing if applied
            if self.differencing_applied:
                # For differenced data, we need to integrate the forecast
                last_values = self.original_data.loc[self.stationary_data.index[-1]]
                for i in range(len(self.forecast)):
                    if i == 0:
                        self.forecast.iloc[i] = self.forecast.iloc[i] + last_values
                    else:
                        self.forecast.iloc[i] = self.forecast.iloc[i] + self.forecast.iloc[i - 1]

            print(f"Forecast generated for dates: {[d.strftime('%Y-%m') for d in self.forecast_dates]}")
            return self.forecast
        except Exception as e:
            print(f"Error generating forecast: {e}")
            return None

    def calculate_yoy_inflation(self):
        """Calculate Year-over-Year inflation for forecasted months"""
        if self.forecast is None:
            raise ValueError("Forecast must be generated before calculating YoY inflation")

        try:
            # Get CPI data
            cpi_data = self.original_data['CPIAUCSL']

            # Calculate YoY% for each forecasted month
            yoy_values = []
            for i, date in enumerate(self.forecast_dates):
                # Get the value from 12 months prior
                base_date = date - pd.DateOffset(months=12)

                if base_date in cpi_data.index:
                    base_value = cpi_data.loc[base_date]
                else:
                    # If base date not in data, use the closest available date
                    base_value = cpi_data.iloc[-12 + i] if i < 12 else cpi_data.iloc[-1]

                forecast_value = self.forecast['CPIAUCSL'].iloc[i]
                yoy_value = (forecast_value / base_value - 1) * 100
                yoy_values.append(yoy_value)

            self.yoy_forecast = pd.Series(yoy_values, index=self.forecast_dates)
            return self.yoy_forecast
        except Exception as e:
            print(f"Error calculating YoY inflation: {e}")
            return None

    def run_monte_carlo_simulations(self, n_simulations=10000):
        """Run Monte Carlo simulations for future paths"""
        if self.results is None:
            raise ValueError("Model must be fitted before running simulations")

        try:
            sigma = self.results.sigma_u
            coefficients = self.results.coefs
            n_variables = len(self.stationary_data.columns)
            n_steps = len(self.forecast_dates)

            # Initialize simulation array
            self.simulations = np.zeros((n_simulations, n_steps, n_variables))
            last_obs = self.stationary_data.values[-self.results.k_ar:]

            # Run simulations
            for i in range(n_simulations):
                # Initialize with the last observations
                current = last_obs.copy()
                simulated_path = np.zeros((n_steps, n_variables))

                for j in range(n_steps):
                    # Generate innovation
                    innovation = np.random.multivariate_normal(np.zeros(n_variables), sigma)

                    # Calculate next step using all lags
                    next_step = np.zeros(n_variables)
                    for lag in range(self.results.k_ar):
                        next_step += np.dot(coefficients[lag], current[-lag - 1])
                    next_step += innovation

                    simulated_path[j] = next_step
                    current = np.vstack([current, next_step])

                self.simulations[i] = simulated_path

            print(f"Ran {n_simulations} Monte Carlo simulations")
            return self.simulations
        except Exception as e:
            print(f"Error running Monte Carlo simulations: {e}")
            return None

    def calculate_yoy_simulations(self):
        """Calculate YoY inflation for simulation paths"""
        if self.simulations is None:
            raise ValueError("Simulations must be run before calculating YoY")

        try:
            # Get CPI data
            cpi_data = self.original_data['CPIAUCSL']
            cpi_idx = self.stationary_data.columns.get_loc('CPIAUCSL')
            n_simulations, n_steps, _ = self.simulations.shape

            # Initialize YoY simulation array
            self.yoy_simulations = np.zeros((n_simulations, n_steps))

            # Calculate YoY for each simulation path
            for i in range(n_simulations):
                # Reconstruct the full path from differenced simulations if needed
                simulated_path = self.simulations[i].copy()

                if self.differencing_applied:
                    # Integrate the differenced simulations
                    last_value = self.original_data['CPIAUCSL'].loc[self.stationary_data.index[-1]]
                    integrated_path = np.zeros(n_steps)
                    for j in range(n_steps):
                        if j == 0:
                            integrated_path[j] = simulated_path[j, cpi_idx] + last_value
                        else:
                            integrated_path[j] = simulated_path[j, cpi_idx] + integrated_path[j - 1]
                else:
                    integrated_path = simulated_path[:, cpi_idx]

                # Calculate YoY for each point in the simulated path
                for j in range(n_steps):
                    forecast_date = self.forecast_dates[j]
                    base_date = forecast_date - pd.DateOffset(months=12)

                    if base_date in cpi_data.index:
                        base_value = cpi_data.loc[base_date]
                    else:
                        # If base date not in data, use the closest available date
                        base_value = cpi_data.iloc[-12 + j] if j < 12 else cpi_data.iloc[-1]

                    yoy_value = (integrated_path[j] / base_value - 1) * 100
                    self.yoy_simulations[i, j] = yoy_value

            return self.yoy_simulations
        except Exception as e:
            print(f"Error calculating YoY simulations: {e}")
            return None

    def calculate_probability_and_price(self):
        """Calculate probability and derivative price"""
        if self.yoy_simulations is None:
            raise ValueError("YoY simulations must be calculated first")

        try:
            # Calculate probability of exceeding 4% in any month
            exceeds_threshold = np.any(self.yoy_simulations > 4, axis=1)
            probability = np.mean(exceeds_threshold)

            # Calculate fair price
            fair_price = 100 * np.exp(-0.02145) * probability

            # Calculate confidence intervals for probability
            se = np.sqrt(probability * (1 - probability) / self.yoy_simulations.shape[0])
            prob_65_low = max(0, probability - norm.ppf(0.825) * se)
            prob_65_high = min(1, probability + norm.ppf(0.825) * se)
            prob_95_low = max(0, probability - norm.ppf(0.975) * se)
            prob_95_high = min(1, probability + norm.ppf(0.975) * se)

            # Calculate fair price ranges
            fair_price_65_low = 100 * np.exp(-0.02145) * prob_65_low
            fair_price_65_high = 100 * np.exp(-0.02145) * prob_65_high
            fair_price_95_low = 100 * np.exp(-0.02145) * prob_95_low
            fair_price_95_high = 100 * np.exp(-0.02145) * prob_95_high

            return {
                'probability': probability,
                'fair_price': fair_price,
                'prob_65_range': (prob_65_low, prob_65_high),
                'prob_95_range': (prob_95_low, prob_95_high),
                'fair_price_65_range': (fair_price_65_low, fair_price_65_high),
                'fair_price_95_range': (fair_price_95_low, fair_price_95_high)
            }
        except Exception as e:
            print(f"Error calculating probability and price: {e}")
            return {
                'probability': 0,
                'fair_price': 0,
                'prob_65_range': (0, 0),
                'prob_95_range': (0, 0),
                'fair_price_65_range': (0, 0),
                'fair_price_95_range': (0, 0)
            }

    def plot_confidence_intervals(self):
        """Plot forecast with confidence intervals"""
        if self.yoy_forecast is None or self.yoy_simulations is None:
            raise ValueError("Forecast and simulations must be calculated first")

        try:
            # Calculate confidence intervals
            forecast_mean = self.yoy_forecast.values
            forecast_std = np.std(self.yoy_simulations, axis=0)

            ci_65_low = forecast_mean - norm.ppf(0.825) * forecast_std
            ci_65_high = forecast_mean + norm.ppf(0.825) * forecast_std
            ci_95_low = forecast_mean - norm.ppf(0.975) * forecast_std
            ci_95_high = forecast_mean + norm.ppf(0.975) * forecast_std

            # Create figure
            plt.figure(figsize=(12, 6))

            # Plot forecast with confidence intervals
            plt.plot(self.forecast_dates, forecast_mean, 'b-', linewidth=2, label='Expected YoY Inflation')
            plt.fill_between(self.forecast_dates, ci_65_low, ci_65_high, alpha=0.5,
                             color='orange', label='65% Confidence Interval')
            plt.fill_between(self.forecast_dates, ci_95_low, ci_95_high, alpha=0.3,
                             color='orange', label='95% Confidence Interval')
            plt.axhline(y=4, color='r', linestyle='--', label='4% Threshold')

            plt.title('6-Month YoY Inflation Forecast with Confidence Intervals')
            plt.ylabel('YoY% Inflation')
            plt.legend()
            plt.grid(True)

            # Format x-axis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting confidence intervals: {e}")

    def plot_monte_carlo_paths(self, n_sample_paths=100):
        """Plot sample Monte Carlo simulation paths"""
        if self.yoy_simulations is None:
            raise ValueError("Simulations must be calculated first")

        try:
            # Create figure
            plt.figure(figsize=(12, 6))

            # Plot sample Monte Carlo paths
            sample_paths = self.yoy_simulations[:n_sample_paths]
            for i in range(n_sample_paths):
                plt.plot(self.forecast_dates, sample_paths[i], alpha=0.1, color='blue')

            # Add mean and threshold
            plt.plot(self.forecast_dates, self.yoy_forecast.values, 'b-', linewidth=2, label='Mean Forecast')
            plt.axhline(y=4, color='r', linestyle='--', label='4% Threshold')

            plt.title(f'{n_sample_paths} Sample Monte Carlo Simulation Paths')
            plt.xlabel('Date')
            plt.ylabel('YoY% Inflation')
            plt.legend()
            plt.grid(True)

            # Format x-axis
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting Monte Carlo paths: {e}")

    def run_analysis(self):
        """Run complete analysis with simplified preprocessing"""
        print("Starting analysis with first-order differencing...")

        # Step 1: Apply differencing if needed
        self.apply_differencing()

        # Step 2: Fit VAR model with optimal lag (max 6)
        optimal_lag = self.find_optimal_lag(max_lags=6)
        print(f"Fitting VAR({optimal_lag}) model...")
        self.fit_model(lag_order=optimal_lag)

        print("Generating 6-month forecast...")
        self.generate_forecast(steps=6)

        print("Calculating YoY inflation...")
        self.calculate_yoy_inflation()

        print("Running Monte Carlo simulations...")
        self.run_monte_carlo_simulations(n_simulations=10000)

        print("Calculating YoY for simulations...")
        self.calculate_yoy_simulations()

        print("Calculating probability and derivative price...")
        results = self.calculate_probability_and_price()

        print("Plotting confidence intervals...")
        self.plot_confidence_intervals()

        print("Plotting Monte Carlo paths...")
        self.plot_monte_carlo_paths(n_sample_paths=1000)

        return results


def main():
    # Initialize the pricer
    data_path = "cleaned_data.csv"
    pricer = InflationDerivativePricer(data_path)

    # Run the complete analysis
    print("=" * 60)
    print("ANALYSIS WITH FIRST-ORDER DIFFERENCING")
    print("=" * 60)
    results = pricer.run_analysis()

    # Print results
    print("=" * 60)
    print("FORECAST AND DERIVATIVE PRICING RESULTS")
    print("=" * 60)
    print(f"Probability of YoY inflation exceeding 4% in any month: {results['probability']:.4f}")
    print(f"Fair price of derivative: {results['fair_price']:.4f}")
    print("\nConfidence Intervals for Probability:")
    print(f"65% Range: [{results['prob_65_range'][0]:.4f}, {results['prob_65_range'][1]:.4f}]")
    print(f"95% Range: [{results['prob_95_range'][0]:.4f}, {results['prob_95_range'][1]:.4f}]")
    print("\nConfidence Intervals for Fair Price:")
    print(f"65% Range: [{results['fair_price_65_range'][0]:.4f}, {results['fair_price_65_range'][1]:.4f}]")
    print(f"95% Range: [{results['fair_price_95_range'][0]:.4f}, {results['fair_price_95_range'][1]:.4f}]")

    print("\nMonthly YoY Inflation Forecast Values:")
    for i, date in enumerate(pricer.forecast_dates):
        print(f"{date.strftime('%B %Y')}: {pricer.yoy_forecast.iloc[i]:.4f}%")


if __name__ == "__main__":
    main()