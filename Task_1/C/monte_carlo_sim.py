import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_df():
    df = pd.read_csv('../bitcoin_spx.csv')
    df.set_index('Date', inplace=True)

    # Convert all other columns to numeric (just in case)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    bitcoin_data = df["Bitcoin"]
    sp_data = df["S&P 500"]

    return bitcoin_data, sp_data

def simulate(df):
    num_simulations = 10000
    forcast_months = 12

    simulation_array = np.zeros((num_simulations, forcast_months))
    investment_amount = 1000

    monthly_returns = df.pct_change().dropna()

    for i in range(num_simulations):
        cumulative_returns = np.random.choice(monthly_returns, size=forcast_months, replace=True).cumsum()
        simulation_array[i, :] = investment_amount * (1 + cumulative_returns)

    return simulation_array

def plot(simulation_array):
    plt.figure(figsize=(12, 8))
    for i in range(len(simulation_array)):
        plt.plot(simulation_array[i], alpha=0.25)
    plt.title('Monte Carlo Simulation Results')
    plt.xlabel('Months')
    plt.ylabel('Portfolio value')
    plt.show()

df1, df2 = load_df()
simulation_array = simulate(df1)
plot(simulation_array)
simulation_array = simulate(df2)
plot(simulation_array)