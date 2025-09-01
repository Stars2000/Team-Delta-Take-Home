# Team Delta Take Home Assignment
## Overview
This repository contains the analysis and implementation for an 
econometrics assignment focused on financial data analysis and 
derivative pricing. The project is structured into two main tasks:

1. Bitcoin and S&P 500 Returns Analysis: Calculation of monthly returns,
risk-return metrics, and asset preference justification.
2. Inflation Derivative Pricing: Modeling and valuation of a derivative 
contract based on U.S. inflation data, and additional data.

## Task 1: Bitcoin vs. S&P 500
In the Task_1 directory the various scripts can be found sorted by section from A to C.
The results for the monthly returns and the png files of the monte carlo simulations 
can be found in the Results directory. Below is a summary of the other results.

#### Section B:
Bitcoin Monthly Returns:
1. Mean: 10.89%
2. Std Dev: 44.43%

S&P 500 Monthly Returns:
1. Mean: 1.45%
2. Std Dev: 3.87%

#### Section C:
Additional stats:
Bitcoin Monthly Returns:
1. Sharpe Ratio: 1.57

S&P 500 Monthly Returns:
1. Sharpe Ratio: 1.08

## Task 2: Inflation Derivative Pricing
A full report wil be emailed to the team Delta email before the interview. 
Refer to the report for a full explaination of the proces.

#### Attempt 1: ARIMA
1. Handles trend through differencing (d=1)
2. Captures short-term dynamics through autoregressive 
(p) and moving average (q) components

#### Attempt 2: VAR(5)
1. Vector Autoregression with 5 lags
2. Incorporates additional economic indicators from Extra_data/

##### Implementation
1. Data cleaning (Data_cleaning/clean_data.py)
2. Var forecasting (var_forecaster.py)

##### Key Findings (forecasting last six months of 2025)
1. Estimated Probability of payout: 40.88%
2. Estimated Fair Price: 40.01

## Usage
1. Install required Python Libraries:
"pip install -r requirements.txt"
2. Run Scripts