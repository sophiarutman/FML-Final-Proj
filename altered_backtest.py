import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def get_data(start, end, symbol, column_name="Adj Close", include_spy=False, 
        data_folder="./data"):
    # Construct an empty DataFrame with the requested date range.
    dates = pd.date_range(start, end)
    df = pd.DataFrame(index=dates)

    # Read SPY.
    df_spx = pd.read_csv(f'{data_folder}/^SPX.csv', index_col=['Date'], 
            parse_dates=True, na_values=['nan'], usecols=['Date',column_name])

    # Use SPY to eliminate non-market days.
    df = df.join(df_spx, how='inner')
    df = df.rename(columns={column_name:'^SPX'})

    # Append the data for the remaining symbols, retaining all market-open days.
    df_sym = pd.read_csv(f'{data_folder}/{symbol}.csv', index_col=['Date'], 
            parse_dates=True, na_values=['nan'], usecols=['Date',column_name])
    df = df.join(df_sym, how='left')
    df = df.rename(columns={column_name:symbol})

    # Eliminate SPY if requested.
    if not include_spy: del df['^SPX']

    return df

""" Implement the strategy layed out by the given trade file. Iterates 
over the day when a trade is requested and tracks the shares held of 
each company on every day a move was made. The share holding at the date
of the previous trade is used as a base for the holding calculation at
the subsequent trade. This holding dataframe is forward filled to fill in the
share holdings at the respective days. Then this complete data frame is 
multiplied by a dataframe with the stock values for everyday the market was 
traded. This dataframe is summed across the rows resulting in a column dataframe
showing the portfolio value at each day."""

def assess_strategy(trade_df : pd.DataFrame, symbol, starting_value = 1000000,
        fixed_cost = 9.95, floating_cost = 0.005, start='2018-01-01', end='2019-12-31'):

    #extract data from the trade file
    trades = trade_df
    #create a dataframe to store share holdings
    dates = pd.date_range(start, end)
    holdings = pd.DataFrame(index=dates)
        
    #initialize each symbol column to nans
    holdings[symbol] = np.nan
    #set the first value
    holdings.loc[start][symbol] = 0
    
    #Get stock data for the companies traded
    stock_data = get_data(start, end, symbol, column_name="Adj Close", 
            include_spy=False, data_folder="./data")
    #set up dataframes to deal with cash component
    stock_data['CASH'] = 1
    holdings['CASH'] = np.nan
    holdings.loc[start]['CASH'] = starting_value

    #initialize prev data to the initial values
    prev_date = holdings.loc[start]
    
    #iterate over each trade
    for row in trades.iterrows():
        #extract features of the trade
        date = row[0]
        shares = row[1]["Trades"]
        if shares > 0:
            direction = "BUY"
        else: 
            direction = "SELL"
        order_value = stock_data.loc[date][symbol] * shares

        #grab previous holding data
        holdings.loc[date] = prev_date

        #perform trade
        if  direction == "BUY":            
            holdings.loc[date][symbol] += shares
            holdings.loc[date]['CASH'] -= abs(order_value) 
        else:
            holdings.loc[date][symbol] += shares
            holdings.loc[date]['CASH'] += abs(order_value)

        #charge trading fees
        holdings.loc[date]['CASH'] -= (order_value * floating_cost)
        holdings.loc[date]['CASH'] -= (fixed_cost)

        #update previous date, note this will allow multiple trades per day
        prev_date = holdings.loc[date]

    #fill in days trades weren't made
    all_portfolio_values = (holdings.ffill().bfill() * stock_data).sum(axis=1)
    #remove days the market wasn't traded
    daily_portf_values = all_portfolio_values.where(all_portfolio_values != 0).dropna()

    # ----------------------
    #Uncomment to show stats
    # ----------------------
    #generate_stats(start, end, daily_portf_values.copy(), symbol)

    return daily_portf_values

"""Calculates and prints portfolio statistics for the portfolio
   that was created by the given trade strategy. """
def generate_stats(start_date, end_date, portfolio_values, symbol, starting_value=1000000, 
        risk_free_rate=0.0, sample_freq=252, plot_returns=False):
    
    #Extract SPX data
    df = get_data(start_date, end_date, symbol, "Adj Close", False)
    df = df.ffill().bfill()

    #Calculate Cumlative Return of a portfolio that held ^SPX over this time
    cr_per_stock = df.div(df.iloc[0])
    adjusted_cr = (cr_per_stock.multiply(starting_value)).sum(axis=1)

    #Calculate Daily Return of a portfolio that held ^SPX over this time
    dr = adjusted_cr/(adjusted_cr.shift()) - 1
    benchmark_cr = cr_per_stock.iloc[-1].sum() - 1
    benchmark_adr = dr.mean()
    benchmark_sd = dr.std()
    #Calculate Sharpe Ratio of a portfolio that held ^SPX over this time
    benchmark_sharpe_ratio = math.sqrt(sample_freq) * (benchmark_adr - risk_free_rate)/benchmark_sd

    #Calculate Cumlative Return of the implemented portfolio over this time
    cr_portfolio = portfolio_values.div(portfolio_values.iloc[0])
    #Calculate Daily Return of the implemented portfolio over this time
    dr = portfolio_values/(portfolio_values.shift()) - 1

    portfolio_cr = cr_portfolio.iloc[-1].sum() - 1
    portfolio_adr = dr.mean()
    portfolio_sd = dr.std()
    #Calculate Sharpe Ratio of the implemented portfolio over this time
    
    #Print portfolio and benchmark statistics
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}\n")

    print(f"Portfolio CR: {portfolio_cr}")
    print(f"Portfolio ADR: {portfolio_adr}")   
    print(f"Portfolio SD: {portfolio_sd}\n")

    #print(f"Final Portfolio Value: {final_portfolio_value}")

    return


