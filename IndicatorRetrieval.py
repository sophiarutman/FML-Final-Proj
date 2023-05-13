import pandas as pd
import numpy as np

def get_data( start, end, symbol, data_file="FML-Final-Proj/lobbying.csv"):
    # Construct an empty DataFrame with the requested date range.
    dates = pd.date_range(start, end)
    df = pd.DataFrame(index=dates)

    # Read SPY.
    df_spy = pd.read_csv('./data/SPY.csv', index_col=['Date'], parse_dates=True, na_values=['nan'], usecols=['Date',"Adj Close"])

    # Use SPY to eliminate non-market days.
    df = df.join(df_spy, how='inner')
    df = df.rename(columns={"Adj Close":'SPY'})

    # Append the data for the symbol, retaining all market-open days.
    df_sym = pd.read_csv(data_file, index_col=['Date'], parse_dates=True, na_values=['nan'], usecols=['Date',symbol])
    df = df.join(df_sym, how='left')

    del df['SPY']

    return df


#c = IndicatorRetrieval()
#c.main()