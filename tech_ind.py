import pandas as pd
import matplotlib.pyplot as plt

def get_data(start, end, symbols, column_name="Adj Close", include_spy=True, data_folder="./data"):

    # Construct an empty DataFrame with the requested date range.
    dates = pd.date_range(start, end)
    df = pd.DataFrame(index=dates)

    # Read SPY.
    df_spy = pd.read_csv(f'{data_folder}/SPY.csv', index_col=['Date'], parse_dates=True, na_values=['nan'], usecols=['Date',column_name])

    # Use SPY to eliminate non-market days.
    df = df.join(df_spy, how='inner')
    df = df.rename(columns={column_name:'SPY'})

    # Append the data for the remaining symbols, retaining all market-open days.
    for sym in symbols:
        df_sym = pd.read_csv(f'{data_folder}/{sym}.csv', index_col=['Date'], parse_dates=True, na_values=['nan'], usecols=['Date',column_name])
        df = df.join(df_sym, how='left')
        df = df.rename(columns={column_name:sym})

    # Eliminate SPY if requested.
    if not include_spy: del df['SPY']

    return df

def MACDIndicator(data, symbol):
    df = data.copy()
    df['EMA8'] = df[symbol].ewm(8).mean()
    df['EMA26'] = df[symbol].ewm(26).mean()
    df['MACD'] = df['EMA8'] - df['EMA26']
    return df


def RSIIndicator(data, symbol):
    df = data.copy()
    df.loc[df[symbol] < df[symbol].shift(-1), "Gain"] = df[symbol].shift(-1) - df[symbol]
    df.loc[df[symbol] > df[symbol].shift(-1), "Loss"] = df[symbol] - df[symbol].shift(-1)
    df["Gain"] = df["Gain"].fillna(value = 0)
    df["AVGGain"] = df["Gain"].rolling(window=15).mean()
    df["Loss"] = df["Loss"].fillna(value = 0)
    df["AVGLoss"] = df["Loss"].rolling(window=15).mean()
    df["RS"] = df["AVGGain"] / df["AVGLoss"]
    df["RSI"] = 100 - (100 / (1 + df["RS"]))

    return df

def lobbyingValues(data, symbol): 

    df = data.copy()
    symbolLobbies = df.loc[symbol]
    symbolLobbies = symbolLobbies.rolling(300).mean()
    symbolLobbies.bfill().ffill()
    #how to bin?

    return symbolLobbies




if __name__ == "__main__":
    
    start = "2018-01-01"
    end = "2019-12-31"
    symbols = ["DIS"]
    df = get_data(start, end, symbols)
    df_MACD = MACDIndicator(df)
    df_RSI = RSIIndicator(df)
    df_EMA = EMAIndicator(df)
    df_BB = BBIndicator(df)


    fig, ax = plt.subplots()
    ax.set_xlabel("Date")
    ax.set_ylabel("MACD Value")
    ax.set_title("Figure 1: MACD Indicator")
    ax.plot(df_MACD['MACD'])
    ax2 = ax.twinx()
    ax2.set_ylabel("DIS Stock Price")
    ax2.plot(df_MACD['DIS'], color='green')
    fig.legend(["MACD", "DIS"])
    plt.show()

    plt.figure(3)
    fig, ax = plt.subplots()
    ax.set_xlabel("Date")
    ax.set_ylabel("RSI")
    ax.set_title("Figure 2: RSI")
    ax.plot(df_RSI['RSI'])
    ax2 = ax.twinx()
    ax2.set_ylabel("DIS Stock Price")
    ax2.plot(df_MACD['DIS'], color='green')
    fig.legend(["RSI", "DIS"])
    plt.show()


    plt.figure(4)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.suptitle("Figure 3A: Bollinger Band Indicator")
    plt.plot(df_BB['SMA15'])
    plt.plot(df_BB['Upper'])
    plt.plot(df_BB['Lower'])
    plt.plot(df_BB['DIS'])
    plt.legend(["SMA-15", "Upper", "Lower", "DIS"])
    plt.show()

    plt.figure(5)
    plt.xlabel("Date")
    plt.ylabel("Percent")
    plt.suptitle("Figure 3B: Bollinger Band Percent")
    plt.plot(df_BB['BB'])
    plt.legend(["BB%"])
    plt.show()
    
    plt.figure(6)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.suptitle("Figure 4: EMA-50 EMA-8 Crossover")
    plt.plot(df_EMA['EMA50'])
    plt.plot(df_EMA['EMA16'])
    plt.plot(df_EMA['DIS'])
    plt.legend(["EMA50", "EMA16", "DIS"])
    plt.show()
    
