import pandas as pd
import tech_ind
import IndicatorRetrieval as ir
import matplotlib.pyplot as plt


def prepare_world (start_date, end_date, symbol, data_folder, lobbyingWindow):
    """
    Read the relevant price data and calculate some indicators.
    Return a DataFrame containing everything you need.
    """
    dates = pd.date_range(start_date, end_date)
    df = pd.DataFrame(index=dates)
    df_symbol = pd.read_csv(f'{data_folder}/' + symbol + '.csv', index_col=['Date'], parse_dates=True, na_values=['nan'], usecols=['Date',"Adj Close"])
    df = df.join(df_symbol, how="inner")
    df = df.rename(columns={"Adj Close": symbol})
    df = tech_ind.MACDIndicator(df, symbol)
    df = tech_ind.RSIIndicator(df, symbol)

    indicator_data = ir.get_data(start_date, end_date, symbol)
    df = tech_ind.LobbyingIndicator(df, indicator_data, symbol, lobbyingWindow)

    df = df[[symbol, "MACD", "RSI", "Lobbying"]]
    
        
    df = df.ffill().bfill()

    plt.figure(1)
    plt.title(symbol + " MACD Indicator")
    plt.plot(df["MACD"])
    plt.xlabel("Date")
    plt.ylabel("MACD")
    plt.show()

    plt.figure(1)
    plt.title(symbol + " RSI Indicator")
    plt.plot(df['RSI'])
    plt.xlabel("Date")
    plt.ylabel("RSI")
    plt.show()

    plt.figure(1)
    plt.title(symbol + " Lobbying Values")
    plt.plot(df["Lobbying"])
    plt.xlabel("Date")
    plt.ylabel("$")
    plt.show()

    plt.figure(1)
    plt.title(symbol + " Cost")
    plt.plot(df[symbol])
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.show()
        
    return df

prepare_world("2018-01-01", "2022-12-31", "AMZN", "./data", 30)