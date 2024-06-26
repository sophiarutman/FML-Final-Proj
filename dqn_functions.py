import IndicatorRetrieval as ir
import tech_ind
import numpy as np
import pandas as pd
import numpy as np
import math

# prints formatted price
def formatPrice(n):
	return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

# returns the vector containing stock data from a fixed file
def getStockDataDF(symbol, start_date, end_date, lobbyingWindow):
	dates = pd.date_range(start_date, end_date)
	df = pd.DataFrame(index=dates)
	df_symbol = pd.read_csv(f'./data/' + symbol + '.csv', index_col=['Date'], parse_dates=True, na_values=['nan'], usecols=['Date',"Adj Close"])
	df = df.join(df_symbol, how="inner")
	df = df.rename(columns={"Adj Close": symbol})
	df = tech_ind.MACDIndicator(df, symbol)
	df = tech_ind.RSIIndicator(df, symbol)
	indicator_data = ir.get_data(start_date, end_date, symbol)
	df = tech_ind.LobbyingIndicator(df, indicator_data, symbol, lobbyingWindow)
	df = df[[symbol, "MACD", "RSI", "Lobbying"]]

	df = df.ffill().bfill()

	return df

# returns an an n-day state representation ending at time t
def getState(data, symbol, t,  max_price, max_macd, max_rsi, max_lobb):
	row = data[t:t+1]
	res = []

	price = row[symbol][0]
	res.append(price / max_price)

	macd = row["MACD"][0]
	res.append(macd / max_macd)	

	rsi = row["RSI"][0]
	res.append(rsi / max_rsi)	

	lobbying = row["Lobbying"][0]
	res.append(lobbying / max_lobb)	

	return np.array([res])