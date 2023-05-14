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

# returns the sigmoid
def sigmoid(x):
	return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t
def getState(data, t):
	row = data[t:t+1]
	res = []
	for ind in row:
		val = row[ind]
		
		res.append(sigmoid(val[0]))

	return np.array([res])