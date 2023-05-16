import keras
from keras.models import load_model

from dqn import Agent
from dqn_functions import *
import altered_backtest as ab
import matplotlib.pyplot as plt

symbol = "GOOGL"
window_size = 45
start, end = "2020-01-01", "2022-12-31"

model = load_model("/Users/jsoeder/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/FML/FML-Final-Proj/models/model_ep19")

agent = Agent(True, symbol)

data = getStockDataDF(symbol, start, end, window_size)
print(data.to_string())
l = len(data[symbol]) - 1

max_price = max(data[symbol])
max_rsi =  max(data["RSI"])
max_macd = max(data["MACD"])
max_lobb = max(data["Lobbying"])

state = getState(data, symbol, 0, max_price, max_macd, max_rsi, max_lobb)
state = np.append(state, 0)
state = np.expand_dims(state, axis=0)  # Add an extra dimension at axis=0
total_profit, pa = 0, 0

trades = pd.DataFrame(index=data.index)
trades["Trades"] = 0

t = 0
for row in data.iterrows():
	date = row[0].date()
	action = agent.act(state)

	# sit
	next_state = getState(data, symbol, t + 1, max_price, max_macd, max_rsi, max_lobb)
	next_state = np.append(next_state, action)
	next_state = np.expand_dims(next_state, axis=0)  # Add an extra dimension at axis=0

	reward = 0
	price =  data[symbol].values[t]
	date = data.index

	if action == 1 and pa != 1: #long
		if pa == 0:
			total_profit -= (price)
			trades["Trades"][date] = 1000
			print("Buy at: " + formatPrice(price))
		elif pa == 2:
			# Short to Flat
			total_profit -= (price)
			# Flat to Long
			total_profit -= (price)
			trades["Trades"][date] = 1000
			print("Short to Long at: " + formatPrice(price))

	elif action == 2 and pa != 2: # short
		if pa == 0:
			total_profit += (price)
			print("Short at: " + formatPrice(price))
			trades["Trades"][date] = -1000
		elif pa == 1:
			# Long to Flat
			total_profit += (price)
			# Flat to Short
			total_profit += (price)
			trades["Trades"][date] = -1000
			print("Long to Short at: " + formatPrice(price))
		else:
			trades["Trades"][date] = 0
	
	elif action == 0 and pa != 0: #flat
		if pa == 1: #Long to flat
			# Long to Flat
			total_profit += (price)
			trades["Trades"][date] = 0
			print("Long to Flat at: " + formatPrice(price))

		elif pa == 2:
			# Short to Flat
			total_profit -= (price)
			trades["Trades"][date] = 0
			print("Short to Flat at: " + formatPrice(price))
		else:
			trades["Trades"][date] = 0
	else:
		trades["Trades"][date] = 0

	pa = action

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	t+=1

	if done:
		print("--------------------------------")
		print(symbol + " Total Profit: " + formatPrice(total_profit))
		print("--------------------------------")

baseline = pd.DataFrame(0, index=trades.index, columns=["Trades"])
baseline["Trade"] = 0
baseline["Trade"][0] = 1000

dqn = ab.assess_strategy(trades, symbol, starting_value = 1000000,
        fixed_cost = 0.0, floating_cost = 0.000, start=start, end=end)

base = (ab.assess_strategy(baseline, starting_value = 200000, symbol="DIS") / 200000 ) - 1

plt.figure(1)
plt.title('Deep Q Strategy vs. Baseline')
plt.xlabel("Date")
plt.ylabel('Cumlative Return')
plt.plot(dqn)
plt.plot(base)
plt.legend(['DQN', 'Baseline'])
plt.show()