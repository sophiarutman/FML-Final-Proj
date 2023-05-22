import keras
from keras.models import load_model

from dqn import Agent
from dqn_functions import *
import altered_backtest as ab
import matplotlib.pyplot as plt

symbol = "CMCSA"
window_size = 40
start, end = "2021-01-01", "2022-01-25"

model = load_model("model_name")

agent = Agent(True)

data = getStockDataDF(symbol, start, end, window_size)
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
	date = str(row[0].date())
	action = agent.act(state)

	# sit
	next_state = getState(data, symbol, t + 1, max_price, max_macd, max_rsi, max_lobb)
	next_state = np.append(next_state, action)
	next_state = np.expand_dims(next_state, axis=0)  # Add an extra dimension at axis=0

	reward = 0
	price =  data[symbol].values[t]

	if action == 1 and pa != 1: #long
		if pa == 0:
			total_profit -= (price)
			trades.loc[date] = 1000
			print("Buy at: " + formatPrice(price))
		elif pa == 2:
			# Short to Flat
			total_profit -= (price)
			# Flat to Long
			total_profit -= (price)
			trades.loc[date] = 1000
			print("Short to Long at: " + formatPrice(price))

	elif action == 2 and pa != 2: # short
		if pa == 0:
			total_profit += (price)
			print("Short at: " + formatPrice(price))
			trades.loc[date] = -1000
		elif pa == 1:
			# Long to Flat
			total_profit += (price)
			# Flat to Short
			total_profit += (price)
			trades.loc[date] = -1000
			print("Long to Short at: " + formatPrice(price))
		else:
			trades.loc[date] = 0
	
	elif action == 0 and pa != 0: #flat
		if pa == 1: #Long to flat
			# Long to Flat
			total_profit += (price)
			trades.loc[date] = 0
			print("Long to Flat at: " + formatPrice(price))

		elif pa == 2:
			# Short to Flat
			total_profit -= (price)
			trades.loc[date] = 0
			print("Short to Flat at: " + formatPrice(price))
		else:
			trades.loc[date] = 0
	else:
		trades.loc[date] = 0

	pa = action

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	t+=1

	if done:
		print("--------------------------------")
		print(symbol + " Total Profit: " + formatPrice(total_profit))
		print("--------------------------------")
		break
print(trades.to_string())
baseline = pd.DataFrame(0, index=trades.index, columns=["Trades"])
baseline["Trades"] = 0
baseline["Trades"][0] = 1000

dqn = (ab.assess_strategy(trades, symbol, starting_value = 200000,
        fixed_cost = 0.0, floating_cost = 0.000, start=start, end=end) / 200000) -1
print(dqn.to_string())

base = (ab.assess_strategy(baseline, symbol, starting_value = 200000,
        fixed_cost = 0.0, floating_cost = 0.000, start=start, end=end) / 200000) - 1
print(base.to_string())
plt.figure(1)
plt.title('Deep Q Strategy vs. Baseline for CMCSA (40) Model with 1 episode')
plt.xlabel("Date")
plt.ylabel('Cumlative Return')
plt.plot(dqn)
plt.plot(base)
plt.legend(['DQN', 'Baseline'])
plt.show()