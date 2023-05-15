import keras
from keras.models import load_model

from dqn import Agent
from dqn_functions import *
import sys

symbol = "AMZN"
window_size = 20
episode_count = 5
start, end = "2020-01-01", "2022-12-31"
model = load_model("models/" + symbol)

agent = Agent(True, symbol)
data = getStockDataDF(symbol, start, end, window_size)
l = len(data[symbol]) - 1
batch_size = 16

max_price = max(data[symbol])
max_rsi =  max(data["RSI"])
max_macd = max(data["MACD"])
max_lobb = max(data["Lobbying"])

state = getState(data, symbol, 0, max_price, max_macd, max_rsi, max_lobb)
state = np.append(state, 0)
state = np.expand_dims(state, axis=0)  # Add an extra dimension at axis=0
total_profit = 0
agent.inventory = []

for t in range(l):
	action = agent.act(state)

	# sit
	next_state = getState(data, symbol, t + 1, max_price, max_macd, max_rsi, max_lobb)
	state = np.append(state, 0)
	state = np.expand_dims(state, axis=0)  # Add an extra dimension at axis=0

	reward = 0
	price =  data[symbol].values[t]

	if action == 1: # buy
		agent.inventory.append(price)
		print("Buy: " + formatPrice(price))

	elif action == 2 and len(agent.inventory) > 0: # sell
		bought_price = agent.inventory.pop(0)
		reward = max(price - bought_price, 0)
		total_profit += price - bought_price
		print("Sell: " + formatPrice(price) + " | Profit: " + formatPrice(price - bought_price))

	done = True if t == l - 1 else False
	agent.memory.append((state, action, reward, next_state, done))
	state = next_state

	if done:
		print("--------------------------------")
		print(symbol + " Total Profit: " + formatPrice(total_profit))
		print("--------------------------------")