from dqn import Agent
from dqn_functions import *
import sys



symbol = "AMZN"
window_size = 30
episode_count = 4
start, end = "2018-01-01", "2020-12-31"

agent = Agent()

data = getStockDataDF(symbol, start, end, window_size)

max_price = max(data[symbol])
max_rsi =  max(data["RSI"])
max_macd = max(data["MACD"])
max_lobb = max(data["Lobbying"])

l = len(data[symbol]) 
batch_size = 16
total_profit = 0

for e in range(episode_count):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, symbol, 0, max_price, max_macd, max_rsi, max_lobb)
	state = np.append(state, 0)
	state = np.expand_dims(state, axis=0)  # Add an extra dimension at axis=0
	pa = 0
	last_price = data[symbol].values[0]

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)
		

		next_state = getState(data, symbol, t + 1, max_price, max_macd, max_rsi, max_lobb)
		next_state = np.append(next_state, action)
		next_state = np.expand_dims(next_state, axis=0)  # Add an extra dimension at axis=0


		reward = 0
		price =  data[symbol].values[t]
		holdings = 0

		if action == 1 and pa != 1: #long
			if pa == 0:
				reward = 0
				last_price = price
				total_profit -= (price * 1000)
				print("Buy at: " + formatPrice(price))
			elif pa == 2:
				reward = max(last_price - price, 0)
				# Short to Flat
				total_profit -= (price * 1000)
				# Flat to Long
				total_profit -= (last_price * 1000)
				last_price = price
				print("Short to Long at: " + formatPrice(price))

		elif action == 2 and pa != 2: # short
			if pa == 0:
				reward = 0
				last_price = price
				total_profit += (price * 1000)
				print("Short at: " + formatPrice(price))
			elif pa == 1:
				reward = max(price - last_price, 0)
				last_price = price
				# Long to Flat
				total_profit += (price * 1000)
				# Flat to Short
				total_profit += (price * 1000)
				print("Long to Short at: " + formatPrice(price))
		
		elif action == 0 and pa != 0: #flat
			if pa == 1: #Long to flat
				reward = max(price - last_price, 0)
				# Long to Flat
				total_profit += (price * 1000)
				print("Long to Flat at: " + formatPrice(price))

			elif pa == 2: #Short to flat
				reward = max(last_price - price, 0)
				# Short to Flat
				total_profit -= (price * 1000)
				print("Short to Flat at: " + formatPrice(price))

		print("Profit: " + formatPrice(total_profit))
		pa = action

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:

			if (action == 1):
				total_profit += (price * 1000)
			elif (action == 2):
				total_profit -= (price * 1000)

			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")



		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))

print("--------------------------------")
print("Total Profit: " + formatPrice(total_profit))
print("--------------------------------")