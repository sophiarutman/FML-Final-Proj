from dqn import Agent
from dqn_functions import *
import sys



symbol = "AMZN"
window_size = 1
episode_count = 1
start, end = "2018-01-01", "2020-12-31"


agent = Agent(window_size)

data = getStockDataDF(symbol, start, end, window_size)
l = len(data[symbol]) - 1
batch_size = 6

for e in range(episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0)
	state = np.append(state, 0)

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		print(t)
		action = agent.act(state)

		next_state = getState(data, t + 1)
		next_state = np.append(next_state, action)

		reward = 0
		price =  data[symbol].values[t]
		holdings = 0

		if action == 1: # buy
			holdings = 1000
			agent.inventory.append(price)
			reward = max( - price, 0)
			total_profit -= (price * 1000)
			print("Buy: " + formatPrice(price))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(price - bought_price, 0)
			total_profit += (price - bought_price) * 1000
			print("Sell: " + formatPrice(price) + " | Profit: " + formatPrice(price - bought_price) * 1000)
		
		else:
			print("Flat")

		done = True if t == l - 1 else False
		agent.memory.append((state, action, reward, next_state, done))
		state = next_state

		if done:
			print("--------------------------------")
			print("Total Profit: " + formatPrice(total_profit))
			print("--------------------------------")

		if len(agent.memory) > batch_size:
			agent.expReplay(batch_size)

	if e % 10 == 0:
		agent.model.save("models/model_ep" + str(e))