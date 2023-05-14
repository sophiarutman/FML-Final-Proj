from dqn import Agent
from dqn_functions import *
import sys



symbol = "AMZN"
window_size = 1
episode_count = 500
start, end = "2018-01-01", "2020-12-31"


agent = Agent(window_size)

data = getStockDataDF(symbol, start, end, window_size)
l = len(data[symbol]) - 1
batch_size = 32

for e in range(episode_count + 1):
	print("Episode " + str(e) + "/" + str(episode_count))
	state = getState(data, 0)
	next_state.append(0) 

	total_profit = 0
	agent.inventory = []

	for t in range(l):
		action = agent.act(state)

		next_state = getState(data, t + 1)
		next_state.append(action) 

		reward = 0
		price =  df[symbol].values[i]

		if action == 1: # buy
			agent.inventory.append(data[t][symbol])
			print("Buy: " + formatPrice(data[t][symbol]))

		elif action == 2 and len(agent.inventory) > 0: # sell
			bought_price = agent.inventory.pop(0)
			reward = max(data[t][symbol] - bought_price, 0)
			total_profit += data[t][symbol] - bought_price
			print("Sell: " + formatPrice(data[t][symbol]) + " | Profit: " + formatPrice(data[t][symbol] - bought_price))

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