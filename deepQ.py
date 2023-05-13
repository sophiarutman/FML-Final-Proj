import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas_datareader as data_reader

import tech_ind
import IndicatorRetrieval as ir

from tensorflow import keras

from tqdm import tqdm_notebook, tqdm
from collections import deque


class DeepQ():
  
    def __init__(self, states, actions=3, name="DeepQ"):

        self.states = states
        self.actions = actions
        self.memory = deque([2000])
        self.inventory = []
        self.model = None
        self.model_name = name

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995

    def model_builder(self):
        
        model = keras.models.Sequential()
        
        model.add(keras.layers.Dense(units=32, activation='relu', input_dim=self.states))
        model.add(keras.layers.Dense(units=64, activation='relu'))
        model.add(keras.layers.Dense(units=128, activation='relu'))
        model.add(keras.layers.Dense(units=self.actions, activation='linear'))

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001))

        self.model = model
        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.actions)

        action = self.model.predict(state)
        return action

    def batch_train(self, batch_size):

        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        for state, action, reward, next_state, done in batch:
            reward = reward
            if not done:
                reward = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

        target = self.model.predict(state)
        target[0][action] = reward

        self.model.fit(state, target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay


    def stock_price_format(self, n):
        if n < 0:
            return "- # {0:2f}".format(abs(n))
        else:
            return "$ {0:2f}".format(abs(n))


    def prepare_world (self, start_date, end_date, symbol, data_folder, lobbyingWindow):
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

        
        self.df = df
        df = df.ffill().bfill()
        return df[symbol]

    def state_creator(self, data, timestep, window_size):
        starting_id = timestep - window_size + 1

        if starting_id >= 0:
            windowed_data = data[starting_id : timestep + 1]
        else:
            windowed_data = [data[0]] * abs(starting_id) + list(data[0 : timestep + 1])

        state = []
        for i in range(window_size - 1):
            diff = windowed_data[i + 1] - windowed_data[i]
            sigmoid = 1 / (1 + math.exp(-diff))
            state.append(sigmoid)

        return np.array([state])

  
if __name__ == '__main__':

    window_size = 10
    episodes = 1000

    trader = DeepQ(window_size)
    trader.model_builder()
    trader.model.summary()

    data = trader.prepare_world("2018-01-01", "2020-12-31", "AMZN", "./data", 30)

    batch_size = 32
    data_samples = len(data) - 1

    for episode in range(1, episodes + 1):
  
        print("Episode: {}/{}".format(episode, episodes))
        
        state = trader.state_creator(data, 0, window_size + 1)
        
        total_profit = 0
        trader.inventory = []
        
        for t in tqdm(range(data_samples)):
            
            action = trader.trade(state)
            
            next_state = trader.state_creator(data, t+1, window_size + 1)
            reward = 0
            
            if action == 1: #Buying
                trader.inventory.append(data[t])
                print("AI Trader bought: ", trader.stocks_price_format(data[t]))
            
            elif action == 2 and len(trader.inventory) > 0: #Selling
                buy_price = trader.inventory.pop(0)           
                reward = max(data[t] - buy_price, 0)
                total_profit += data[t] - buy_price
                print("AI Trader sold: ", trader.stocks_price_format(data[t]), " Profit: " + trader.stocks_price_format(data[t] - buy_price) )
            
            if t == data_samples - 1:
                done = True
            else:
                done = False
            
            trader.memory.append((state, action, reward, next_state, done))
            
            state = next_state
            
            if done:
                print("########################")
                print("TOTAL PROFIT: {}".format(total_profit))
                print("########################")
            
            if len(trader.memory) > batch_size:
                trader.batch_train(batch_size)
            
        if episode % 10 == 0:
            trader.model.save("ai_trader_{}.h5".format(episode))

            
