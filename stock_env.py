from TabularQLearner import TabularQLearner
import argparse
import tech_ind
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class StockEnvironment:
    def __init__ (self, fixed = None, floating = None, starting_cash = None, share_limit = None):
        self.shares = share_limit
        self.fixed_cost = fixed
        self.floating_cost = floating
        self.starting_cash = starting_cash
        self.df = pd.DataFrame()
        self.QTrader = None

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
        df = tech_ind.Lobbying(df, symbol, lobbyingWindow)
        #insert indicator for Lobbying Data
        df = df[[symbol, "MACD", "RSI", "Lobbying"]]
        df["RSIQuantile"] = pd.qcut(df["RSI"], 4,labels=["0", "1", "2", "3"])
        df["MACDQuantile"] = pd.qcut(df["MACD"], 4, labels=["0", "1", "2", "3"])
        df["Lobbying"] = pd.qcut(df["Lobbying"], 4, labels=["0", "1", "2", "3"])
        self.df = df
        df = df.ffill().bfill()
        return df

    def calc_state (self, df, day, holdings):
        """ Quantizes the state to a single number. """
        rsi = df.at[day, "RSIQuantile"]
        macd = df.at[day, "MACDQuantile"]
        lobbying = df.at[day, "LobbyingQuantile"]
        if holdings < 0:
            hold = "0"
        elif holdings > 0:
            hold = "2"
        else:
            hold = "1"
        strnum = "1" + rsi + macd + lobbying + hold 
        state = int(strnum)
        return state

    def train_learner(self, start = None, end = None, symbol = None, trips = 0, dyna = 0, eps = 0.0, eps_decay = 0.0):
        """
        Construct a Q-Learning trader and train it through many iterations of a stock
        world. Store the trained learner in an instance variable for testing.
        Print a summary result of what happened at the end of each trip.
        Feel free to include portfolio stats or other information, but AT LEAST:
        Trip 499 net result: $13600.00
        """
        #update number of states to reflect states
        self.QTrader = TabularQLearner(133333, 3, epsilon=eps, epsilon_decay=eps_decay, dyna=dyna)
        world_df = self.prepare_world(start, end, symbol, "./data")
        first_day = world_df.index[0]
        start_state = self.calc_state(world_df, first_day, 0)
        cur_port_val = self.starting_cash
        for i in range(trips):
            action = self.QTrader.test(start_state)
            prev_date = first_day
            prev_holding = 0
            cash = self.starting_cash
            for date in world_df.index:
                if date == first_day:
                    continue
                prev_cash = cash
                price = self.shares * world_df[symbol].loc[date]
                if action == 0:
                    #action is SHORT
                    holdings = -1 * self.shares
                    if prev_holding == 0:
                        #if prev position is FLAT, add shares to cash
                        cash += price
                        cash -= (price * self.floating_cost) + self.fixed_cost
                    elif prev_holding > 0:
                        #if prev position is LONG, add double shares to cash 
                        cash += price * 2
                        cash -= (price * self.floating_cost * 2) + self.fixed_cost
                elif action == 1:
                    #action is FLAT
                    holdings = 0
                    if prev_holding > 0:
                        #if prev position is LONG, add shares to cash
                        cash += price
                    elif prev_holding < 0:
                        #if prev position is SHORT, substract purchased shares from cash
                        cash -= price
                    if prev_holding != 0:
                        cash -= (price * self.floating_cost) + self.fixed_cost
                else:
                    #action is LONG
                    holdings = self.shares
                    if prev_holding == 0:
                        #if prev position is FLAT, subtract purchased shares from cash
                        cash -= price
                        cash -= (price * self.floating_cost) + self.fixed_cost
                    elif prev_holding < 0:
                        #if prev position is SHORT, subtract double purchased shares from cash
                        cash -= price * 2
                        cash -= (price * self.floating_cost * 2) + self.fixed_cost
                sPrime = self.calc_state(world_df, date, holdings)
                prev_price = world_df[symbol].loc[prev_date]

                cur_port_val = holdings * world_df[symbol].loc[date] + cash
                prev_port_val = prev_holding * prev_price + prev_cash

                reward = cur_port_val - prev_port_val

                if action == 0 and prev_holding < 0:
                    if price < prev_price:
                        reward += (prev_price - price) / 8000
                        #why 8000?
                    else:
                        reward += (price - prev_price) / 8000
                elif action == 2 and prev_holding > 0:
                    if price > prev_price:
                        reward += (price - prev_price) / 8000
                    else:
                        reward += (prev_price - price) / 8000
                
                action = self.QTrader.train(sPrime, reward)
                prev_holding = holdings
                prev_date = date
            print("After " + str(i) + " trips, the net gain is " + str(cur_port_val - self.starting_cash))
            print("Cumulative Returns: " + str(cur_port_val / self.starting_cash - 1))

        return 

    def test_learner(self, start = None, end = None, symbol = None):
        """
        Evaluate a trained Q-Learner on a particular stock trading task.
        Print a summary result of what happened during the test.
        Feel free to include portfolio stats or other information, but AT LEAST:
        Test trip, net result: $31710.00
        Benchmark result: $6690.0000
        """
        world_df = self.prepare_world(start, end, symbol, "./data")
        first_day = world_df.index[0]
        start_state = self.calc_state(world_df, first_day, 0)
        cur_port_val = self.starting_cash
        prev_holding = 0
        cash = self.starting_cash
        world_df["PORT_CR"] = self.starting_cash
        prev_state = start_state
        for date in world_df.index:
            action = self.QTrader.test(prev_state)
            price = self.shares * world_df[symbol].loc[date]
            if action == 0:
            #action is SHORT
                holdings = -1 * self.shares
                if prev_holding == 0:
                    #if prev position is FLAT, add shares to cash
                    cash += price
                    cash -= (price * self.floating_cost) + self.fixed_cost
                elif prev_holding > 0:
                    #if prev position is LONG, add double shares to cash 
                    cash += price * 2
                    cash -= (price * self.floating_cost * 2) + self.fixed_cost
            elif action == 1:
                #action is FLAT
                holdings = 0
                if prev_holding > 0:
                    #if prev position is LONG, add shares to cash
                    cash += price
                elif prev_holding < 0:
                    #if prev position is SHORT, substract purchased shares from cash
                    cash -= price
                if prev_holding != 0:
                    cash -= (price * self.floating_cost) + self.fixed_cost
            else:
                #action is LONG
                holdings = self.shares
                if prev_holding == 0:
                    #if prev position is FLAT, subtract purchased shares from cash
                    cash -= price
                    cash -= (price * self.floating_cost) + self.fixed_cost
                elif prev_holding < 0:
                    #if prev position is SHORT, subtract double purchased shares from cash
                    cash -= price * 2
                    cash -= (price * self.floating_cost * 2) + self.fixed_cost
            sPrime = self.calc_state(world_df, date, holdings)
            prev_state = sPrime
            prev_holding = holdings
            cur_port_val = holdings * world_df[symbol].loc[date] + cash
            world_df.at[date, "PORT_CR"] = cur_port_val
        bench_cash = self.starting_cash - self.shares * world_df.at[first_day, symbol]
        world_df["BenchCash"] = bench_cash + world_df[symbol] * self.shares 
        world_df["CR"] = world_df["BenchCash"] / world_df["BenchCash"].loc[first_day] - 1
        world_df["PORT_CR"] = world_df["PORT_CR"] / world_df.at[first_day, "PORT_CR"] - 1


        print("Testing... Net Gain is " + str(cur_port_val - self.starting_cash))
        print("Cumulative Returns: " + str(cur_port_val/self.starting_cash - 1))
        print("Benchmark Returns: " + str(world_df["CR"].iloc[-1]))

        #plt.figure(1)
        #plt.suptitle("Q-Learner Performance Trading " + symbol + " vs. Buy and Hold Benchmark")
        #plt.title("Floating Cost of " + str(self.floating_cost))
        #plt.plot(world_df["CR"])
        #plt.plot(world_df["PORT_CR"])
        #plt.xlabel("Date")
        #plt.ylabel("Cumulative Return")
        #plt.legend([symbol, "Portfolio"])
        #plt.show()
        return
    
    def gridworld(self, start, end, sym, datafolder): 

        def prepare_world_grid (self, start_date, end_date, symbol, data_folder, lobbyingWindow):
            """
            Read the relevant price data and calculate some indicators.
            Return a DataFrame containing everything you need.
            """
            dates = pd.date_range(start_date, end_date)
            df = pd.DataFrame(index=dates)
            df_symbol = pd.read_csv(f'{data_folder}/' + symbol + '.csv', index_col=['Date'], parse_dates=True, na_values=['nan'], usecols=['Date',"Adj Close"])
            df = df.join(df_symbol, how="inner")
            df = df.rename(columns={"Adj Close": symbol})
            df = tech_ind.Lobbying(df, symbol, lobbyingWindow)
            #insert indicator for Lobbying Data
            df = df[[symbol, "Lobbying"]]
            df["Lobbying"] = pd.qcut(df["Lobbying"], 4, labels=["0", "1", "2", "3"])
            self.df = df
            df = df.ffill().bfill()
            return df
        
        def calc_state_grid(self, df, day, holdings):
            """ Quantizes the state to a single number. """
            lobbying = df.at[day, "LobbyingQuantile"]
            if holdings < 0:
                hold = "0"
            elif holdings > 0:
                hold = "2"
            else:
                hold = "1"
            strnum = "1" + lobbying + hold 
            state = int(strnum)
            return state
        
        def train_learner(self, start = None, end = None, symbol = None, trips = 0, dyna = 0, eps = 0.0, eps_decay = 0.0):
            """
            Construct a Q-Learning trader and train it through many iterations of a stock
            world. Store the trained learner in an instance variable for testing.
            Print a summary result of what happened at the end of each trip.
            Feel free to include portfolio stats or other information, but AT LEAST:
            Trip 499 net result: $13600.00
            """
            #update number of states to reflect states
            self.QTrader = TabularQLearner(133333, 3, epsilon=eps, epsilon_decay=eps_decay, dyna=dyna)
            world_df = self.prepare_world_grid(start, end, symbol, "./data")
            first_day = world_df.index[0]
            start_state = self.calc_state_grid(world_df, first_day, 0)
            cur_port_val = self.starting_cash
            for i in range(trips):
                action = self.QTrader.test(start_state)
                prev_date = first_day
                prev_holding = 0
                cash = self.starting_cash
                for date in world_df.index:
                    if date == first_day:
                        continue
                    prev_cash = cash
                    price = self.shares * world_df[symbol].loc[date]
                    if action == 0:
                        #action is SHORT
                        holdings = -1 * self.shares
                        if prev_holding == 0:
                            #if prev position is FLAT, add shares to cash
                            cash += price
                            cash -= (price * self.floating_cost) + self.fixed_cost
                        elif prev_holding > 0:
                            #if prev position is LONG, add double shares to cash 
                            cash += price * 2
                            cash -= (price * self.floating_cost * 2) + self.fixed_cost
                    elif action == 1:
                        #action is FLAT
                        holdings = 0
                        if prev_holding > 0:
                            #if prev position is LONG, add shares to cash
                            cash += price
                        elif prev_holding < 0:
                            #if prev position is SHORT, substract purchased shares from cash
                            cash -= price
                        if prev_holding != 0:
                            cash -= (price * self.floating_cost) + self.fixed_cost
                    else:
                        #action is LONG
                        holdings = self.shares
                        if prev_holding == 0:
                            #if prev position is FLAT, subtract purchased shares from cash
                            cash -= price
                            cash -= (price * self.floating_cost) + self.fixed_cost
                        elif prev_holding < 0:
                            #if prev position is SHORT, subtract double purchased shares from cash
                            cash -= price * 2
                            cash -= (price * self.floating_cost * 2) + self.fixed_cost
                    sPrime = self.calc_state(world_df, date, holdings)
                    prev_price = world_df[symbol].loc[prev_date]

                    cur_port_val = holdings * world_df[symbol].loc[date] + cash
                    prev_port_val = prev_holding * prev_price + prev_cash

                    reward = cur_port_val - prev_port_val

                    if action == 0 and prev_holding < 0:
                        if price < prev_price:
                            reward += (prev_price - price) / 8000
                            #why 8000?
                        else:
                            reward += (price - prev_price) / 8000
                    elif action == 2 and prev_holding > 0:
                        if price > prev_price:
                            reward += (price - prev_price) / 8000
                        else:
                            reward += (prev_price - price) / 8000
                    
                    action = self.QTrader.train(sPrime, reward)
                    prev_holding = holdings
                    prev_date = date
                print("After " + str(i) + " trips, the net gain is " + str(cur_port_val - self.starting_cash))
                print("Cumulative Returns: " + str(cur_port_val / self.starting_cash - 1))
            #call train, test for many different values, find greatest cumulative return out of all of them to use

        return 

        
        



        
if __name__ == '__main__':
    # Load the requested stock for the requested dates, instantiate a Q-Learning agent,
    # and let it start trading.
    parser = argparse.ArgumentParser(description='Stock environment for Q-Learning.')
    date_args = parser.add_argument_group('date arguments')
    date_args.add_argument('--train_start', default='2018-01-01', metavar='DATE',help='Start of training period.')
    date_args.add_argument('--train_end', default='2019-12-31', metavar='DATE',help='End of training period.')
    date_args.add_argument('--test_start', default='2020-01-01', metavar='DATE', help='Start of testing period.')
    date_args.add_argument('--test_end', default='2021-12-31', metavar='DATE',help='End of testing period.')

    learn_args = parser.add_argument_group('learning arguments')
    learn_args.add_argument('--dyna', default=0, type=int, help='Dyna iterations perexperience.')
    learn_args.add_argument('--eps', default=0.99, type=float, metavar='EPSILON', help='Starting epsilon for epsilon-greedy.')
    learn_args.add_argument('--eps_decay', default=0.99995, type=float, metavar='DECAY', help='Decay rate for epsilon-greedy.')

    sim_args = parser.add_argument_group('simulation arguments')
    sim_args.add_argument('--cash', default=200000, type=float, help='Starting cash for the agent.')
    sim_args.add_argument('--fixed', default=0.00, type=float, help='Fixedtransaction cost.')
    sim_args.add_argument('--floating', default='0.00', type=float, help='Floating transaction cost.')
    sim_args.add_argument('--shares', default=1000, type=int, help='Number of shares to trade (also position limit).')
    sim_args.add_argument('--symbol', default='DIS', help='Stock symbol to trade.')
    sim_args.add_argument('--trips', default=500, type=int, help='Round trips through training data.')

    args = parser.parse_args()
    # Create an instance of the environment class.
    env = StockEnvironment(fixed = args.fixed, floating = args.floating, starting_cash = args.cash, share_limit = args.shares)

    # Construct, train, and store a Q-learning trader.
    env.train_learner(start = args.train_start, end = args.train_end, symbol = args.symbol, trips = args.trips, dyna = args.dyna,
                    eps = args.eps, eps_decay = args.eps_decay)

    # Test the learned policy and see how it does.
    # In sample.
    env.test_learner(start = args.train_start, end = args.train_end, symbol = args.symbol)

    # Out of sample. Only do this once you are fully satisfied with the in sample performance!
    env.test_learner(start = args.test_start, end = args.test_end, symbol = args.symbol)