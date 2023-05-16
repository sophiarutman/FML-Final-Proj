import search_env

class CalcCumulReturns:


    def __init__ (self, symbol, window):
        self.symbol = symbol
        self.window = window
        pass

    def calc_returns(self):
        
        cr_sum_train = 0
        cr_sum_test = 0

        for i in range(5):
            env = search_env.SearchEnvironment(fixed = 9.95, floating = 0.005, starting_cash = 200000, share_limit = 1000)
            
            cr_train = env.train_learner(start = "2018-01-01", end = "2020-12-31", symbol = self.symbol, trips = 500, window = self.window, dyna = 0,
                eps = 0.99, eps_decay = 0.99995)
            cr_sum_train += cr_train

            cr_test = env.test_learner(start = "2021-01-01", end = "2022-12-31", symbol = self.symbol, lobbyingWindow = self.window)
            cr_sum_test += cr_test

        average_train = cr_sum_train / 5
        average_test = cr_sum_test / 5

        print("Average Training CR of "  + str(self.symbol) + str(self.window) + ": " + str(average_train))
        print("Average Testing CR of " + str(self.symbol) + str(self.window) + ": " + str(average_test))


        
        return average_train, average_test
    

if __name__ == '__main__':

    results = {}
    #AMZN: done
    """FB:  File "/Users/srutman/Library/CloudStorage/OneDrive-BowdoinCollege/FML-Final-Proj/search_env.py", line 39, in prepare_world
    df["LobbyingQuantile"] = pd.qcut(df['Lobbying'], 4, labels=[ "1", "2", "3", "4"])
    File "/opt/homebrew/lib/python3.10/site-packages/pandas/core/reshape/tile.py", line 379, in qcut
    fac, bins = _bins_to_cuts(
    File "/opt/homebrew/lib/python3.10/site-packages/pandas/core/reshape/tile.py", line 420, in _bins_to_cuts
    raise ValueError(
    ValueError: Bin edges must be unique: array([ 500.,  500.,  750.,  750., 2500.])."""
    #{"AMZN": 30, "FB": 40, "CMCSA":40, "GOOGL":45, "BA":25, "LMT":40, "T":25, "NOC":15, "RTX":15, "ABT":35}
    windowdict = {"FB":40, "ABT":35}


    for symWin in windowdict:
        print(symWin)
        calculating = CalcCumulReturns(symWin, windowdict[symWin])
        train, test = calculating.calc_returns()
        results[symWin] = (train, test)
    
    print(results)
