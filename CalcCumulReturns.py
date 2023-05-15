import search_env
import pandas as pd

class CalcCumulReturns:


    def __init__ (self, symbol, window):
        self.symbol = symbol
        self.window = window
        pass

    def optimize_window(self):
        
        cr_sum_train = 0
        cr_sum_test = 0

        for i in range(5):
            env = search_env.SearchEnvironment(fixed = 9.95, floating = 0.005, starting_cash = 200000, share_limit = 1000)
            
            cr_train = env.train_learner(start = "2018-01-01", end = "2020-12-31", symbol = self.symbol, trips = 500, window = self.window, dyna = 0,
                eps = 0.99, eps_decay = 0.99995)
            cr_sum_train += cr_train

            cr_test = env.test_learner(start = "2021-01-01", end = "2022-12-31", symbol = self.symbol, lobbyingWindow = self.window)
            cr_sum_train += cr_test

        average_train = cr_sum_train / 5
        average_test = cr_sum_test / 5

        print("Average Training CR of Window " + str(self.window) + ": " + str(average_train))
        print("Average Training CR of Window " + str(self.window) + ": " + str(average_test))


        
        return average_train
    

if __name__ == '__main__':

    results = {}

    windowdict = {"AMZN":30,"FB":40,"CMCSA":40, "GOOGL":45, "BA":25, "LMT":40, "T":25, "NOC":15, "RTX":15, "ABT":35}


    for symWin in windowdict:
        print(symWin)
        opt = ParameterSearch(symWin.key, symWin.value)
        best_window, cr = opt.optimize_window()
        results[sym] = (best_window, cr)
    
    print(results)
