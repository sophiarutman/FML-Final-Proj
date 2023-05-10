import search_env
import pandas as pd

class ParameterSearch:


    def __init__ (self, symbol):
        self.symbol = symbol
        pass

    def optimize_window(self):

        best_window = 0
        highest_cr = 0

        for i in range(1):
            cur_window = i * 5
            cr_sum = 0
            if cur_window == 0:
                cur_window = 1

            for i in range(5):
                env = search_env.SearchEnvironment(fixed = 9.95, floating = 0.005, starting_cash = 200000, share_limit = 1000)
                cr = env.train_learner(start = "2018-01-01", end = "2020-12-31", symbol = self.symbol, trips = 500, window = cur_window, dyna = 0,
                    eps = 0.99, eps_decay = 0.99995)
                cr_sum += cr

            average = cr_sum / 5
            if average > highest_cr:
                best_window = cur_window
                highest_cr = average
        
        return best_window, highest_cr
    

if __name__ == '__main__':

    results = {}

    #symbols = ["AMZN","META","CMCSA","GOOGL","BA","LMT","T","NOC",'RTX',"ABT"]
    symbols = ["AMZN"]

    for sym in symbols:
        opt = ParameterSearch(sym)
        best_window, cr = opt.optimize_window()
        print(best_window)
        results[sym] = (best_window, cr)
    
    print(results)
