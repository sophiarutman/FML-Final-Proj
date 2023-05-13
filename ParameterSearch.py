import search_env
import pandas as pd

class ParameterSearch:


    def __init__ (self, symbol):
        self.symbol = symbol
        pass

    def optimize_window(self):

        sym = self.symbol
        best_window = 0
        highest_cr = -10

        for i in range(10):
            cur_window = i * 5
            cr_sum = 0
            if cur_window == 0:
                cur_window = 1

            for i in range(5):
                env = search_env.SearchEnvironment(fixed = 9.95, floating = 0.005, starting_cash = 200000, share_limit = 1000)
                cr = env.train_learner(start = "2018-01-01", end = "2020-12-31", symbol = sym, trips = 500, window = cur_window, dyna = 0,
                    eps = 0.99, eps_decay = 0.99995)
                cr_sum += cr

            average = cr_sum / 5

            print("Average CR of Window " + str(cur_window) + ": " + str(average))
            if average > highest_cr:
                best_window = cur_window
                highest_cr = average
        
        return best_window, highest_cr
    

if __name__ == '__main__':

    results = {}

    symbols = ["AMZN","FB","CMCSA","GOOGL","BA","LMT","T","NOC",'RTX',"ABT"]

    for sym in symbols:
        print(sym)
        opt = ParameterSearch(sym)
        best_window, cr = opt.optimize_window()
        results[sym] = (best_window, cr)
    
    print(results)
