import pandas as pd
import numpy as np

class IndicatorRetrieval:        

    def __init__(self) -> None:
        pass

    def get_data(self, start, end, symbol, data_file="/Users/jsoeder/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/FML/FML-Final-Proj/lobbying.csv"):
        # Construct an empty DataFrame with the requested date range.
        dates = pd.date_range(start, end)
        df = pd.DataFrame(index=dates)

        # Read SPY.
        df_spy = pd.read_csv('/Users/jsoeder/Library/CloudStorage/OneDrive-BowdoinCollege/Desktop/FML/FML-Final-Proj/data/SPY.csv', 
            index_col=['Date'], parse_dates=True, na_values=['nan'], usecols=['Date',"Adj Close"])

        # Use SPY to eliminate non-market days.
        df = df.join(df_spy, how='inner')
        df = df.rename(columns={"Adj Close":'SPY'})

        # Append the data for the symbol, retaining all market-open days.
        df_sym = pd.read_csv(data_file, index_col=['Date'], parse_dates=True, na_values=['nan'], usecols=['Date',symbol])
        df = df.join(df_sym, how='left')

        del df['SPY']

        return df

    def main(self):
        data = self.get_data('2018-01-01','2022-12-31','AMZN')
        print(data.to_string())


#c = IndicatorRetrieval()
#c.main()