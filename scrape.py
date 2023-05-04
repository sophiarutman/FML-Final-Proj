from bs4 import BeautifulSoup
import requests
import pandas as pd


class DataScraping:

    def __init__(self, start, end) -> None:
        self.dates = pd.date_range(start, end)
        self.lobbying_df = pd.DataFrame(index=self.dates)
        

    def date_translation(self, date_str : str):
        # Split the date string into its components
        date_components = date_str.split()

        # Extract the year, month, and day components
        year = date_components[2][:-1]

        if date_components[0] == 'Jan.':
            month = "01"
        elif date_components[0] == 'Feb.':
            month = "02"
        elif date_components[0] == 'March':
            month = "03"
        elif date_components[0] == 'April':
            month = "04"
        elif date_components[0] == 'May':
            month = "05"
        elif date_components[0] == 'June':
            month = "06"
        elif date_components[0] == 'July':
            month = "07"
        elif date_components[0] == 'Aug.':
            month = "08"
        elif date_components[0] == 'Sept.':
            month = "09"
        elif date_components[0] == 'Oct.':
            month = "10"
        elif date_components[0] == 'Nov.':
            month = "11"
        elif date_components[0] == 'Dec.':
            month = "12"
        else:
            # If the date string is invalid, raise an error
            raise ValueError("Invalid date string")

        day = date_components[1][:-1]

        # Combine the components into a string in the desired format
        formatted_date = f"{year}-{month}-{day.zfill(2)}"

        return formatted_date

    def scrape_lobbying_history(self, symbol, url):

        result = requests.get(url)
        doc = BeautifulSoup(result.text, "html.parser")

        table = doc.find(id="myTable")
        rows = table.find_all("tr")

        self.lobbying_df[symbol] = 0

        for i in range(1, len(rows)):
            current_row = rows[i]
            data = current_row.find_all("td")
            date_string = data[1].string
            date = self.date_translation(date_string)

            amount = int((data[2].string).replace(",", ""))

            if date in self.lobbying_df.index:
                self.lobbying_df.loc[date, symbol] += amount

        return self.lobbying_df

if __name__ == "__main__":

    train_start, train_end = "2018-01-01", "2020-12-31"

    scraper = DataScraping(train_start, train_end)

    symbol = "AMZN"
    url = "https://www.quiverquant.com/lobbyingsearch/ticker/AMZN?searchticker=AMZN"
    scraper.scrape_lobbying_history(symbol, url)
    
    symbol = "META"
    url = "https://www.quiverquant.com/lobbyingsearch/ticker/META?searchticker=META"
    scraper.scrape_lobbying_history(symbol, url)

    symbol = "CMCSA"
    url = "https://www.quiverquant.com/lobbyingsearch/ticker/CMCSA?searchticker=CMCSA"
    scraper.scrape_lobbying_history(symbol, url)

    symbol = "GOOGL"
    url = "https://www.quiverquant.com/lobbyingsearch/ticker/GOOGL?searchticker=GOOGL"
    scraper.scrape_lobbying_history(symbol, url)
    
    symbol = "BA"
    url = "https://www.quiverquant.com/lobbyingsearch/ticker/BA?searchticker=BA"
    scraper.scrape_lobbying_history(symbol, url)

    symbol = "LMT"
    url = "https://www.quiverquant.com/lobbyingsearch/ticker/LMT?searchticker=LMT"
    scraper.scrape_lobbying_history(symbol, url)

    symbol = "T"
    url = "https://www.quiverquant.com/lobbyingsearch/ticker/T?searchticker=T"
    scraper.scrape_lobbying_history(symbol, url)

    symbol = "NOC"
    url = "https://www.quiverquant.com/lobbyingsearch/ticker/NOC?searchticker=NOC"
    scraper.scrape_lobbying_history(symbol, url)
    
    symbol = "RTX"
    url = "https://www.quiverquant.com/lobbyingsearch/ticker/RTX?searchticker=RTX"
    scraper.scrape_lobbying_history(symbol, url)

    symbol = "ABT"
    url = "https://www.quiverquant.com/lobbyingsearch/ticker/ABT?searchticker=ABT"
    result = scraper.scrape_lobbying_history(symbol, url)

    print(result.to_string())

  
    #test_start, test_end = "2021-01-01", "2022-12-31"