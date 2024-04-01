# Capitol to Capital: Using Lobbying Data as a Technical Indicator
## John Soeder, Sophia Rutman, and Rahul Dasgupta

### Overview
Lobbying is the act of influencing local, state level, or national politics through the use of external or internal lawyers. Large companies typically pay large sums of money to be represented in meetings discussing laws or policies that will positively affect their own business or negatively affect a competitor.  We looked at the ten corporations that put the largest sums of money into lobbying in order to track the price of their publicly traded shares over time and find a correlation between lobbying and price increase or decrease from 2018- 2022. These ten companies were Amazon (AMZN), Meta (META/FB), Comcast (CMCSA), Google (GOOGL),  Boeing (BA), Lockheed Martin (LMT), AT&T (T), Northrop Grumman (NOC), Raytheon (RTX) and Abbott Labs (ABT). The amount of money they lobbied with over this time period is represented here: ​

We utilized web scraping to gather lobbying data over this time frame from Quiver Quantitative and implemented Q-Learning to train and test two machine learning models. Our goal for this project is to find a correlation between the timing and amount of money spent in lobbying, and when to take a long, short or flat position with shares of the specific corporation that lobbied. ​

### Machine Learning Agents

We used the Pandas Python package to analyze the lobbying data CSV. After creating a DataFrame of different indicators from the lobbying data and stock price information, we utilized Tabular Q-Learning and  Deep Q Learning to train and test two different machine learning models. Each agent looked at one individual stock. ​

The DataFrame itself contained four columns: the stock price for a particular corporation, and a column for each of our three technical indicators. Each row corresponded to a different date over five years. Overall, we had 192 unique states as which a row could possibly be categorized for the tabular learner. Since the neural network uses a continuous state space, only rows with the exact same values will be in the same state. ​

Our reward function for the tabular learner was based on the difference in portfolio value of the current state and the following state. It was then modified depending on the actions and holdings of the previous day of training.
​
Our reward function for the neural network was based on the difference between the price per share during the last transaction and the current price per share. This difference was negative if the agent did not move from a flat position in order to encourage the model to trade. Otherwise, the sign of the reward value reflects the performance of the model.​

For both Agents, there are three possible actions: move into a long position, short position, or flat position. If the current action is the same as the action taken for the previous day, the position will not change. The long position was always +1000 shares, and the short -1000 shares. ​

We trained each learner with data from 2018-01-01 to 2020-12-31 and tested each learner with data from 2021-01-01 to 2022-01-25. ​

### Technical Indicators
#### Lobbying Indicator
An optimal window for each corporation is defined as the rolling average window of lobbying spending that returns the highest cumulative returns when tested with the Tabular Q-Learner. We looked at windows from 5 to 50 days at an interval of 5 days, creating 5 separate agents for each time window and averaging together the average cumulative returns of each tested agent after completing training. The window with the highest average cumulative return and this return are both stored. After moving through all 10-time windows, we determined the optimal time window as well as the highest average cumulative return. The final lobbying indicators were the rolling averages every day over our five-year period. 

#### Additional Indicators
We used a Relative Strength Index Indicator and a Moving Average Convergence and Divergence Oscillator (MACD) together with our Lobbying Indicator to generate our state space.

### Results
We found that the lobbying indicator has a clear effect on the return of our portfolio only when using the Tabular Q-Learner. The ability of the training data to beat the baseline for each share decreased as lobbying frequency decreased for the Tabular Q-Learner.

As for the Deep Q Learner, models trained on Northrop Grumman, Lockheed Martin, Google, and Amazon indicator values did not beat the benchmark consistently, even in training. The model beat the baseline for Comcast during testing, but this did not occur for other corporations. This success may have to do with the numbers of days that Comcast lobbied and the amount of money that they lobbied, as they have some of the highest totals of both. The failure of the other models is likely explained by insufficient training. We found no evidence that the lobbying indicator improves a Deep Q Learning Agent, but some of the trends observed in our models were promising. For example, we observed consistent improvement across episodes during train for the Google and Comcast models. Also, we observed consistent performance across different models for the same stock with similar episode counts.​
