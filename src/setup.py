import pandas as pd

data = pd.read_csv('../bitstampUSD_1-min_data_2012-01-01_to_2020-12-31.csv').dropna()
print(str(data))
data.to_csv('../bitstamp.csv')
