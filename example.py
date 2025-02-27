import pandas as pd

data = pd.read_csv("/home/trent/Desktop/data-structuring/data/structured_data/structured_cleaned_wikipedia_data_Statue of Liberty.csv")

data = data.dropna()

print(data.info())
