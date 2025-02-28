import pandas as pd
import logging

data = pd.read_csv("/home/trent/Desktop/data-structuring/data/structured_data/structured_cleaned_healthcare_reform_data.csv")

# data = data.dropna()
# data = data.drop('Unnamed: 0', axis=1)
# data = data.drop_duplicates(subset=['Headline'], keep='first', inplace=True)
print(data.info())
# print(data['Content'][12])

# data.to_csv("/home/trent/Desktop/data-structuring/data/structured_data/structured_cleaned_healthcare_reform_data.csv")
logging.info("finished export")