import pandas as pd
import urllib

data = pd.read_csv("cleaned_javascript_tutorial_data.csv")

#data = data.dropna()

# try:
#     data = data.drop('Posted', axis=1)
#     print("dropped successful")
# except Exception as e:
#      print(f"Column not found: {e}")
     
# #data = data.drop_duplicates(subset=['Headline'], keep='first')

# def is_valid_url(url):
#     try:
#         result = urllib.parse.urlparse(url)
#         return all([result.scheme, result.netloc])
#     except ValueError:
#         return False

# # Apply the URL Validation
# #data ['Link_valid'] = data['Link'].apply(is_valid_url)

data = data.reset_index(drop=True)
print(data.info())
# print(data['Link'][170])

# data.to_csv("/home/trent/Desktop/data-structuring/data/structured_data/structured_cleaned_War.csv", index=False)
#print("finished processing ... ")

