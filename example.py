import pandas as pd
import urllib

data = pd.read_csv("data/structured_data/structured_cleaned_digital_banking_data.csv")

data = data.dropna()

try:
    data = data.drop('Links', axis=1)
    print("dropped successful")
except Exception as e:
     print(f"Column not found: {e}")
     
#data = data.drop_duplicates(subset=['Headline'], keep='first')

def is_valid_url(url):
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# # Apply the URL Validation
data ['Url_valid'] = data['Url'].apply(is_valid_url)

data = data.reset_index(drop=True)
print(data.info())
# print(data['Link'][170])

data.to_csv("data/structured_data/structured_cleaned_digital_banking_data.csv", index=False)
#print("finished processing ... ")

