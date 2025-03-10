import pandas as pd
import urllib

data = pd.read_csv("/home/trent/Desktop/data-structuring/data/structured_data/structured_cleaned_structured_cleaned_healthcare_reform_data.csv")

data = data.dropna()

try:
    data = data.drop('Images', axis=1)
    data = data.drop('References', axis=1)
    data = data.drop('Categories', axis=1)
    data = data.drop('Url Valid', axis=1)
    print("dropped successful")
except Exception as e:
     print(f"Column not found: {e}")

#dictionary to be used to rename column names
rename_dict = {
    #"Headline": "Title",
    "Url": "URL",
    #"Description": "Summary",
    #"Fullcontent": "Content"
}

reorder_dict = ['Title', 'Summary', 'Content', 'URL']

data = data.rename(columns=rename_dict)[reorder_dict]

# data = data.rename(columns=rename_dict)

# dropping duplicates 
# #data = data.drop_duplicates(subset=['Headline'], keep='first')

def is_valid_url(url):
    try:
        result = urllib.parse.urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# # Apply the URL Validation
data ['Is URL Valid'] = data['URL'].apply(is_valid_url)

data = data.reset_index(drop=True)
print(data.info())
# print(data['Link'][170])

data.to_csv("/home/trent/Desktop/data-structuring/data/structured_data/structured_cleaned_healthcare_reform_data.csv", index=False)
#print("finished processing ... ")

