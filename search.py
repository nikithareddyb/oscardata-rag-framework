import pandas as pd
import chromadb

df = pd.read_csv('oscar-dataset/the_oscar_award.csv')

df = df.loc[df['year_ceremony'] == 2023]
df = df.dropna(subset=['film'])
df.loc[:, 'category'] = df['category'].str.lower()
df.loc[:, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ' to win the award'
df.loc[df['winner'] == False, 'text'] = df['name'] + ' got nominated under the category, ' + df['category'] + ', for the film ' + df['film'] + ' but did not win'

client = chromadb.Client()
collection = client.get_or_create_collection("oscars-2023")
docs = df["text"].tolist()
ids = [str(x) for x in df.index.tolist()]

collection.add(
    documents=docs,
    ids=ids
)

results = collection.query(
    query_texts=["best music"],
    n_results=3
)

print(results['documents'])
