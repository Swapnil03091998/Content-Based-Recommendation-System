pip install -U sentence-transformers
pip install tqdm

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

data = pd.read_csv("/kaggle/input/best-books-10k-multi-genre-data/goodreads_data.csv").drop("Unnamed: 0", axis=1)

# Preprocess the data
data = data.dropna()
data = data.drop_duplicates()

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings1 = model.encode(data['Genres'].iloc[0]+data['Description'].iloc[0])

embeddings2 = model.encode(data['Genres'].iloc[1]+data['Description'].iloc[1])

cos_sim = util.cos_sim(embeddings1, embeddings2)
print("Cosine-Similarity:", cos_sim)

book_description_genres = list()

for i in range(data.shape[0]):
    #print(i)
    book_description_genres.append(str(data['Genres'].iloc[i])+str(data['Description'].iloc[i]))
    
#Encode all sentences
embeddings = model.encode(book_description_genres)

#Compute cosine similarity between all pairs
cos_sim = util.cos_sim(embeddings, embeddings)

# Recommendation Generation
def get_recommendations(book_id, similarity_matrix, k=5):
    # Get the top k similar books to the given book_id
    similar_books = list(enumerate(similarity_matrix[book_id]))
    similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)
    similar_books = similar_books[1:k+1]
    recommended_books = [data.iloc[i[0]]["Book"] for i in similar_books]
    return recommended_books

# Example Usage
book_id = 18
recommended_books = get_recommendations(book_id, cos_sim)
print(f"Recommended books for {data.iloc[book_id]['Book']}: {recommended_books}")

#Calculating Average Precision using ratings

precision_values = []

for i, book in tqdm(data[:1000].iterrows(), desc="Progress Bar"):
    recommended_books = get_recommendations(i, cos_sim, k=5)
    #get ratings of all recommended books
    ratings = data[data['Book'].isin(recommended_books)]['Avg_Rating']
    #calculate TP - high rating books(relevant books)
    true_positives = len([x for x in ratings if x>=4])
    # False positive: recommended item is not relevant
    false_positives = len(recommended_books) - true_positives
    
    # Precision: proportion of recommended items that are relevant
    precision = true_positives / (true_positives + false_positives)
    
    precision_values.append(precision)
    

print("average precision", round(sum(precision_values)/len(precision_values),2))

data1 = data.copy()
data1['Genres'] = data1['Genres'].apply(lambda x: x.replace("[", "").replace("]", "").strip())

def get_intersecting_genres(i, j):
    genres1 = {x.strip() for x in data1.iloc[i]['Genres'].split(",")}
    genres2 = {x.strip() for x in data1.iloc[j]['Genres'].split(",")}
    int_genres = genres1.intersection(genres2)
    
    return len(int_genres)
  
#Calculating MAP - Mean Average Precision using genres

avg_precision_values = []

for i, book in tqdm(data[:100].iterrows(), desc="Number of iterations"):
    recommended_books = get_recommendations(i, cos_sim, k=5)
    precision_recos = []
    for reco_book in recommended_books:
        
        genre_intersections = get_intersecting_genres(i, data1[data1['Book']==reco_book].index[0])
        precision = genre_intersections/len({x.strip() for x in data1.iloc[i]['Genres'].split(",")})
        
        precision_recos.append(precision)
        
    #print("\n precision of the recos", precision_recos)
    average_precision = sum(precision_recos)/len(precision_recos)
    avg_precision_values.append(average_precision)
        
    

print("mean average precision", round(sum(avg_precision_values)/len(avg_precision_values),2))
    
    
