import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
data = pd.read_csv("/kaggle/input/best-books-10k-multi-genre-data/goodreads_data.csv").drop("Unnamed: 0", axis=1)

# Preprocess the data
data = data.dropna()
data = data.drop_duplicates()

# Feature Extraction
tfidf = TfidfVectorizer(stop_words='english')
description_tfidf = tfidf.fit_transform(data["Description"])
genres_tfidf = tfidf.fit_transform(data["Genres"])
ratings = data["Avg_Rating"].apply(lambda x: (x - data["Avg_Rating"].min())/(data["Avg_Rating"].max()-data["Avg_Rating"].min())).values.reshape(-1, 1)

# Similarity Calculation
book_vectors = pd.DataFrame(description_tfidf.toarray())
book_vectors = pd.concat([book_vectors, pd.DataFrame(genres_tfidf.toarray())], axis=1)
book_vectors = pd.concat([book_vectors, pd.DataFrame(ratings, columns=["ratings"])], axis=1)
similarity_matrix = cosine_similarity(book_vectors)

# Recommendation Generation
def get_recommendations(book_id, k=5):
    # Get the top k similar books to the given book_id
    similar_books = list(enumerate(similarity_matrix[book_id]))
    similar_books = sorted(similar_books, key=lambda x: x[1], reverse=True)
    similar_books = similar_books[1:k+1]
    recommended_books = [data.iloc[i[0]]["Book"] for i in similar_books]
    return recommended_books

# Example Usage
book_id = 18
recommended_books = get_recommendations(book_id)
print(f"Recommended books for {data.iloc[book_id]['Book']}: {recommended_books}")

#Calculating Average Precision using ratings

precision_values = []

for i, book in data[:1000].iterrows():
    recommended_books = get_recommendations(i, k=5)
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

for i, book in data[:100].iterrows():
    recommended_books = get_recommendations(i, k=5)
    precision_recos = []
    for reco_book in recommended_books:
        
        genre_intersections = get_intersecting_genres(i, data1[data1['Book']==reco_book].index[0])
        precision = genre_intersections/len({x.strip() for x in data1.iloc[i]['Genres'].split(",")})
        
        #print("inter", genre_intersections)
        #print(f"precision of {reco_book} recommendation is", precision)
        precision_recos.append(precision)
        
    #print("\n precision of the recos", precision_recos)
    average_precision = sum(precision_recos)/len(precision_recos)
    avg_precision_values.append(average_precision)
        
    

print("mean average precision", round(sum(avg_precision_values)/len(avg_precision_values),2))




