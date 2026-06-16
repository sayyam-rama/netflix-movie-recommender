import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle

movies = pd.read_csv("data/all_movies.csv")

movies["genre"] = movies["genre"].fillna("").astype(str)
movies["description"] = movies["description"].fillna("").astype(str)
movies["director"] = movies["director"].fillna("").astype(str)
movies["star"] = movies["star"].fillna("").astype(str)

movies["rating"] = pd.to_numeric(
    movies["rating"],
    errors="coerce"
)

movies["votes"] = pd.to_numeric(
    movies["votes"],
    errors="coerce"
)

movies = movies[
    (movies["rating"] >= 6.0) &
    (movies["votes"] >= 1000)
]

movies = movies.reset_index(drop=True)

print(f"Movies Loaded: {len(movies)}")

movies["combined_features"] = (
    (movies["genre"] + " ") * 5 +
    (movies["director"] + " ") * 4 +
    (movies["star"] + " ") * 3 +
    movies["description"]
)

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = tfidf.fit_transform(
    movies["combined_features"]
)

print("TF-IDF Matrix Shape:")
print(tfidf_matrix.shape)

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=11
)

model.fit(tfidf_matrix)

print("Recommendation model ready!")

pickle.dump(
    model,
    open("models/knn_model.pkl", "wb")
)

pickle.dump(
    tfidf,
    open("models/tfidf.pkl", "wb")
)

pickle.dump(
    movies,
    open("models/movies.pkl", "wb")
)

print("Models saved!")