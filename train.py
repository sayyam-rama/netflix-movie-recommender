import pandas as pd
import os

folder = "data"

all_dfs = []

for file in os.listdir(folder):

    if file.endswith(".csv"):

        print(f"Loading {file}")

        df = pd.read_csv(
            os.path.join(folder, file)
        )

        all_dfs.append(df)

movies = pd.concat(
    all_dfs,
    ignore_index=True
)

print("Before removing duplicates:")
print(movies.shape)

movies = movies.drop_duplicates(
    subset=["movie_id"]
)

print("After removing duplicates:")
print(movies.shape)
print(movies.head())

movies["combined_features"] = (
    (movies["genre"] + " ") * 6 +
    (movies["director"] + " ") * 5 +
    (movies["star"] + " ") * 4
)
movies = movies.dropna(subset=["combined_features"])
movies["combined_features"] = movies["combined_features"].str.strip()

movies["rating"] = pd.to_numeric(
    movies["rating"],
    errors="coerce"
)

movies = movies[
    movies["rating"] >= 5
]

movies["votes"] = pd.to_numeric(
    movies["votes"],
    errors="coerce"
)

movies = movies[
    movies["votes"] >= 5000
]

print("NaN values:", movies["combined_features"].isna().sum())
print("Empty rows:", (movies["combined_features"] == "").sum())
movies = movies.reset_index(drop=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

tfidf = TfidfVectorizer(
    stop_words="english",
    max_features=10000
)

tfidf_matrix = tfidf.fit_transform(
    movies["combined_features"]
)

model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=11
)

model.fit(tfidf_matrix)

print("Movies Loaded:", len(movies))