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

movies.to_csv(
    "data/all_movies.csv",
    index=False
)

print("all_movies.csv created successfully!")