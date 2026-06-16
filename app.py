import streamlit as st
import pickle
import requests


st.set_page_config(
    page_title="Netflix Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


model = pickle.load(
    open("models/knn_model.pkl", "rb")
)

movies = pickle.load(
    open("models/movies.pkl", "rb")
)

tfidf = pickle.load(
    open("models/tfidf.pkl", "rb")
)

tfidf_matrix = tfidf.transform(
    movies["combined_features"]
)


import requests

API_KEY = "ec38da8f2aad55ef261b8fba61fd1b19"


@st.cache_data
def fetch_poster(movie_name, year):

    try:

        year = str(year) if year else ""

        url = (
            "https://api.themoviedb.org/3/search/movie"
            f"?api_key={API_KEY}"
            f"&query={movie_name}"
            f"&year={year}"
        )

        response = requests.get(url)

        if response.status_code != 200:
            return None

        data = response.json()

        if "results" not in data:
            return None

        if len(data["results"]) == 0:
            return None

        poster_path = data["results"][0].get(
            "poster_path"
        )

        if not poster_path:
            return None

        return (
            "https://image.tmdb.org/t/p/w500"
            + poster_path
        )

    except Exception as e:

        print(
            f"Poster error for {movie_name}: {e}"
        )

        return None


def recommend(movie_name):

    movie_name = movie_name.lower().strip()

    matches = movies[
        movies["movie_name"]
        .astype(str)
        .str.lower()
        .str.contains(movie_name, na=False)
    ]

    if len(matches) == 0:
        return []

    idx = matches.index[0]

    distances, indices = model.kneighbors(
        tfidf_matrix[idx],
        n_neighbors=11
    )

    recommendations = []

    for i in range(1, len(indices[0])):

        movie_idx = indices[0][i]

        movie_info = movies.iloc[movie_idx]

        similarity = round(
            (1 - distances[0][i]) * 100,
            1
        )

        recommendations.append({
            "title": movie_info["movie_name"],
            "rating": movie_info["rating"],
            "genre": movie_info["genre"],
            "director": movie_info["director"],
            "year": movie_info["year"],
            "description": movie_info["description"],
            "similarity": similarity
        })

    return recommendations


st.title("🎬 Netflix Style Movie Recommender")

st.markdown(
    "Discover movies similar to your favorites."
)

movie_list = sorted(
    movies["movie_name"]
    .dropna()
    .unique()
)

selected_movie = st.selectbox(
    "Choose a Movie",
    movie_list
)


if st.button("Get Recommendations"):

    recommendations = recommend(
        selected_movie
    )

    st.subheader(
        f"Movies similar to {selected_movie}"
    )

    cols = st.columns(5)

    for idx, movie in enumerate(recommendations):

        with cols[idx % 5]:

            poster = fetch_poster(
                movie["title"],
                movie["year"]
            )

            if poster:

                st.image(
                    poster,
                    width=220
                )

            else:

                st.markdown(
                    """
                    <div style="
                        width:220px;
                        height:330px;
                        background:#333;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                        border-radius:10px;
                        color:white;
                    ">
                        No Poster
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                f"### {movie['title']}"
            )

            st.write(
                f"⭐ Rating: {movie['rating']}"
            )

            st.write(
                f"🎯 Match: {movie['similarity']}%"
            )

            st.write(
                f"📅 Year: {movie['year']}"
            )

            st.progress(
                movie["similarity"] / 100
            )

            st.caption(
                movie["genre"]
            )

            with st.expander(
                "📖 View Plot"
            ):

                st.write(
                    movie["description"]
                )

            st.markdown("---")