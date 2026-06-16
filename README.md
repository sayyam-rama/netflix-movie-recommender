# 🎬 Netflix-Style Movie Recommender

A machine learning-powered movie recommendation system inspired by Netflix. The application recommends movies based on genres, directors, actors, and plot descriptions using content-based filtering techniques.

## Features

* 🎥 Personalized movie recommendations
* ⭐ Similarity match scores
* 🎭 Movie metadata (genre, director, rating, year)
* 🖼️ Movie posters using TMDb API
* 📖 Plot descriptions for recommendations
* 🌙 Interactive Streamlit web interface
* 🤖 TF-IDF + K-Nearest Neighbors recommendation engine

## Tech Stack

* Python
* Pandas
* Scikit-Learn
* TF-IDF Vectorization
* K-Nearest Neighbors (KNN)
* Streamlit
* TMDb API

## Dataset

IMDb Movies Dataset containing 50,000+ movies. The dataset was cleaned, filtered, and processed to build a high-quality recommendation engine.

## How It Works

1. Movie metadata is combined into a feature set.
2. TF-IDF converts text data into numerical vectors.
3. KNN identifies movies with similar feature vectors.
4. Recommendations are ranked and displayed in a Streamlit web app.

## Installation

```bash
git clone https://github.com/sayyam-rama/netflix-movie-recommender.git
cd netflix-movie-recommender

pip install -r requirements.txt

streamlit run app.py
```
## Author

**Sayyam Rama**
