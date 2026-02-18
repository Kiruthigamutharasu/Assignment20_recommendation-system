# ===============================
# Movie Recommendation System
# ===============================

import streamlit as st
import os
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# Create pickle files automatically if missing
# -------------------------------------------------

if not os.path.exists("movies.pkl"):

    df = pd.read_csv("data/tmdb_5000_movies.csv")
    df = df[['title', 'overview']]
    df['overview'] = df['overview'].fillna('')

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])

    similarity = cosine_similarity(tfidf_matrix)

    pickle.dump(df, open("movies.pkl", "wb"))
    pickle.dump(similarity, open("similarity.pkl", "wb"))


# -------------------------------------------------
# Load pickle files
# -------------------------------------------------

df = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))


# -------------------------------------------------
# Recommendation function
# -------------------------------------------------

def recommend(movie):
    idx = df[df['title'] == movie].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return [df.iloc[i[0]].title for i in scores]


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------

st.title("🎬 Movie Recommendation System")

selected_movie = st.selectbox(
    "Select a Movie",
    df['title'].values
)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.write(movie)
