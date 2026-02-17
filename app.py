# ===============================
# PART 4 — Streamlit UI
# ===============================

import streamlit as st
import pickle

# Load saved files
df = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

def recommend(movie):
    idx = df[df['title'] == movie].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    return [df.iloc[i[0]].title for i in scores]


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
