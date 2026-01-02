import streamlit as st
import pickle
import pandas as pd
import requests
import urllib.parse
import re
from difflib import get_close_matches

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Movie & Song Recommender",
    page_icon="🎬",
    layout="wide"
)

# ================= API KEYS =================
TMDB_API_KEY = "60fe564978f3988e59089797fae6648a"
OMDB_API_KEY = "9bcbcb36"

# ================= PLACEHOLDERS =================
MOVIE_PLACEHOLDER = "https://via.placeholder.com/500x750?text=Poster+Unavailable"
SONG_PLACEHOLDER = "https://via.placeholder.com/300x300?text=Album+Art"

# ================= LOAD MODELS =================
movies = pickle.load(open("movies.pkl", "rb"))
movie_similarity = pickle.load(open("similarity.pkl", "rb"))

music_df = pickle.load(open("dfmusic.pkl", "rb"))
tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
song_nn_model = pickle.load(open("song_nn_model.pkl", "rb"))

# ================= NORMALIZATION =================
def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]", "", text)
    return text

# ================= LINKS =================
def spotify_link(song):
    return f"https://open.spotify.com/search/{urllib.parse.quote(song)}"

def imdb_direct_link(imdb_id, movie):
    if imdb_id:
        return f"https://www.imdb.com/title/{imdb_id}/"
    return f"https://www.imdb.com/find?q={urllib.parse.quote(movie)}"

# ================= POSTERS =================
def fetch_movie_poster_and_imdb(movie):
    poster, imdb_id = None, None

    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": movie}
        data = requests.get(url, params=params, timeout=5).json()
        if data.get("results"):
            r = data["results"][0]
            if r.get("poster_path"):
                poster = f"https://image.tmdb.org/t/p/w500{r['poster_path']}"
    except:
        pass

    try:
        url = "http://www.omdbapi.com/"
        params = {"apikey": OMDB_API_KEY, "t": movie}
        data = requests.get(url, params=params, timeout=5).json()
        imdb_id = data.get("imdbID")
        if not poster and data.get("Poster") not in (None, "N/A"):
            poster = data["Poster"]
    except:
        pass

    return poster or MOVIE_PLACEHOLDER, imdb_id

def fetch_song_poster(song):
    try:
        url = "https://itunes.apple.com/search"
        params = {"term": song, "media": "music", "limit": 1}
        data = requests.get(url, params=params, timeout=5).json()
        if data.get("results"):
            return data["results"][0].get("artworkUrl100")
    except:
        pass
    return SONG_PLACEHOLDER

# ================= SMART MATCH =================
def smart_movie_match(user_input):
    user_norm = normalize(user_input)
    movie_map = {normalize(t): t for t in movies["title"]}

    if user_norm in movie_map:
        return movie_map[user_norm]

    for k, v in movie_map.items():
        if user_norm in k:
            return v

    close = get_close_matches(user_norm, movie_map.keys(), n=1, cutoff=0.4)
    return movie_map[close[0]] if close else None

def smart_song_match(user_input):
    user_norm = normalize(user_input)
    song_map = {normalize(s): s for s in music_df["song"]}

    if user_norm in song_map:
        return song_map[user_norm]

    for k, v in song_map.items():
        if user_norm in k:
            return v

    close = get_close_matches(user_norm, song_map.keys(), n=1, cutoff=0.4)
    return song_map[close[0]] if close else None

# ================= RECOMMENDERS =================
def recommend_movie(user_input):
    movie = smart_movie_match(user_input)
    if not movie:
        return []

    idx = movies[movies["title"] == movie].index[0]
    scores = sorted(
        list(enumerate(movie_similarity[idx])),
        reverse=True,
        key=lambda x: x[1]
    )
    return [movies.iloc[i[0]].title for i in scores[1:6]]

def recommend_song(user_input):
    base_song = smart_song_match(user_input)
    if not base_song:
        return []

    idx = music_df[music_df["song"] == base_song].index[0]
    vector = tfidf.transform([music_df.iloc[idx]["text"]])
    _, indices = song_nn_model.kneighbors(vector, n_neighbors=15)

    seen, recs = set(), []
    for i in indices[0]:
        s = music_df.iloc[i].song
        if s != base_song and s not in seen:
            seen.add(s)
            recs.append(s)
        if len(recs) == 5:
            break
    return recs

# ================= UI =================
st.title("🎬🎵 Movie & Song Recommendation System")

tab1, tab2 = st.tabs(["🎬 Movies", "🎵 Songs"])

# ---------- MOVIES ----------
with tab1:
    choice = st.radio(
        "How would you like to choose a movie?",
        ["Select from list", "Type movie name"],
        horizontal=True
    )

    with st.form("movie_form"):
        movie_input = (
            st.selectbox("Select a movie", movies["title"].values)
            if choice == "Select from list"
            else st.text_input("Type a movie name")
        )
        submit = st.form_submit_button("✨ Recommend Movies")

    if submit:
        recs = recommend_movie(movie_input)
        cols = st.columns(5)
        for col, movie in zip(cols, recs):
            poster, imdb_id = fetch_movie_poster_and_imdb(movie)
            with col:
                st.markdown(
                    f"""
                    <a href="{imdb_direct_link(imdb_id, movie)}" target="_blank">
                        <img src="{poster}" style="width:100%;border-radius:12px"/>
                        <p style="text-align:center;font-weight:600">🎬 {movie}</p>
                    </a>
                    """,
                    unsafe_allow_html=True
                )

# ---------- SONGS ----------
with tab2:
    choice = st.radio(
        "How would you like to choose a song?",
        ["Select from list", "Type song name"],
        horizontal=True
    )

    with st.form("song_form"):
        song_input = (
            st.selectbox("Select a song", music_df["song"].values)
            if choice == "Select from list"
            else st.text_input("Type a song name")
        )
        submit = st.form_submit_button("✨ Recommend Songs")

    if submit:
        recs = recommend_song(song_input)
        cols = st.columns(5)
        for col, song in zip(cols, recs):
            with col:
                st.markdown(
                    f"""
                    <a href="{spotify_link(song)}" target="_blank">
                        <img src="{fetch_song_poster(song)}" style="width:100%;border-radius:12px"/>
                        <p style="text-align:center;font-weight:600">🎵 {song}</p>
                    </a>
                    """,
                    unsafe_allow_html=True
                )

st.markdown("---")
st.markdown("<center>Project Created By Harsh Gangapurkar</center>", unsafe_allow_html=True)


