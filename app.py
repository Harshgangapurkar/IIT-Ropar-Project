import streamlit as st
import pickle
import pandas as pd
import requests
import urllib.parse

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Movie & Song Recommender",
    page_icon="🎬",
    layout="wide"
)

# ================= API KEYS (ONE-TIME SETUP) =================
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

# ================= LINK HELPERS =================
def spotify_link(song):
    return f"https://open.spotify.com/search/{urllib.parse.quote(song)}"

def imdb_direct_link(imdb_id, movie):
    if imdb_id:
        return f"https://www.imdb.com/title/{imdb_id}/"
    return f"https://www.imdb.com/find?q={urllib.parse.quote(movie)}"

# ================= POSTER + IMDb ID =================
def fetch_movie_poster_and_imdb(movie):
    poster = None
    imdb_id = None

    # TMDB → Poster
    try:
        url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": movie}
        data = requests.get(url, params=params, timeout=5).json()
        for r in data.get("results", []):
            if r.get("poster_path"):
                poster = f"https://image.tmdb.org/t/p/w500{r['poster_path']}"
                break
    except:
        pass

    # OMDb → IMDb ID + Poster fallback
    try:
        url = "http://www.omdbapi.com/"
        params = {"apikey": OMDB_API_KEY, "t": movie}
        data = requests.get(url, params=params, timeout=5).json()

        if data.get("imdbID"):
            imdb_id = data["imdbID"]

        if not poster and data.get("Poster") and data["Poster"] != "N/A":
            poster = data["Poster"]
    except:
        pass

    if not poster:
        poster = MOVIE_PLACEHOLDER

    return poster, imdb_id

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

# ================= RECOMMENDERS =================
def recommend_movie(movie):
    idx = movies[movies["title"].str.lower() == movie.lower()].index
    if len(idx) == 0:
        return []

    idx = idx[0]
    distances = sorted(
        list(enumerate(movie_similarity[idx])),
        reverse=True,
        key=lambda x: x[1]
    )
    return [movies.iloc[i[0]].title for i in distances[1:6]]

def recommend_song(song):
    song = song.lower().strip()
    idx = music_df[music_df["song"].str.lower() == song].index
    if len(idx) == 0:
        return []

    idx = idx[0]
    vector = tfidf.transform([music_df.iloc[idx]["text"]])
    _, indices = song_nn_model.kneighbors(vector, n_neighbors=15)

    seen = set()
    recs = []

    for i in indices[0]:
        s = music_df.iloc[i].song
        if s.lower() != song and s not in seen:
            seen.add(s)
            recs.append(s)
        if len(recs) == 5:
            break

    return recs

# ================= UI =================
st.title("🎬🎵 Movie & Song Recommendation System")

tab1, tab2 = st.tabs(["🎬 Movies", "🎵 Songs"])

# ================= MOVIES =================
with tab1:
    st.subheader("Movie Recommendation")

    choice = st.radio(
        "How would you like to choose a movie?",
        ["Select from list", "Type movie name"],
        horizontal=True
    )

    with st.form("movie_form"):
        if choice == "Select from list":
            movie_input = st.selectbox("Select a movie", movies["title"].values)
        else:
            movie_input = st.text_input("Type a movie name")

        submit = st.form_submit_button("✨ Recommend Movies")

    if submit:
        recs = recommend_movie(movie_input)
        cols = st.columns(5)

        for col, movie in zip(cols, recs):
            poster, imdb_id = fetch_movie_poster_and_imdb(movie)
            imdb_url = imdb_direct_link(imdb_id, movie)

            with col:
                st.markdown(
                    f"""
                    <a href="{imdb_url}" target="_blank">
                        <img src="{poster}" style="width:100%;border-radius:12px"/>
                        <p style="text-align:center;font-weight:600;color:white">
                            🎬 {movie}
                        </p>
                    </a>
                    """,
                    unsafe_allow_html=True
                )

# ================= SONGS =================
with tab2:
    st.subheader("Song Recommendation")

    choice = st.radio(
        "How would you like to choose a song?",
        ["Select from list", "Type song name"],
        horizontal=True
    )

    with st.form("song_form"):
        if choice == "Select from list":
            song_input = st.selectbox("Select a song", music_df["song"].values)
        else:
            song_input = st.text_input("Type a song name")

        submit = st.form_submit_button("✨ Recommend Songs")

    if submit:
        recs = recommend_song(song_input)
        cols = st.columns(5)

        for col, song in zip(cols, recs):
            poster = fetch_song_poster(song)
            spotify_url = spotify_link(song)

            with col:
                st.markdown(
                    f"""
                    <a href="{spotify_url}" target="_blank">
                        <img src="{poster}" style="width:100%;border-radius:12px"/>
                        <p style="text-align:center;font-weight:600;color:white">
                            🎵 {song}
                        </p>
                    </a>
                    """,
                    unsafe_allow_html=True
                )

st.markdown("---")
st.markdown(
    "<center>Project Created By Harsh Gangapurkar</center>",
    unsafe_allow_html=True
)


