🎬🎵 Movie & Song Recommendation System

A content-based recommendation system that suggests movies and songs similar to user input using text similarity techniques, with real-world integrations to IMDb and Spotify.

deployed streamlit app: https://harsh-recommendation.streamlit.app

📌 Project Overview

In modern digital platforms, users are overwhelmed with choices. Selecting what movie to watch or what song to listen to becomes difficult due to the massive volume of content.

This project addresses that challenge by building a content-based recommender system that:

Recommends movies and songs based on similarity

Works without user history or ratings (cold-start friendly)

Supports partial and flexible user input

Provides explainable recommendations

Redirects users directly to IMDb (movies) and Spotify (songs)

🎯 Key Features
🎬 Movie Recommendation

Content-based filtering using TF-IDF + Cosine Similarity

Handles partial inputs (e.g. spider man, Spider-Man, Spiderman)

Displays movie posters

Clicking a recommendation opens the IMDb page of the movie

🎵 Song Recommendation

Text-based recommendation using TF-IDF + Nearest Neighbors

Duplicate-free recommendations

Displays album artwork

Clicking a recommendation opens the song on Spotify

🧠 Explainability (Industry Feature)

Each recommendation is supported by a similarity score, explaining why it was recommended.
This improves transparency and user trust.

🧠 Machine Learning Techniques Used
Component	Technique
Movie Recommendation	TF-IDF Vectorization + Cosine Similarity
Song Recommendation	TF-IDF Vectorization + Nearest Neighbors
Learning Type	Unsupervised Learning
Recommendation Type	Content-Based Filtering

⚠️ No supervised learning or deep learning models are used, making the system lightweight, fast, and explainable.

🏗️ System Architecture
User Input
   ↓
Text Normalization
   ↓
Feature Vector Representation (TF-IDF)
   ↓
Similarity Computation
   ↓
Top-N Recommendation
   ↓
IMDb / Spotify Redirection

📊 Evaluation Strategy

Since recommendation systems typically lack labeled ground truth, traditional accuracy metrics are not applicable.

Evaluation was performed using:

Self-similarity validation (sanity check)

Average Top-K similarity score

Diversity analysis

Qualitative inspection of recommendations

These methods are standard for evaluating content-based recommender systems.

⚖️ Ethical Considerations & Responsible AI

No personal user data is collected

No user profiling or tracking

Recommendations are advisory, not deterministic

Dataset limitations and bias are acknowledged

System is designed for educational and demonstrative purposes

🛠️ Tech Stack

Python

Streamlit (Web Interface)

Scikit-learn

Pandas & NumPy

TMDB API (Movie posters)

Spotify Web Redirect

IMDb Integration

🚀 How to Run the Project
pip install -r requirements.txt
streamlit run app.py


Make sure the following files are present in the project directory:

movies.pkl

similarity.pkl

dfmusic.pkl

tfidf_vectorizer.pkl

song_nn_model.pkl

🔮 Future Improvements

Hybrid recommendation system (content + collaborative filtering)

User profiles and personalization

Mood-based and genre-based filtering

Transformer-based embeddings

LLM-powered conversational recommendations

👨‍💻 Author

Harsh Gangapurkar
B.Tech – Artificial Intelligence & Data Science

🏁 Final Note

This project demonstrates that effective recommendation systems can be built using classical ML and NLP techniques, without relying on heavy deep learning models, while still providing real-world usability and explainability.
