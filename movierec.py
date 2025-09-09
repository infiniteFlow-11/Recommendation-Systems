# data_prep.py
import pandas as pd

# Load the datasets
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Merge them on 'movieId'
df = pd.merge(ratings, movies, on='movieId')

# Create the user-item interaction matrix
# We are using title as index, which assumes unique titles.
# In a production system, you'd handle duplicate titles.
movie_matrix = df.pivot_table(index='title', columns='userId', values='rating')

# Fill NaN values with 0. This is crucial as NaN would break calculations.
# It implies that if a user hasn't rated a movie, the rating is 0.
movie_matrix.fillna(0, inplace=True)

print("User-Item Matrix created successfully.")
print(movie_matrix.head())

# Part of the main app script, but showing the logic here
from sklearn.metrics.pairwise import cosine_similarity

# Compute the cosine similarity matrix
# The matrix will have movies as both rows and columns.
# The value at (row_i, col_j) is the similarity between movie_i and movie_j.
similarity_matrix = cosine_similarity(movie_matrix)

# We can convert this back to a DataFrame for easier inspection
similarity_df = pd.DataFrame(similarity_matrix, index=movie_matrix.index, columns=movie_matrix.index)

print("Similarity Matrix:")
print(similarity_df.head())

# The recommendation function
def recommend(movie_title, movie_matrix, similarity_df, n_recommendations=5):
    # Get the similarity scores for the movie
    movie_similarities = similarity_df[movie_title]
    
    # Sort the movies based on similarity scores in descending order
    similar_movies = movie_similarities.sort_values(ascending=False)
    
    # Get the top N movies (excluding the movie itself, which will have a similarity of 1.0)
    top_movies = similar_movies.iloc[1:n_recommendations+1].index
    
    return top_movies.tolist()

# Example usage:
# recommendations = recommend('Toy Story (1995)', movie_matrix, similarity_df)
# print(recommendations)
# Expected Output: ['Toy Story 2 (1999)', 'Jurassic Park (1993)', 'Forrest Gump (1994)', ...]

# TMDB integration function
import requests

def fetch_details(movie_title):
    # NOTE: Replace 'YOUR_API_KEY' with your actual TMDB API key
    API_KEY = 'YOUR_API_KEY'
    
    # Search for the movie to get its ID
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
    response = requests.get(search_url)
    data = response.json()
    
    if data['results']:
        movie_id = data['results'][0]['id']
        poster_path = data['results'][0].get('poster_path')
        overview = data['results'][0].get('overview')
        
        if poster_path:
            poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            return poster_url, overview
    
    # Return a placeholder if no poster is found
    return "https://via.placeholder.com/500x750.png?text=No+Image+Found", "No overview available."

# create_pickle.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --- Data Loading and Preprocessing ---
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
df = pd.merge(ratings, movies, on='movieId')
movie_matrix = df.pivot_table(index='title', columns='userId', values='rating')
movie_matrix.fillna(0, inplace=True)

# --- Similarity Computation ---
similarity_matrix = cosine_similarity(movie_matrix)
similarity_df = pd.DataFrame(similarity_matrix, index=movie_matrix.index, columns=movie_matrix.index)

# --- Save to Pickle Files ---
with open('movie_list.pkl', 'wb') as f:
    pickle.dump(movie_matrix.index.tolist(), f)

with open('similarity.pkl', 'wb') as f:
    pickle.dump(similarity_df, f)

print("Pickle files created successfully.")

# app.py
import streamlit as st
import pandas as pd
import pickle
import requests

# --- Helper Functions ---
def fetch_details(movie_title):
    """Fetches movie poster and overview from TMDB API."""
    API_KEY = 'YOUR_API_KEY' # Replace with your TMDB API key
    try:
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
        response = requests.get(search_url)
        response.raise_for_status() # Raises an HTTPError for bad responses
        data = response.json()
        
        if data['results']:
            result = data['results'][0]
            poster_path = result.get('poster_path')
            overview = result.get('overview')
            
            if poster_path:
                poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
                return poster_url, overview
    except requests.exceptions.RequestException as e:
        st.error(f"API request failed: {e}")
    
    return "https://via.placeholder.com/500x750.png?text=Image+Not+Found", "No overview available."

def recommend(movie_title, similarity_df, movie_list, n_recommendations=5):
    """Recommends movies based on cosine similarity."""
    if movie_title not in similarity_df.columns:
        return []
    
    movie_similarities = similarity_df[movie_title]
    similar_movies = movie_similarities.sort_values(ascending=False)
    top_movies = similar_movies.iloc[1:n_recommendations+1].index
    return top_movies.tolist()

# --- Load Data ---
# Use st.cache_data to load data only once
@st.cache_data
def load_data():
    with open('movie_list.pkl', 'rb') as f:
        movie_list = pickle.load(f)
    with open('similarity.pkl', 'rb') as f:
        similarity_df = pickle.load(f)
    return movie_list, similarity_df

movie_list, similarity_df = load_data()

# --- Streamlit UI ---
st.title('🎬 Movie Recommendation System')
st.markdown("Based on Item-Item Collaborative Filtering")

selected_movie = st.selectbox(
    "Select a movie you like, and we'll recommend similar ones:",
    movie_list
)

if st.button('Get Recommendations'):
    with st.spinner('Finding similar movies...'):
        recommendations = recommend(selected_movie, similarity_df, movie_list, n_recommendations=5)
        
        if recommendations:
            st.subheader(f"Recommendations for '{selected_movie}':")
            
            # Display recommendations in columns
            cols = st.columns(5)
            for i, movie in enumerate(recommendations):
                with cols[i]:
                    poster_url, overview = fetch_details(movie)
                    st.image(poster_url)
                    st.markdown(f"**{movie}**")
        else:
            st.warning("Could not find recommendations for this movie.")