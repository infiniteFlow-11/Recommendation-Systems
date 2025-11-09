import streamlit as st
import joblib
import requests
import pandas as pd

# --- TMDB API Configuration ---
# IMPORTANT: Store your API key in Streamlit secrets
# For local development, you can set it here, but DO NOT commit it.
# API_KEY = "YOUR_TMDB_API_KEY_HERE" 
# Use this when deploying:
try:
    API_KEY = st.secrets["TMDB_API_KEY"]
except FileNotFoundError:
    # Fallback for local testing if secrets.toml isn't set up
    API_KEY = "de457bc45dcf2a29428e13a3f6fa91db" # Replace with your key for local run

def fetch_movie_details(movie_title):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_title}"
    try:
        response = requests.get(search_url)
        response.raise_for_status() 
        data = response.json()
        
        if data['results']:
            movie_data = data['results'][0]
            poster_path = movie_data.get('poster_path')
            overview = movie_data.get('overview', 'No overview available.')
            
            if poster_path:
                full_poster_url = f"https://image.tmdb.org/t/p/w500/{poster_path}"
            else:
                full_poster_url = None # Handle missing posters
                
            return full_poster_url, overview
        else:
            return None, "Details not found."
            
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return None, "Error fetching details."

def get_recommendations(movie_title, similarity_matrix, movie_titles_index, num_recommendations=10):
    try:
        movie_index = movie_titles_index.get_loc(movie_title)
    except KeyError:
        st.error(f"Movie '{movie_title}' not in the dataset.")
        return []
        
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    sorted_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    
    top_movie_indices = [i[0] for i in sorted_scores[1:num_recommendations+1]]
    recommended_movies = [movie_titles_index[i] for i in top_movie_indices]
    
    return recommended_movies

# --- Load Model Artifacts ---
@st.cache_data
def load_model():
    try:
        similarity_matrix = joblib.load('movie_similarity.pkl')
        movie_titles = joblib.load('movie_titles.pkl')
        return similarity_matrix, movie_titles
    except FileNotFoundError:
        st.error("Model files not found. Please run 'build_model.py' first.")
        return None, None

similarity_matrix, movie_titles = load_model()

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title('🎬 Movie Recommendation Engine')
st.markdown("Select a movie you like, and we'll suggest 10 similar ones based on user ratings.")

if similarity_matrix is not None and movie_titles is not None:
    selected_movie = st.selectbox(
        'Type or select a movie from the dropdown:',
        movie_titles
    )

    if st.button('Get Recommendations', type="primary"):
        with st.spinner('Finding similar movies...'):
            recommendations = get_recommendations(selected_movie, similarity_matrix, movie_titles)
            
            if recommendations:
                st.subheader(f"Because you liked '{selected_movie}':")
                
                # Display recommendations in 2 rows of 5
                num_cols = 5
                rows = [st.columns(num_cols) for _ in range(len(recommendations) // num_cols + 1)]
                
                cols = [col for row in rows for col in row]
                
                for i, movie_title in enumerate(recommendations):
                    with cols[i]:
                        poster_url, overview = fetch_movie_details(movie_title)
                        
                        if poster_url:
                            st.image(poster_url, caption=movie_title)
                        else:
                            st.markdown(f"**{movie_title}**")
                            st.caption("No poster available")
                        
                        with st.expander("See details"):
                            st.write(overview)
            else:
                st.write("No recommendations found.")

else:
    st.warning("Model is not loaded. Please ensure 'movie_similarity.pkl' and 'movie_titles.pkl' exist.")