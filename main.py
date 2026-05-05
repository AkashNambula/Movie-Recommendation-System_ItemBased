import os
import zipfile
import urllib.request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DATA_DIR = "ml-100k"
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ZIP_FILE = "ml-100k.zip"

# Global variables
movie_user_matrix = None
item_similarity_df = None
movies_df = None
global_popularity = None


# 🔹 Download dataset
def download_and_extract_data():
    if not os.path.exists(DATA_DIR):
        print("📥 Downloading dataset...")
        urllib.request.urlretrieve(DATA_URL, ZIP_FILE)

        print("📦 Extracting dataset...")
        with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
            zip_ref.extractall(".")

        print("✅ Dataset ready!\n")


# 🔹 Prepare data
def prepare_data():
    global movie_user_matrix, item_similarity_df, movies_df, global_popularity

    print("⚙️ Preparing data...\n")

    # Load movies
    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
    movies_df = pd.read_csv(f'{DATA_DIR}/u.item', sep='|', names=m_cols, usecols=range(5), encoding='latin-1')

    # Convert titles to lowercase (for better matching)
    movies_df['title'] = movies_df['title'].str.lower()

    # Load ratings
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(f'{DATA_DIR}/u.data', sep='\t', names=r_cols, encoding='latin-1')

    # 🔥 Popular movies
    movie_counts = ratings_df.groupby('movie_id').size().reset_index(name='count')
    top_popular_ids = movie_counts.sort_values(by='count', ascending=False).head(5)['movie_id']
    global_popularity = movies_df[movies_df['movie_id'].isin(top_popular_ids)]['title'].tolist()

    # 🔹 ITEM-USER MATRIX (IMPORTANT CHANGE)
    movie_user_matrix = ratings_df.pivot(index='movie_id', columns='user_id', values='rating')

    # Fill missing values
    matrix_filled = movie_user_matrix.fillna(0)

    # 🔹 ITEM SIMILARITY
    item_similarity = cosine_similarity(matrix_filled)

    item_similarity_df = pd.DataFrame(
        item_similarity,
        index=movie_user_matrix.index,
        columns=movie_user_matrix.index
    )

    print("✅ Data prepared successfully!\n")


# 🔹 Recommendation logic (ITEM-BASED)
def get_recommendations(movie_name):
    movie_name = movie_name.lower()

    if movie_name not in movies_df['title'].values:
        print("❌ Movie not found! Try another name.")
        return

    # Get movie_id
    movie_id = movies_df[movies_df['title'] == movie_name]['movie_id'].values[0]

    # Similar movies
    similar_movies = item_similarity_df[movie_id].sort_values(ascending=False)

    print("\n🎬 Top 5 Similar Movies:\n")

    count = 0
    for mid in similar_movies.index[1:]:  # skip same movie
        title = movies_df[movies_df['movie_id'] == mid]['title'].values[0]
        print(f"{count+1}. {title}")
        count += 1
        if count == 5:
            break

    print("\n🔥 Popular Movies:\n")
    for movie in global_popularity:
        print(f"- {movie}")


# 🔹 Main program
if __name__ == "__main__":
    download_and_extract_data()
    prepare_data()

    while True:
        movie_name = input("\n👉 Enter Movie Name (or 'exit'): ")

        if movie_name.lower() == 'exit':
            print("👋 Exiting program...")
            break

        get_recommendations(movie_name)