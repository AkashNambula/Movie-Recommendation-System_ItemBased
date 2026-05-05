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
user_item_matrix = None
user_similarity_df = None
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
    global user_item_matrix, user_similarity_df, movies_df, global_popularity

    print("⚙️ Preparing data...\n")

    # Load movies
    m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
    movies_df = pd.read_csv(f'{DATA_DIR}/u.item', sep='|', names=m_cols, usecols=range(5), encoding='latin-1')

    # Load ratings
    r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings_df = pd.read_csv(f'{DATA_DIR}/u.data', sep='\t', names=r_cols, encoding='latin-1')

    # Popular movies
    movie_counts = ratings_df.groupby('movie_id').size().reset_index(name='count')
    top_popular_ids = movie_counts.sort_values(by='count', ascending=False).head(5)['movie_id']
    global_popularity = movies_df[movies_df['movie_id'].isin(top_popular_ids)]['title'].tolist()

    # User-item matrix
    user_item_matrix = ratings_df.pivot(index='user_id', columns='movie_id', values='rating')

    # Fill missing values
    matrix_filled = user_item_matrix.fillna(0)

    # Similarity
    user_similarity = cosine_similarity(matrix_filled)
    user_similarity_df = pd.DataFrame(user_similarity,
                                     index=user_item_matrix.index,
                                     columns=user_item_matrix.index)

    print("✅ Data prepared successfully!\n")


# 🔹 Recommendation logic
def get_recommendations(user_id):
    if user_id not in user_item_matrix.index:
        print("❌ User not found! Enter between 1–943")
        return

    user_ratings = user_item_matrix.loc[user_id]
    unseen_movies = user_ratings[user_ratings.isna()].index

    similar_users = user_similarity_df[user_id].drop(user_id)
    top_similar_users = similar_users.sort_values(ascending=False).head(50)

    recommendations = []

    for movie_id in unseen_movies:
        ratings_for_movie = user_item_matrix.loc[top_similar_users.index, movie_id].dropna()

        if len(ratings_for_movie) > 0:
            similarities = top_similar_users.loc[ratings_for_movie.index]

            predicted_rating = (
                np.dot(similarities, ratings_for_movie) / similarities.sum()
                if similarities.sum() > 0 else 0
            )

            title = movies_df[movies_df['movie_id'] == movie_id]['title'].values[0]

            recommendations.append((title, predicted_rating))

    # Sort
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Output
    print("\n🎬 Top 5 Recommended Movies:\n")
    for i, movie in enumerate(recommendations[:5], start=1):
        print(f"{i}. {movie[0]}")

    print("\n🔥 Popular Movies:\n")
    for movie in global_popularity:
        print(f"- {movie}")


# 🔹 Main program
if __name__ == "__main__":
    download_and_extract_data()
    prepare_data()

    while True:
        try:
            user_id = int(input("\n👉 Enter User ID (1–943) or 0 to exit: "))

            if user_id == 0:
                print("👋 Exiting program...")
                break

            get_recommendations(user_id)

        except ValueError:
            print("⚠️ Please enter a valid number!")