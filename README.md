# Movie Recommender System

A web-based movie recommendation engine that uses User-Based Collaborative Filtering to provide personalized movie suggestions and a global popularity baseline.

## Features

- **Personalized Recommendations:** Predicts movie ratings based on a user's similarity to other users using cosine similarity.
- **Explainable AI:** Provides clear reasoning for its recommendations (e.g., average rating by similar users).
- **Popularity Baseline:** Displays the top globally popular movies alongside personalized picks to show general trends.
- **Auto Data Ingestion:** Automatically downloads and extracts the MovieLens 100K dataset on server startup if it is not present.
- **Clean Interface:** A responsive, side-by-side frontend built with HTML, CSS, and Vanilla JavaScript.

## Tech Stack

- **Backend:** Python 3.x, FastAPI, Uvicorn
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Frontend:** HTML5, CSS3, Vanilla JS
- **Dataset:** MovieLens 100K

## Installation

1. Clone this repository or download the source code to your local machine.
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the FastAPI server:

   ```bash
   uvicorn main:app --reload
   ```
   *Note: On the very first run, the server will download and extract the ~5MB MovieLens dataset automatically. Please be patient while it prepares the data.*

2. Open your web browser and navigate to:

   ```text
   http://127.0.0.1:8000
   ```

3. Enter a User ID (between 1 and 943) in the input field and click "Get Recommendations" to view your personalized results.

## API Endpoints

- `GET /` : Serves the frontend web application.
- `GET /api/recommendations/{user_id}` : Returns a JSON object containing `personalized` recommendations and a `popularity_baseline`.
