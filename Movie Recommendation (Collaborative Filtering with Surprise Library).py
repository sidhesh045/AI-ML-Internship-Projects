import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------------
# Step 1: Load MovieLens Dataset
# -----------------------------
ratings = pd.read_csv(
    "ml-100k/u.data",
    sep="\t", names=["user_id", "movie_id", "rating", "timestamp"]
)

movies = pd.read_csv(
    "ml-100k/u.item",
    sep="|", encoding="latin-1",
    names=["movie_id", "title", "release_date", "video_release_date", "imdb_url",
           "unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
           "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
           "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
)

# -----------------------------
# Step 2: Merge Ratings & Movies
# -----------------------------
df = pd.merge(ratings, movies[["movie_id", "title"]], on="movie_id")

# Create user-movie rating matrix
user_movie_matrix = df.pivot_table(index="user_id", columns="title", values="rating")

# -----------------------------
# Step 3: User Similarity
# -----------------------------
user_similarity = cosine_similarity(user_movie_matrix.fillna(0))
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

# -----------------------------
# Step 4: Recommendation Function
# -----------------------------
def recommend_movies(user_id, n=5):
    # Find similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:]
    top_user = similar_users.index[0]

    # Movies the target user has already seen
    user_movies = user_movie_matrix.loc[user_id].dropna().index

    # Movies the similar user has rated highly
    similar_user_ratings = user_movie_matrix.loc[top_user].dropna()
    recommended_movies = similar_user_ratings[~similar_user_ratings.index.isin(user_movies)].sort_values(ascending=False)

    return recommended_movies.head(n).index.tolist()

# -----------------------------
# Example Usage
# -----------------------------
print("Top 5 movie recommendations for User 1:")
print(recommend_movies(1, 5))

print("\nTop 5 movie recommendations for User 50:")
print(recommend_movies(50, 5))
