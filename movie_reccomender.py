import pandas as pd
import numpy as np
from numpy.linalg import norm
import ipywidgets as widgets
from IPython.display import display

# Load data
data = pd.read_csv("data/movies.csv")
data = data.loc[data["votes"] > 5000, ["name", "genre", "year", "popularityScore", "gross", "runtime"]]

# Create new columns and normalize data
data["popularityScore"] = data["score"] * data["votes"]
del data["score"], data["votes"]

# Preprocess categorical data (e.g., director, writer, star)
data["director"] = data["director"].apply(lambda x: (x, (data[data["director"] == x]["director"].count()) // 10))
# Repeat similar for 'writer', 'star'

# Transform data into usable vectors
countrys = ["USCan", "GB", "Oceania", "EU", "East Asia", "Other"]
data["country"] = data["country"].apply(lambda country: [1 if country == c else 0 for c in countrys])

genres = ["Comedy", "Action", "Drama", "Crime", "Biography", "Adventure", "Animation", "Horror"]
data["genre"] = data["genre"].apply(lambda genre: [1 if genre == g else 0 for g in genres])

# Cosine Similarity Functions
def calculate_cosine_similarity(a, b):
    return round(np.dot(a, b) / (norm(a) * norm(b)), 3)

def flatten_data(row):
    return np.array([item for idx in row for item in (idx if isinstance(idx, list) else [idx])])

# If needed, compute cosine similarity matrix
compute_cosine_similarity = False
if compute_cosine_similarity:
    # Compute the similarity matrix
    precomputed_data = [flatten_data(row) for row in data.values]
    matrix = [[calculate_cosine_similarity(precomputed_data[row], precomputed_data[col]) for col in range(len(data))] for row in range(len(data))]
    similarity_matrix = pd.DataFrame(matrix)

# Recommendation function
def movie_recommendations(movies_liked, similarity_matrix, movie_names):
    indices = [movie_names.index(movie) for movie in movies_liked]
    best_recs = []
    similarities = {}
    similarity_row = np.mean([similarity_matrix[i] for i in indices], axis=0)
    highest_vals = sorted(similarity_row, reverse=True)[1:10]

    for val in highest_vals:
        new_best_recs = np.where(similarity_row == val)[0]
        best_recs.extend(new_best_recs)
        for rec in new_best_recs:
            similarities[rec] = max(similarities.get(rec, 0), 100 * val)

    similarities_sorted = dict(sorted(similarities.items(), key=lambda x: x[1], reverse=True))
    rec_list = [f"{movie_names[rec]}...{round(similarities[rec], 2)}% match" for rec in similarities_sorted if movie_names[rec] not in movies_liked]
    return rec_list

# Define movie dropdown widget (used in Google Colab or Jupyter Notebooks)
dropdown = widgets.SelectMultiple(
    options=[],
    description='Movies:',
    disabled=False
)

def update_dropdown(change):
    search_text = change['new']
    filtered_movie_names = [name for name in movie_names if search_text.lower() in name.lower()]
    dropdown.options = filtered_movie_names

# Load precomputed similarity matrix or calculate
similarity_matrix = pd.read_csv("data/similarity.csv", header=None)

# Example usage
movies_liked = ['Movie1', 'Movie2']
recommendations = movie_recommendations(movies_liked, similarity_matrix, movie_names)
print("Recommended Movies: ", recommendations)
