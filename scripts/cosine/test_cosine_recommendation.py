import os
import sys

sys.path.append(os.path.abspath("src"))

from src.data.load import load_data
from src.cosine.recommend import individual_recommend, combined_recommend
from src.utils.paths import COSINE_SIMILARITY_PATH, MOVIE_NAMES_PATH


def main():
    print("=== Testing Cosine Recommendation System ===")

    print("\n[1] Load Similarity Matrix...")
    similarity_matrix = load_data(COSINE_SIMILARITY_PATH)
    movie_names = load_data(MOVIE_NAMES_PATH)["movie_name"].tolist()

    print(
        f"Data loaded. Similarity Matrix shape: {similarity_matrix.shape}, "
        f"Movie names count: {len(movie_names)}"
    )

    movies_liked = [
        "Star Trek II: The Wrath of Khan (1982)",
        "Star Wars: Episode VI - Return of the Jedi (1983)",
        "Planet of the Apes (2001)",
        "Final Fantasy: The Spirits Within (2001)",
        "Interstellar (2014)",
    ]

    print("\n[2] Individual Recommendation...")
    print(f"INPUT: {movies_liked}")
    recommendation = individual_recommend(movies_liked, similarity_matrix, movie_names)
    print(f"\nRecommendations: {recommendation}")

    print("\n[3] Combined Recommendation...")
    print(f"INPUT: {movies_liked}")
    recommendation = combined_recommend(movies_liked, similarity_matrix, movie_names)
    print(f"\nRecommendations: {recommendation}")

    print("\nCosine recommendations completed successfully!")


if __name__ == "__main__":
    main()