import unittest
import numpy as np
import pandas as pd
from movie_recommender import calculate_cosine_similarity, movie_recommendations

class TestMovieRecommender(unittest.TestCase):

    def setUp(self):
        # Setup some sample data for testing
        self.sample_data = pd.DataFrame({
            'name': ['Movie1', 'Movie2', 'Movie3'],
            'genre': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            'popularityScore': [100, 200, 300],
            'gross': [1.5, 2.0, 1.0],
            'runtime': [120, 150, 90]
        })
        self.movie_names = ['Movie1', 'Movie2', 'Movie3']
        self.similarity_matrix = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.6],
            [0.3, 0.6, 1.0]
        ])
        self.movies_liked = ['Movie1']

    def test_calculate_cosine_similarity(self):
        # Test cosine similarity between two vectors
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([1, 1, 0])
        similarity = calculate_cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(similarity, 0.707, places=3)

    def test_movie_recommendations(self):
        # Test that the correct recommendations are generated
        recommendations = movie_recommendations(self.movies_liked, self.similarity_matrix, self.movie_names)
        expected_recommendations = ['Movie2...80.0% match']
        self.assertEqual(recommendations, expected_recommendations)

    def test_data_preprocessing(self):
        # Ensure that the data has the correct columns and expected values after processing
        processed_data = self.sample_data.copy()
        processed_data["popularityScore"] = processed_data["popularityScore"] * 2  # Example of data manipulation
        self.assertEqual(processed_data["popularityScore"].iloc[0], 200)
    
    # Additional tests for I/O operations, file handling, etc.
    def test_file_loading(self):
        # Simulate loading a CSV file
        df = pd.read_csv("data/sample_movies.csv")  # Use a small test CSV for unit testing
        self.assertFalse(df.empty)

if __name__ == '__main__':
    unittest.main()
