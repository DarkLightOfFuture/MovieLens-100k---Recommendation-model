import unittest
from model import RecommendationModel


class TestRecommendationModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Runs once before all tests"""
        print("Loading model for tests...")
        cls.model = RecommendationModel()
        print("Model loaded!\n")

    def test_data_loading(self):
        print("Testing data loading...")
        self.assertIsNotNone(self.model.users_df, "! Users dataframe should not be None")
        self.assertIsNotNone(self.model.movies_df, "! Movies dataframe should not be None")
        self.assertIsNotNone(self.model.reviews_df, "! Reviews dataframe should not be None")

        self.assertGreater(len(self.model.users_df), 0, "! There is no users data")
        self.assertGreater(len(self.model.movies_df), 0, "! There is no movies data")
        self.assertGreater(len(self.model.reviews_df), 0, "! There is no reviews data")
        print("Data loaded correctly!\n")

    def test_encoders_fitted(self):
        """Test if encoders are properly fitted"""
        print("Testing encoders...")

        # Check if encoders have classes
        self.assertTrue(len(self.model.user_id_enc.classes_) > 0, "! User encoder has no classes")
        self.assertTrue(len(self.model.movie_id_enc.classes_) > 0, "! Movie encoder has no classes")
        self.assertTrue(len(self.model.gender_enc.classes_) > 0, "! Gender encoder has no classes")
        self.assertTrue(len(self.model.occp_enc.classes_) > 0, "! Occupation encoder has no classes")

        print("Encoders working correctly!\n")

    def test_model_building(self):
        print("Testing model building...")

        # Try to build the model
        try:
            history = self.model.build()
            self.assertIsNotNone(history, "! Model should return training history")
            print("Model built successfully!\n")

        except Exception as e:
            self.fail(f"! Model building failed with error: {e}")

    def test_recommendations_for_existing_user(self):
        print("Testing recommendations...")

        real_user_id = self.model.df['user id'].iloc[0]
        recommendations = self.model.recommend_movies(real_user_id, top_n=5)

        # Check results
        self.assertIsNotNone(recommendations, "! Recommendations cannot be None")
        self.assertLessEqual(len(recommendations), 15, "! Should return at most 15 recommendations")

        if not recommendations.empty:
            self.assertTrue('movie_id' in recommendations.columns, "! Should have movie_id column")
            self.assertTrue('title' in recommendations.columns, "! Should have title column")
            self.assertTrue('predicted_rating' in recommendations.columns, "! Should have predicted_rating column")

        print("Recommendations working correctly!\n")

    def test_unknown_user_handling(self):
        print("Testing unknown user handling...")

        # Not existing user
        fake_user_id = 999999
        recommendations = self.model.recommend_movies(fake_user_id, top_n=5)

        # Should return empty DataFrame
        self.assertIsNotNone(recommendations, "! Should return something (even if empty)")
        print("Unknown user handling works!\n")

    def test_simulating_records(self):
        """Test if simulated records are included in dataframe"""
        print("Testing simulated records...")

        self.assertTrue(len(self.model.df) > self.model.init_df_size, "! Dataframe should include simulated records")

        print("Simulated records included correctly!\n")

print("Running unit tests...")
print("=" * 50)

loader = unittest.TestLoader()
tests_set = loader.loadTestsFromTestCase(TestRecommendationModel)

runner = unittest.TextTestRunner(verbosity=1)
result = runner.run(tests_set)
if "__main__" == __name__:
    print("=" * 50)
    if result.wasSuccessful():
        print("All tests passed!")
    else:
        print("! Some tests failed")

        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")