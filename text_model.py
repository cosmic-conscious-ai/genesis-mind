import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from logger import logger
from sklearn.exceptions import NotFittedError
import logging


class TextModel:
    def __init__(self):
        self.model = LinearRegression()
        self.vectorizer = TfidfVectorizer()
        self.past_data = []
        self.logger = logging.getLogger(self.__class__.__name__)

    def fit_vectorizer(self, data):
        try:
            if not data or all(not d.strip() for d in data):
                self.logger.warning(
                    "Empty data provided. Vectorizer not fitted.")
                return
            self.vectorizer.fit(data)
            self.logger.info("Vectorizer fitted to the data.")
        except Exception as e:
            self.logger.error(f"Error fitting vectorizer: {e}")

    def vectorize(self, data):
        try:
            if isinstance(data, str):
                data = [data]

            if not data or all(not item.strip() for item in data):
                self.logger.warning(
                    "Trying to vectorize empty or None data. Returning None.")
                return None
            if not self.is_vectorizer_fitted():
                self.logger.warning(
                    "Vectorizer is not fitted. Cannot vectorize data.")
                return None

            return self.vectorizer.transform(data).toarray()
        except Exception as e:
            self.logger.error(f"Error during vectorization: {e}")
            return None

    def train(self):
        try:
            if len(self.past_data) < 2 or any(d is None for d in self.past_data):
                self.logger.warning("Not enough data to train the model.")
                return

            X = np.array(self.past_data[:-1])
            y = np.array(self.past_data[1:])

            if len(X.shape) == 3:
                num_samples, num_features, _ = X.shape
                X = X.reshape(num_samples, num_features)
            if len(y.shape) == 3:
                num_samples, num_features, _ = y.shape
                y = y.reshape(num_samples, num_features)

            self.model.fit(X, y)
            self.logger.info("Model trained using past data.")
        except Exception as e:
            self.logger.error(f"Error during model training: {e}")

    def predict(self, data, top_n=10):
        """
        Predict the next state based on the current state and return the top n words.
        """
        vectorized_data = self.vectorize(data)
        if vectorized_data is None:
            return None

        if not hasattr(self.model, 'coef_'):
            self.logger.warning(
                "Model hasn't been trained yet. Cannot make predictions.")
            return None

        prediction = self.model.predict([vectorized_data])
        top_indices = prediction[0].argsort()[-top_n:][::-1]
        top_words = [self.vectorizer.get_feature_names_out()[i]
                     for i in top_indices]
        self.logger.info(f"Top {top_n} predictions made for the given data.")
        return top_words

    def is_vectorizer_fitted(self):
        """
        Check if the vectorizer is fitted.
        """
        try:
            self.vectorizer.transform(["test"])
            return True
        except NotFittedError:
            return False

    def perceive(self, data):
        """
        Process the incoming text data and update the past data.
        """
        vectorized_data = self.vectorize([data])  # Wrap data in a list
        if vectorized_data is None:
            return

        # If past_data is empty, initialize it with vectorized_data
        # Otherwise, vertically stack the new data
        if len(self.past_data) == 0:
            self.past_data = vectorized_data
        else:
            self.past_data = np.vstack((self.past_data, vectorized_data))

        self.logger.info("Data perceived and added to past data.")

    def feedback(self, data, correct_data):
        """
        Adjust the model's past data based on feedback.
        """
        vectorized_data = self.vectorize(data)
        vectorized_correct_data = self.vectorize(correct_data)

        if vectorized_data is None or vectorized_correct_data is None:
            return

        data_index = next((i for i, past_datum in enumerate(self.past_data)
                           if np.array_equal(past_datum, vectorized_data)), None)

        if data_index is not None:
            self.past_data[data_index] = vectorized_correct_data
            self.logger.info("Feedback received and past data adjusted.")
