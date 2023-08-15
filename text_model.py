import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer


class TextModel:
    def __init__(self):
        self.model = LinearRegression()
        self.vectorizer = TfidfVectorizer()
        self.past_data = []

    def fit_vectorizer(self, data):
        """
        Fit the TF-IDF vectorizer to the data.
        """
        self.vectorizer.fit(data)

    def vectorize(self, data):
        """
        Convert text data into its vector representation using TF-IDF.
        """
        return self.vectorizer.transform([data]).toarray()[0]

    def train(self):
        """
        Train the linear regression model using the past data.
        """
        if len(self.past_data) < 2:
            return  # Not enough data to train

        X = np.array(self.past_data[:-1])
        y = np.array(self.past_data[1:])
        self.model.fit(X, y)

    def predict(self, data, top_n=10):
        """
        Predict the next state based on the current state and return the top n words.
        """
        vectorized_data = self.vectorizer.transform([data])
        if not hasattr(self.model, 'coef_'):
            return None  # Model hasn't been trained yet

        prediction = self.model.predict(vectorized_data)
        top_indices = prediction[0].argsort()[-top_n:][::-1]
        top_words = [self.vectorizer.get_feature_names_out()[i]
                     for i in top_indices]
        return top_words

    def perceive(self, data):
        """
        Process the incoming text data and update the past data.
        """
        vectorized_data = self.vectorizer.transform([data]).toarray()
        self.past_data.append(vectorized_data[0])
