import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt


class GenesisMind:
    def __init__(self):
        self.state = 0
        self.past_data = []
        self.model = LinearRegression()
        self.vectorizer = TfidfVectorizer()
        self.memory = []

    def perceive(self, data, data_type="text"):
        """
        Process the incoming data and update the mind's state.
        """
        if data_type == "text":
            vectorized_data = self.vectorizer.transform([data]).toarray()
            self.past_data.append(vectorized_data[0])

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

    def visualize_prediction(self, data, top_n=10):
        """
        Visualize the predicted TF-IDF scores for the top n words.
        """

        vectorized_data = self.vectorizer.transform([data])
        prediction = self.model.predict(vectorized_data)
        top_indices = prediction[0].argsort()[-top_n:][::-1]
        top_words = [self.vectorizer.get_feature_names_out()[i]
                     for i in top_indices]
        top_scores = prediction[0][top_indices]

        plt.barh(top_words, top_scores)
        plt.xlabel('TF-IDF Score')
        plt.ylabel('Words')
        plt.title('Top Predicted Words')
        plt.gca().invert_yaxis()
        plt.show()

    def learn(self):
        """
        Train the model based on all past data.
        """
        if len(self.past_data) < 2:
            return  # Not enough data to train

        X = np.array(self.past_data[:-1])
        y = np.array(self.past_data[1:])
        self.model.fit(X, y)

    def fetch_web_data(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.text

    def think(self):
        # For now, let's just square the state value
        self.state = self.state ** 2

    def act(self):
        # For now, let's just return the state value
        return self.state

    def remember(self, data):
        self.memory.append(data)

    def recall(self):
        # For simplicity, return the last remembered item
        return self.memory[-1] if self.memory else None

    def evaluate(self):
        # Placeholder for self-evaluation mechanisms
        pass

    def adapt(self):
        # Placeholder for adaptation mechanisms
        pass

    def fit_vectorizer(self, data):
        """
        Fit the TF-IDF vectorizer to the data.
        """
        self.vectorizer.fit([data])


if __name__ == "__main__":
    mind = GenesisMind()

    # Fetch data multiple times to populate past_data
    all_data = []
    for url in ["https://www.wikipedia.org/", "https://www.example.com/"]:
        data = mind.fetch_web_data(url)
        all_data.append(data)

    # Fit the vectorizer to all the data
    mind.vectorizer.fit(all_data)

    # Perceive and think on each data
    for data in all_data:
        vectorized_data = mind.vectorizer.transform([data])
        mind.perceive(data)
        mind.think()

    # Train the model after perceiving multiple data sources
    mind.learn()  # Train using the past data

    # Predict using the last data
    prediction = mind.predict(all_data[-1])
    print(f"GenesisMind's prediction for the next state: {prediction}")
