import matplotlib.pyplot as plt
from data_processing import fetch_web_data, process_image
from text_model import TextModel
from image_model import ImageModel


class GenesisMind:
    def __init__(self):
        self.state = 0
        self.memory = []
        self.text_model = TextModel()
        self.image_model = ImageModel()

    def perceive(self, data, data_type="text"):
        if data_type == "text":
            self.text_model.perceive(data)
        elif data_type == "image":
            vectorized_image = process_image(data, self.image_model.model)

    def predict(self, data, top_n=10):
        """
        Predict the next state based on the current state and return the top n words.
        """
        return self.text_model.predict(data, top_n)

    def visualize_prediction(self, data, top_n=10):
        """
        Visualize the predicted TF-IDF scores for the top n words.
        """
        vectorized_data = self.text_model.vectorize(data)
        prediction = self.text_model.model.predict([vectorized_data])
        top_indices = prediction[0].argsort()[-top_n:][::-1]
        top_words = [self.text_model.vectorizer.get_feature_names_out()[i]
                     for i in top_indices]
        top_scores = prediction[0][top_indices]

        plt.barh(top_words, top_scores)
        plt.xlabel('TF-IDF Score')
        plt.ylabel('Words')
        plt.title('Top Predicted Words')
        plt.gca().invert_yaxis()
        plt.show()

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

    def feedback(self, data, correct_data):
        # Placeholder for feedback mechanism
        pass

    def evaluate(self):
        # Placeholder for self-evaluation mechanisms
        return "Evaluation result"

    def adapt(self):
        # Placeholder for adaptation mechanisms
        pass


if __name__ == "__main__":
    mind = GenesisMind()

    # Fetch data multiple times to populate past_data
    all_data = []
    for url in ["https://www.wikipedia.org/", "https://www.example.com/"]:
        data = fetch_web_data(url)
        all_data.append(data)

    # Fit the vectorizer to all the data
    mind.text_model.fit_vectorizer(all_data)

    # Perceive and think on each data
    for data in all_data:
        mind.perceive(data)
        mind.think()

    # Train the text model after perceiving multiple data sources
    mind.text_model.train()

    # Predict using the last data
    prediction = mind.predict(all_data[-1])
    print(f"GenesisMind's prediction for the next state: {prediction}")
