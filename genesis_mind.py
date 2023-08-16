import matplotlib.pyplot as plt
from data_processing import fetch_web_data, process_image
from text_model import TextModel
from image_model import ImageModel


class GenesisMind:
    THRESHOLD = 0.5

    def __init__(self, num_classes):
        self.state = 0
        self.memory = []
        self.text_model = TextModel()
        self.image_model = ImageModel(num_classes)
        self.evaluation_history = []

    def perceive(self, data, data_type="text"):
        if data_type == "text":
            self.text_model.perceive(data)
        elif data_type == "image":
            vectorized_image = process_image(data, self.image_model.model)
            self.image_model.perceive(vectorized_image)

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
        """
        Analyze the perceived data, make connections between different pieces of data, 
        and update the state based on this analysis.
        """
        # For simplicity, let's say the AI thinks by averaging the values of the perceived data
        if self.memory:
            average_value = sum(self.memory) / len(self.memory)
            self.state = average_value

    def act(self):
        # For now, let's just return the state value
        return self.state

    def remember(self, data):
        self.memory.append(data)

    def train(self, data_type="text"):
        if data_type == "text":
            self.text_model.train()
        elif data_type == "image":
            self.image_model.train()

    def recall(self):
        # For simplicity, return the last remembered item
        return self.memory[-1] if self.memory else None

    def feedback(self, data, correct_data, data_type="text"):
        if data_type == "text":
            self.text_model.feedback(data, correct_data)
        elif data_type == "image":
            self.image_model.feedback(data, correct_data)

    def evaluate(self, predicted_data, actual_data):
        """
        Assess the accuracy of the predictions made by the AI.
        """
        # Vectorize the text data
        vectorized_predicted = self.text_model.vectorize(predicted_data)
        vectorized_actual = self.text_model.vectorize(actual_data)

        # Compute the Mean Absolute Error (MAE) on the vectorized data
        mae = sum(abs(vectorized_predicted[i] - vectorized_actual[i])
                  for i in range(len(vectorized_predicted))) / len(vectorized_predicted)
        return mae

    def recursive_learn(self, data):
        """
        Learn recursively by perceiving the data, making predictions, evaluating performance, 
        and retraining if necessary.
        """
        if data is None:
            print("Warning: Received None data. Skipping this iteration.")
            return

        self.perceive(data)
        prediction = self.predict(data)
        eval_score = self.evaluate(data, prediction)
        self.evaluation_history.append(eval_score)

        if eval_score > self.THRESHOLD:
            self.train()

    def plot_training_curve(self):
        """
        Plot the training curve based on the evaluation history.
        """
        plt.plot(self.evaluation_history)
        plt.xlabel('Iterations')
        plt.ylabel('Evaluation Score')
        plt.title('Training Curve')
        plt.show()

    def adapt(self, evaluation_score):
        """
        Adapt the AI's models based on the evaluation score.
        """
        # If the evaluation score is above a certain threshold, retrain the models
        if evaluation_score > self.THRESHOLD:
            self.train()


if __name__ == "__main__":
    NUM_CLASSES = 1000
    mind = GenesisMind(NUM_CLASSES)

    # Fetch data multiple times to populate past_data
    all_data = []
    for url in ["https://www.wikipedia.org/", "https://www.google.com/"]:
        data = fetch_web_data(url)
        all_data.append(data)

    # Fit the vectorizer to all the data
    mind.text_model.fit_vectorizer(all_data)

    # Use recursive learning for each data
    for data in all_data:
        mind.recursive_learn(data)

    # Plot the training curve to visualize the AI's performance over time
    mind.plot_training_curve()

    # Predict using the last data
    prediction = mind.predict(all_data[-1])
    print(f"GenesisMind's prediction for the next state: {prediction}")

    # For demonstration purposes, let's evaluate and adapt using the last two data points
    eval_score = mind.evaluate(all_data[-2], all_data[-1])
    print(f"Evaluation Score: {eval_score}")
    mind.adapt(eval_score)
