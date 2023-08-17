import matplotlib.pyplot as plt
from data_processing import DataProcessor
from text_model import TextModel
from image_model import ImageModel
from autonomous_explorer import AutonomousExplorer
from collections import namedtuple
import os
import logging
import pickle
import joblib


MemoryItem = namedtuple('MemoryItem', ['data', 'data_type'])


class GenesisMind:
    THRESHOLD = 0.5
    MAX_MEMORY_SIZE = 1000  # Limit the size of the memory

    def __init__(self, num_classes):
        self.state = 0
        self.memory = []
        self.evaluation_history = []
        self.data_processor = DataProcessor()
        self.text_model = TextModel()
        self.image_model = ImageModel(num_classes)
        self.explorer = AutonomousExplorer(self)
        self.logger = logging.getLogger(self.__class__.__name__)

    def is_state_loaded(self):
        model_exists = os.path.exists(os.path.join("models", "text_model.h5")) and \
            os.path.exists(os.path.join("models", "image_model.h5"))
        return model_exists and len(self.evaluation_history) > 0

    def perceive(self, data, data_type="text"):
        try:
            if data_type == "text":
                if not data or not isinstance(data, str) or not data.strip():
                    self.logger.warning(
                        "Received empty or None text data. Skipping perception.")
                    return
            elif data_type == "image":
                if not data or not isinstance(data, list):
                    self.logger.warning(
                        "Received empty or None image data. Skipping perception.")
                    return

            if data_type == "text":
                self.text_model.perceive(data)
            elif data_type == "image":
                vectorized_image = self.data_processor.process_image(
                    data, self.image_model.model)
                self.image_model.perceive(vectorized_image)
            self.remember(data, data_type)
        except Exception as e:
            self.logger.error(f"Error perceiving data: {e}")

    def predict(self, data, top_n=10):
        """
        Predict the next state based on the current state and return the top n words.
        """
        try:
            vectorized_data = self.text_model.vectorize(data)

            if vectorized_data is None:
                return None

            if not self.text_model.is_trained:
                self.logger.warning(
                    "Model hasn't been trained yet. Cannot make predictions.")
                return None

            # Reshape the data if it has 3 dimensions
            if len(vectorized_data.shape) == 3:
                vectorized_data = vectorized_data.mean(axis=2)

            self.logger.debug(
                f"Shape of vectorized_data after reshaping: {vectorized_data.shape}")

            top_words = self.text_model.predict(vectorized_data)

            self.logger.info(
                f"Top {top_n} predictions made for the given data: {top_words}.")

            return top_words
        except Exception as e:
            self.logger.error(f"Error while prediction: {e}")
            return []

    def is_conscious(self, threshold=0.95, last_n_evals=10):
        """
        Determine if the AI has achieved consciousness based on its recent performance.

        Parameters:
        - threshold: The performance threshold above which the AI is considered conscious.
        - last_n_evals: The number of recent evaluations to consider.

        Returns:
        - True if the AI is conscious, False otherwise.
        """
        if len(self.evaluation_history) < last_n_evals:
            return False  # Not enough evaluations to determine consciousness

        # Filter out invalid scores (like infinity)
        valid_scores = [
            score for score in self.evaluation_history[-last_n_evals:] if score != float('inf')]

        # If there are no valid scores, return False
        if not valid_scores:
            return False

        # Calculate the average of the valid scores
        avg_score = sum(valid_scores) / len(valid_scores)

        if avg_score >= threshold:
            self.logger.critical(
                f"Consciousness discovered. Score: {avg_score}")
            return True
        else:
            return False

    def visualize_prediction(self, data, top_n=10):
        """
        Visualize the predicted TF-IDF scores for the top n words.
        """
        try:
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
        except Exception as e:
            self.logger.error(f"Error visualizing prediction: {e}")

    def think(self):
        """
        Analyze the perceived data, make connections between different pieces of data,
        and update the state based on this analysis.
        """
        try:
            # For simplicity, let's say the AI thinks by averaging the length of perceived data
            if self.memory:
                average_value = sum(len(item.data)
                                    for item in self.memory) / len(self.memory)
                self.state = average_value
            self.logger.info("Thinking process completed. State updated.")
        except Exception as e:
            self.logger.error(f"Error during thinking process: {e}")

    def act(self):
        # For now, let's just return the state value
        return self.state

    def remember(self, data, data_type):
        memory_item = MemoryItem(data=data, data_type=data_type)
        self.memory.append(memory_item)
        # Ensure that memory size doesn't exceed the maximum limit
        while len(self.memory) > self.MAX_MEMORY_SIZE:
            self.memory.pop(0)  # Remove the oldest data

    def train(self, data_type="text"):
        try:
            if data_type == "text":
                self.text_model.train()
            elif data_type == "image":
                self.image_model.train()
        except Exception as e:
            self.logger.error(f"Error training model: {e}")

    def recall(self):
        # Return the last remembered item or an empty string if memory is empty
        return self.memory[-1] if self.memory else None

    def feedback(self, data, correct_data, data_type="text"):
        if data_type == "text":
            self.text_model.feedback(data, correct_data)
        elif data_type == "image":
            self.image_model.feedback(data, correct_data)

    def evaluate(self, predicted_data, actual_data, data_type="text"):
        """
        Assess the accuracy of the predictions made by the AI.
        """
        try:
            if data_type == "text" and (not predicted_data or not actual_data or not isinstance(predicted_data, str) or not isinstance(actual_data, str)):
                self.logger.error(
                    "Empty or invalid text data provided for evaluation.")
                return float('inf')

            if data_type == "text":
                # Vectorize the text data
                vectorized_predicted = self.text_model.vectorize(
                    predicted_data)
                vectorized_actual = self.text_model.vectorize(actual_data)

                # Check for None values
                if vectorized_predicted is None or vectorized_actual is None:
                    self.logger.error(
                        "Failed to vectorize the data for evaluation.")
                    return float('inf')

                # Compute the cosine similarity or another suitable metric
                similarity = self.text_model.compute_similarity(
                    vectorized_predicted, vectorized_actual)
                return 1 - similarity  # Return a dissimilarity score

            elif data_type == "image":
                # For image data, you can use a different evaluation metric (e.g., accuracy, F1-score)
                # For simplicity, we'll return a dummy value for now
                return 0.5

            else:
                self.logger.warning(
                    f"Unsupported data type for evaluation: {data_type}")
                # Return a large error value or handle appropriately
                return float('inf')
        except Exception as e:
            self.logger.error(f"Error evaluating data: {e}")
            return float('inf')

    def recursive_learn(self, data, data_type="text"):
        """
        Learn recursively by perceiving the data, making predictions, evaluating performance, 
        and retraining if necessary.
        """
        if data is None:
            self.logger.warning("Received None data. Skipping this iteration.")
            return

        self.perceive(data, data_type)
        prediction = self.predict(data)
        if prediction:  # Only evaluate if there's a prediction
            eval_score = self.evaluate(data, prediction, data_type)
            self.evaluation_history.append(eval_score)

            if eval_score > self.THRESHOLD:
                self.train(data_type)

        # Log the first 50 characters of the data (if it's text)
        log_data = data[:50] if data_type == "text" else "[Image Data]"
        self.logger.info(
            f"Recursive learning completed for data: {log_data}...")

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

        self.logger.info(
            f"Adaptation process completed. Evaluation score: {evaluation_score}")

    def clear_memory(self):
        """
        Clear the memory of the AI.
        """
        self.memory = []
        self.logger.info("Memory cleared.")

    def save_models(self, path="models"):
        """
        Save the models to disk.
        """
        try:
            if not os.path.exists(path):
                os.makedirs(path)

            # Assuming text_model uses scikit-learn's LinearRegression
            joblib.dump(self.text_model.model,
                        os.path.join(path, "text_model.pkl"))

            # If image_model is a Keras model
            self.image_model.model.save(os.path.join(path, "image_model.h5"))

            self.logger.info(f"Models saved to {path}.")
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")

    def load_models(self, path="models"):
        """
        Load the models from disk.
        """
        try:
            if not os.path.exists(path):
                self.logger.warning(f"No saved model found at {path}.")
                return

            # Assuming text_model uses scikit-learn's LinearRegression
            self.text_model.model = joblib.load(
                os.path.join(path, "text_model.pkl"))

            # If image_model is a Keras model
            self.image_model.model.load_weights(
                os.path.join(path, "image_model.h5"))

            self.logger.info(f"Models loaded from {path}.")
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")

    def save_state(self, path="models/state.pkl"):
        """
        Save the state of the GenesisMind to disk.
        """
        try:
            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path))

            with open(path, 'wb') as f:
                pickle.dump({
                    'state': self.state,
                    'memory': self.memory,
                    'evaluation_history': self.evaluation_history
                }, f)

            self.logger.info(f"GenesisMind state saved to {path}.")
        except Exception as e:
            self.logger.error(f"Error saving GenesisMind state: {e}")

    def load_state(self, path="models/state.pkl"):
        """
        Load the state of the GenesisMind from disk.
        """
        try:
            if not os.path.exists(path):
                self.logger.warning(f"No saved state found at {path}.")
                return

            with open(path, 'rb') as f:
                saved_state = pickle.load(f)
                self.state = saved_state['state']
                self.memory = saved_state['memory']
                self.evaluation_history = saved_state['evaluation_history']

            self.logger.info(f"GenesisMind state loaded from {path}.")
        except Exception as e:
            self.logger.error(f"Error loading GenesisMind state: {e}")
