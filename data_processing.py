import numpy as np
import requests
from bs4 import BeautifulSoup
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import logging


class DataProcessor:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch_web_data(self, url: str) -> str:
        """
        Fetches web data from the given URL and returns its textual content.
        """
        try:
            response = requests.get(url)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
            text_content = soup.text

            if not text_content or text_content.isspace():
                self.logger.warning(
                    f"No or minimal textual content fetched from {url}")
                return ""

            self.logger.info(f"Successfully fetched data from {url}")
            return text_content
        except requests.RequestException as e:
            self.logger.error(f"Error fetching data from {url}. Error: {e}")
            return ""

    def process_image(self, img_path: str, image_model) -> np.ndarray:
        """
        Processes an image and returns its features using the provided image model.
        """
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)
            features = image_model.predict(x)
            self.logger.info(f"Successfully processed image from {img_path}")
            return features.flatten()
        except Exception as e:
            self.logger.error(
                f"Error processing image from {img_path}. Error: {e}")
            return np.array([])
