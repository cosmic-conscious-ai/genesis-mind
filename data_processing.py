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
        Fetches web data from the given URL and returns its full HTML content.
        """
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            html_content = response.text  # Get the full HTML content

            if not html_content or html_content.isspace():
                self.logger.warning(
                    f"No or minimal content fetched from {url}")
                return ""

            self.logger.info(f"Successfully fetched data from {url}")
            return html_content
        except (requests.RequestException, Exception) as e:
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
