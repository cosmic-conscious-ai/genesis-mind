import numpy as np
import requests
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import logging
import random
import time


class DataProcessor:
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        # Add more user agents as needed
    ]

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def fetch_web_data(self, url: str) -> str:
        headers = {
            "User-Agent": random.choice(self.USER_AGENTS)
        }

        self.logger.info(f"Visiting url: {url}")

        for _ in range(3):  # Retry up to 3 times
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                html_content = response.text

                if not html_content or html_content.isspace():
                    self.logger.warning(
                        f"No or minimal content fetched from {url}")
                    return ""

                if "CAPTCHA" in html_content or "Are you a robot?" in html_content:
                    self.logger.warning(
                        f"Possible bot detection at {url}. Consider using a proxy or increasing delay.")
                    return ""

                self.logger.info(f"Successfully fetched data from {url}")
                return html_content

            except (requests.RequestException, Exception) as e:
                self.logger.error(
                    f"Error fetching data from {url} on attempt {_+1}. Error: {e}")
                time.sleep(2**(_+1))  # Exponential backoff

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
