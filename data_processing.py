import numpy as np
import requests
from bs4 import BeautifulSoup
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


def fetch_web_data(url):
    try:
        response = requests.get(url)
        # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        text_content = soup.text

        if not text_content or text_content.isspace():
            print(f"Warning: No or minimal textual content fetched from {url}")
            return ""

        return text_content
    except requests.RequestException as e:
        print(f"Error fetching data from {url}. Error: {e}")
        return ""


def process_image(img_path, image_model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = image_model.predict(x)
    return features.flatten()
