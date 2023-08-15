import numpy as np
import requests
from bs4 import BeautifulSoup
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input


def fetch_web_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.text


def process_image(img_path, image_model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = image_model.predict(x)
    return features.flatten()
