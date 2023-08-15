from keras.applications.vgg16 import VGG16


class ImageModel:
    def __init__(self):
        self.model = VGG16(weights='imagenet', include_top=False)

    def extract_features(self, preprocessed_image):
        """
        Extract features from a preprocessed image using the VGG16 model.
        """
        return self.model.predict(preprocessed_image).flatten()
