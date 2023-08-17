from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import logging


class ImageModel:
    def __init__(self, num_classes, base_model_architecture=VGG16):
        self.logger = logging.getLogger(self.__class__.__name__)
        base_model = base_model_architecture(
            weights='imagenet', include_top=False)

        # Add custom layers for fine-tuning
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        self.model = Model(inputs=base_model.input, outputs=predictions)

        # Freeze all layers of the base model
        for layer in base_model.layers:
            layer.trainable = False

        self.model.compile(
            optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        self.training_data = []
        self.labels = []

    def preprocess_image(self, img_path):
        try:
            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            return preprocess_input(x)
        except Exception as e:
            self.logger.error(f"Error processing image {img_path}: {str(e)}")
            return None

    def extract_features(self, img_path):
        preprocessed_image = self.preprocess_image(img_path)
        if preprocessed_image is None:
            return None
        return self.model.predict(preprocessed_image).flatten()

    def train(self, epochs=10):
        """
        Fine-tune the model using the accumulated training data.
        """
        if not self.training_data or not self.labels:
            self.logger.warning(
                "No training data available. Skipping training.")
            return

        X_train = np.array(self.training_data)
        y_train = np.array(self.labels)
        self.model.fit(X_train, y_train, epochs=epochs)
        self.logger.info(f"Model trained for {epochs} epochs.")

    def feedback(self, img_path, label):
        features = self.extract_features(img_path)
        if features is None:
            return
        self.training_data.append(features)
        self.labels.append(label)
        self.logger.info(
            f"Feedback received for image {img_path} with label {label}.")

    def perceive(self, img_path):
        """
        Process the incoming image data and update the training data.
        """
        features = self.extract_features(img_path)
        if features is None:
            return
        self.training_data.append(features)
        self.logger.info(f"Image {img_path} perceived and features extracted.")

    def predict(self, img_path):
        """
        Predict the class of the given image.
        """
        features = self.extract_features(img_path)
        if features is None:
            return None
        return np.argmax(self.model.predict(np.array([features])))
