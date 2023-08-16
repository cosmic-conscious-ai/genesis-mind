from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import numpy as np


class ImageModel:
    def __init__(self, num_classes):
        base_model = VGG16(weights='imagenet', include_top=False)

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
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        return preprocess_input(x)

    def extract_features(self, img_path):
        preprocessed_image = self.preprocess_image(img_path)
        return self.model.predict(preprocessed_image).flatten()

    def train(self, epochs=10):
        """
        Fine-tune the model using the accumulated training data.
        """
        X_train = np.array(self.training_data)
        y_train = np.array(self.labels)
        self.model.fit(X_train, y_train, epochs=epochs)

    def feedback(self, img_path, label):
        features = self.extract_features(img_path)
        self.training_data.append(features)
        self.labels.append(label)

    def perceive(self, img_path):
        """
        Process the incoming image data and update the training data.
        """
        features = self.extract_features(img_path)
        self.training_data.append(features)
