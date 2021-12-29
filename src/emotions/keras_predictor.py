import numpy as np
from tensorflow import keras


class EmotionPredictor:

    def __init__(self, labels, model_path=f'../resources/models/emotions/emotions'):
        # dictionary which assigns each label an emotion (alphabetical order)
        self.labels = labels
        self.model = keras.models.load_model(model_path + ".json")
        self.model.load_weights(model_path + ".h5")

    def predict(self, image):
        prediction = self.model.predict(image)
        max_index = int(np.argmax(prediction))
        print(f"EmotionPredictor: [{self.labels[max_index]}]")
        return self.labels[max_index]
