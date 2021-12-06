import numpy as np
from emotions.model import create_model


class EmotionPredictor:

    def __init__(self, labels, model_path=f'../resources/models/emotions/emotions.h5'):
        # dictionary which assigns each label an emotion (alphabetical order)
        self.labels = labels
        #self.model = create_model(kernel_size=(5,5))
        self.model = create_model()
        self.model.load_weights(model_path)

    def predict(self, image):
        prediction = self.model.predict(image)
        max_index = int(np.argmax(prediction))
        print(f"EmotionPredictor: [{self.labels[max_index]}]")
        return self.labels[max_index]
