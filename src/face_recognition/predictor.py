import os
import pickle5 as pickle
import cv2


class FaceRecognitionModel:

    def __init__(self, labels2names_map_path=f'../resources/models/faces/faces_labels.bin',
                 model_path=f'../resources/models/faces/faces.yaml'):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.isfile(model_path):
            self.recognizer.read(model_path)
        with open(labels2names_map_path, 'rb') as labels2names_map_file:
            self.labels2names_map = pickle.load(labels2names_map_file)

    def predict(self, image):
        label_predicted, confidence = self.recognizer.predict(image)
        name = self.labels2names_map[label_predicted]
        print(f"FaceRecognizer: [{name}] with confidence: {str(confidence)}")
        return {
            "name": name,
            "confidence": confidence
        }