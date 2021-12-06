from collections import deque
from math import ceil
from types import SimpleNamespace

import cv2
import numpy as np

from face_recognition.detector import FaceDetectorModel


class ConfirmationWindowHandler:

    def __init__(self):
        self.cw = dict()

    def add(self, key, value):
        cw = self.get(key)
        cw.append(value)

    def get(self, key):
        if key not in self.cw:
            self.cw[key] = ConfirmationWindow([], maxlen=20)
        return self.cw[key]


class ConfirmationWindow(deque):

    @property
    def most_common(self):
        value = max(set(self), key=self.count)
        return SimpleNamespace(**{"value": value, "count": self.count(value)})


def process_frame(frame, confirmation_window_handler,
                  face_recognition_model, emotion_model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detector = FaceDetectorModel()
    faces = face_detector.predict(gray)

    for face in faces:
        (x, y, w, h) = face["pos"]
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]

        if face_recognition_model is not None:
            face_result = face_recognition_model.predict(roi_gray)
            name = face_result["name"]
            cv2.putText(frame,
                        f'{name}: {str(round(face_result["confidence"], 2))}',
                        (x - 60, y + h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        (0, 0, 255), 2, cv2.LINE_AA)

        if emotion_model is not None:
            cropped_img = np.expand_dims(
                np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            emotion_label = emotion_model.predict(cropped_img)
            if confirmation_window_handler is not None:
                confirmation_window_handler.get(name).append(emotion_label)
                print(f"CW Name {name}: {confirmation_window_handler.get(name)}")
                if confirmation_window_handler.get(name).most_common.count > \
                        ceil(confirmation_window_handler.get(name).maxlen * 0.6):
                    text = emotion_label
                else:
                    text = "Neutral"
            else:
                text = emotion_label
            cv2.putText(frame, text, (x + 20, y - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                            cv2.LINE_AA)

    return frame
