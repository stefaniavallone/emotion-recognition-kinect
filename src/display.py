import argparse

from emotions.model import labels
from emotions.onnx_predictor import OnnxPredictor
from face_recognition.predictor import FaceRecognitionModel
from video import webcam, kinect
from video.process_frame import ConfirmationWindowHandler


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", help="Enter the number of camera source",
                    type=int, default=1)
    ap.add_argument("--kinect", help="Enter True if you want to use kinect",
                    type=bool, default=False)
    args = ap.parse_args()

    confirmation_window_handler = ConfirmationWindowHandler()
    face_recognition_model = FaceRecognitionModel()
    emotion_model = OnnxPredictor(labels=labels, model_path="../resources/models/emotions/emotions.onnx")
    if args.kinect:
        kinect.process_and_display(confirmation_window_handler=confirmation_window_handler,
                                   face_recognition_model=face_recognition_model,
                                   emotion_model=emotion_model)
    else:
        webcam.process_and_display(camera_source=args.camera,
                                   confirmation_window_handler=confirmation_window_handler,
                                   face_recognition_model=face_recognition_model,
                                   emotion_model=emotion_model)

