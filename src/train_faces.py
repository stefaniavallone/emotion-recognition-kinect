import argparse
import cv2
from face_recognition.train import train_recognizer, save_recognizer


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", help="Enter the number of camera source",
                    type=int, default=1)
    ap.add_argument("--kinect", help="Enter True if you want to use kinect",
                    type=bool, default=False)
    args = ap.parse_args()

    train_dir = "../resources/data/faces"
    model_base_path = "../resources/models/faces"
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer, labels_name_map = train_recognizer(recognizer, train_dir)
    save_recognizer(recognizer, labels_name_map, recognizer_name="faces",
                    base_path=model_base_path)
