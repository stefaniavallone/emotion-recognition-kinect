import argparse
import os
from emotions.model import create_model, save_model
from emotions.train import train_model

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", help="Enter the number of camera source",
                    type=int, default=1)
    ap.add_argument("--kinect", help="Enter True if you want to use kinect",
                    type=bool, default=False)
    args = ap.parse_args()

    data_base_path = "../resources/data/emotions"
    model_base_path = "../resources/models/emotions"
    model = create_model()
    train_dir = os.path.join(data_base_path, "train")
    model = train_model(model, train_dir, num_epochs=1, verbose=1)
    # evaluate_model()
    save_model(model, "emotions2", model_base_path)
