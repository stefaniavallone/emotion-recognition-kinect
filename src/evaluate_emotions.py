import argparse

from tensorflow import keras

from emotions.evaluate import evaluate_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for evaluate the trained emotions model.")
    parser.add_argument('--model_path', type=str, default="../resources/models/emotions/emotion",
                        help='Path of the test directory')
    parser.add_argument('--test_dir', type=str, default="../resources/data/emotions/test",
                        help='Path of the model to evaluate')
    args = parser.parse_args()

    model = keras.models.load_model(args.model_path)
    score = evaluate_model(model, args.test_dir)
    print(score)
