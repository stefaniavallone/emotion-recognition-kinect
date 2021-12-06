import argparse
from emotions.evaluate import evaluate_model
from emotions.model import create_model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    model_path = "../resources/models/emotions/emotions.h5"
    test_dir = "../resources/data/emotions/test"
    model = create_model()
    model.load_weights(model_path)
    score = evaluate_model(model, test_dir)
    print(score)
