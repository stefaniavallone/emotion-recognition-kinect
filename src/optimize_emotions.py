import argparse
from emotions.optimization import optimize


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    args = ap.parse_args()

    train_dir = "../resources/data/emotions/train"
    validation_dir = "../resources/data/emotions/validation"
    test_dir = "../resources/data/emotions/test"

    optimize()

