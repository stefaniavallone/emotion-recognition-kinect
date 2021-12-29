import argparse
from emotions.optimization import optimize


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script to run the hyperparameters tuning for the emotions model.')
    parser.add_argument('--train_dir', type=str, default="../resources/data/emotions/train",
                        help='Path of the train directory')
    parser.add_argument('--test_dir', type=str, default="../resources/data/emotions/test",
                        help='Path of the test directory')
    parser.add_argument('--n_trials', type=int,
                        help='Number of optimization experiments (default=50)', default=50)
    args = parser.parse_args()

    optimize(args.train_dir, args.test_dir, n_trials=args.n_trials)

