import argparse

from emotions.evaluate import evaluate_model
from emotions.model import create_model, save_model
from emotions.train import train_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to train the emotion recognition model.")
    parser.add_argument('--train_dir', type=str, default="../resources/data/emotions/train",
                        help='Path of the train directory')
    parser.add_argument('--test_dir', type=str, default="../resources/data/emotions/test",
                        help='Path of the test directory')
    parser.add_argument('--model_path', type=str, default="../resources/data/emotions/emotion",
                        help='Path of the test directory')
    parser.add_argument('--batch_size', type=int,
                        help='Size of Batch for training (default=32)', default=32)
    parser.add_argument('--epochs', type=int,
                        help='Num of training epochs (default=100)', default=100)
    parser.add_argument('--kernel_size', type=int,
                        help='Size of Kernel for Conv2D layers (default=3)', default=3)
    parser.add_argument('--dropout', type=float,
                        help='Value of dropout (default=0.3106791934814161)',
                        default=0.3106791934814161)
    parser.add_argument('--l2', type=float,
                        help='Value of L2 regularization for Conv2D layers (default=0.04370766874155845)',
                        default=0.04370766874155845)
    parser.add_argument('--lr', type=float,
                        help='Value of learning rate (default=6.454516989719096e-05)', default=6.454516989719096e-05)
    parser.add_argument('--decay', type=float,
                        help='Value of learning rate decay (default=4.461966074951546e-05)',
                        default=4.461966074951546e-05)
    args = parser.parse_args()

    model_base_path = "".join(args.model_path.split("/")[:-1])
    model_name = args.model_path.split("/")[-1]
    model = create_model(kernel_size=args.kernel_size,
                         l2=args.l2,
                         dropout=args.dropout,
                         lr=args.lr,
                         decay=args.decay)
    model = train_model(model, args.train_dir,
                        batch_size=args.batch_size,
                        num_epochs=args.epochs, verbose=1)
    evaluate_model(model, args.test_dir)
    save_model(model, model_name, model_base_path)
