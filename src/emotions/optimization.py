import optuna

from optuna.visualization.matplotlib import *
from optuna.samplers import TPESampler
import joblib

from emotions.evaluate import evaluate_model
from emotions.model import create_model
from emotions.train import train_model


class Objective(object):
    def __init__(self, train_dir, validation_dir, test_dir):
        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.test_dir = test_dir

    def __call__(self, trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        dropout = trial.suggest_uniform('dropout', 0.0, 1.0)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5])

        print(f"Working on parameters: {trial.params}")
        model = create_model(kernel_size=kernel_size,
                             dropout=dropout,
                             lr=lr)
        train_model(model, train_dir=self.train_dir,
                    validation_dir=self.validation_dir,
                    batch_size=batch_size, num_epochs=1,
                    verbose=1)

        # Evaluate the model accuracy on the validation set.
        score = evaluate_model(model, self.test_dir, batch_size)
        return score[1]


def checkpoint_job(study, trial):
    joblib.dump(study, f'study-{trial.number}.pkl')
    print(
        f'Saved study at trial number: {trial.number} with run_score {trial.value} and run_parameters {str(trial.params)}')


def plot_optimization_stats(study):
    plot_contour(study, params=['lr'])
    plot_optimization_history(study)
    plot_parallel_coordinate(study)
    plot_slice(study)


def optimize():
    sampler = TPESampler(seed=10)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(
        Objective(train_dir, validation_dir, test_dir),
        n_trials=1,
        n_jobs=5,
        callbacks=[checkpoint_job])

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    plot_optimization_stats(study)
