import optuna
from optuna.samplers import TPESampler
import joblib

from emotions.evaluate import evaluate_model
from emotions.model import create_model
from emotions.train import train_model


class Objective(object):
    def __init__(self, train_dir, test_dir):
        self.train_dir = train_dir
        self.test_dir = test_dir

    def __call__(self, trial):
        lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
        decay = trial.suggest_loguniform('decay', 1e-7, 1e-4)
        dropout = trial.suggest_uniform('dropout', 0.10, 0.50)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5])
        l2 = trial.suggest_uniform('l2', 0.01, 0.05)

        print(f"Working on parameters: {trial.params}")
        model = create_model(kernel_size=kernel_size,
                             l2=l2,
                             dropout=dropout,
                             lr=lr, decay=decay)
        train_model(model, train_dir=self.train_dir,
                    batch_size=batch_size,
                    num_epochs=100,
                    verbose=1)

        # Evaluate the model accuracy on the validation set.
        score = evaluate_model(model, self.test_dir, batch_size)
        return score[1]


def checkpoint_job(study, trial):
    joblib.dump(study, f'study-{trial.number}.pkl')
    print(
        f'Saved study at trial number: {trial.number} with run_score {trial.value} and run_parameters {str(trial.params)}')


def optimize(train_dir, test_dir, n_trials=50):
    sampler = TPESampler(seed=10)
    study = optuna.create_study(sampler=sampler, direction='maximize')
    study.optimize(
        Objective(train_dir, test_dir),
        n_trials=n_trials,
        n_jobs=4,
        callbacks=[checkpoint_job])

    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

