import logging
import sys

import optuna
import sklearn.datasets
import sklearn.model_selection
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

def objective(trial):
    wine = sklearn.datasets.load_wine()
    X, y = wine.data, wine.target
    X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(
        X, y, test_size=0.25, random_state=0
    )
    classes = list(set(y))
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    model_name = trial.suggest_categorical("model", ["MultinomialNB", "MLPClassifier"])

    if model_name == "MultinomialNB":
        alpha = trial.suggest_float("alpha", 1e-5, 1e1, log=True)
        clf = MultinomialNB(alpha=alpha)


        for step in range(100):
            clf.partial_fit(X_train, y_train, classes=classes)
            intermediate_value = 1.0 - clf.score(X_valid, y_valid)
            trial.report(intermediate_value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()

    else:
        hidden_layer_sizes = trial.suggest_int("hidden_layer_sizes", 10, 100, log=True)
        activation = trial.suggest_categorical("activation", ["relu", "tanh", "logistic"])
        solver = trial.suggest_categorical("solver", ["adam", "sgd"])
        alpha_mlp = trial.suggest_float("alpha_mlp", 1e-5, 1e-1, log=True)

        clf = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha_mlp, random_state=0, max_iter=500) # Увеличиваем max_iter
        for step in range(100):
            clf.partial_fit(X_train_scaled, y_train, classes=classes)
            intermediate_value = 1.0 - clf.score(X_valid_scaled, y_valid)
            trial.report(intermediate_value, step)
            if trial.should_prune():
                raise optuna.TrialPruned()


    return 1.0 - clf.score(X_valid_scaled if model_name == "MLPClassifier" else X_valid, y_valid)

# Два разных семплера
sampler1 = optuna.samplers.TPESampler()
sampler2 = optuna.samplers.RandomSampler()

# Два разных прунера
pruner1 = optuna.pruners.HyperbandPruner()
pruner2 = optuna.pruners.MedianPruner()

# Оптимизация с sampler1 и pruner1
logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
study1 = optuna.create_study(direction="minimize", sampler=sampler1, pruner=pruner1, storage='postgresql://postgres:saSha17122002@localhost/car_app')
study1.optimize(objective, n_trials=20)
print(f"Study 1 best params: {study1.best_params}, best value: {study1.best_value}")
