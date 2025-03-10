# portions of below code were generated with the assistance of AI
import time
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     RandomizedSearchCV, learning_curve,
                                     validation_curve)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (f1_score, classification_report,
                             roc_auc_score, accuracy_score)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTEENN


import optuna
from optuna.pruners import MedianPruner
from tabulate import tabulate
from scipy.stats import randint, loguniform

warnings.filterwarnings("ignore")
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# -----------------------
# FILE PATHS
# -----------------------
marketing_campaign_path = 'marketing_campaign.csv'
spotify_path = 'spotify-2023.csv'


# ==================================
# 1. HELPER FUNCTIONS
# ==================================
def label_encode_categorical(df):
    """
    Label-encodes all categorical (object) features in the DataFrame.
    """
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    return df


def preprocess_marketing_campaign(file_path):
    """
    Preprocesses the marketing campaign data.
    """
    full_data = pd.read_csv(file_path, delimiter='\t', encoding='latin1').dropna()
    y = full_data['Response'].astype(int)
    X = full_data.drop(columns=['ID', 'Dt_Customer', 'Response'], errors='ignore')
    X = label_encode_categorical(X)
    return X, y


def preprocess_spotify(file_path):
    """
    Preprocesses the Spotify data.
    """
    data = pd.read_csv(file_path, encoding='latin1')
    data['streams'] = pd.to_numeric(data['streams'], errors='coerce')
    data.dropna(subset=['streams'], inplace=True)
    data['streams_high'] = (data['streams'] > data['streams'].median()).astype(int)
    X = data.drop(columns=['streams', 'streams_high', 'track_name', 'artist(s)_name'], errors='ignore')
    X = label_encode_categorical(X)
    y = data['streams_high']
    return X, y


# ===========================================
# ADDITIONAL PLOTTING FUNCTIONS FOR ERROR CURVES
# ===========================================
def plot_learning_curve_error(model, X, y, title, filename, cv=3):
    """
    Plots training and testing *error* (1 - F1_micro) as a function of training set size.
    Note: 'CV Error' is computed from cross-validation (validation) folds.
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, train_sizes=np.linspace(0.1, 1.0, 5),
        cv=cv, scoring='f1_micro', n_jobs=-1
    )
    # Convert F1_micro to error = 1 - F1_micro
    train_errors = 1.0 - train_scores
    test_errors = 1.0 - test_scores

    train_errors_mean = np.mean(train_errors, axis=1)
    test_errors_mean = np.mean(test_errors, axis=1)

    plt.figure(figsize=(8, 6))
    plt.plot(train_sizes, train_errors_mean, 'o-', color='blue', label='Training Error')
    plt.plot(train_sizes, test_errors_mean, 'o-', color='orange', label='CV Error')
    plt.title(title)
    plt.xlabel('Training Samples')
    plt.ylabel('Error (1 - F1_micro)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_model_complexity_error(estimator, X, y, param_name, param_range, title, filename, cv=3,
                                force_categorical=False):
    """
    Plots the validation error (1 - F1_micro) as a function of a chosen hyperparameter range.
    This creates a 'validation curve'.

    - If `param_range` is numerical and not forced categorical, it is plotted as a continuous line graph.
    - If `param_range` is categorical or forced to be categorical, it is plotted as a dot plot with category labels on the x-axis.

    The y-axis represents the validation error (1 - F1_micro) computed from cross-validation.
    """
    if param_range is None or len(param_range) == 0:
        print(f"Skipping plot for {param_name} - param_range is empty or None")
        return

    # Determine if hyperparameter is categorical.
    is_categorical = force_categorical or (
                isinstance(param_range[0], str) or any(isinstance(p, str) for p in param_range))

    # For continuous hyperparameters, keep the values as numbers;
    # for categorical ones, convert to strings.
    if is_categorical:
        plot_x = [str(p) for p in param_range]
    else:
        plot_x = param_range

    # Compute validation curve scores (we only use the CV scores here)
    _, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        scoring='f1_micro', cv=cv, n_jobs=-1
    )
    # Convert F1_micro to error
    test_errors = 1.0 - test_scores
    test_errors_mean = np.mean(test_errors, axis=1)

    plt.figure(figsize=(8, 6))
    if is_categorical:
        x_indices = np.arange(len(plot_x))
        plt.scatter(x_indices, test_errors_mean, color='orange', label='Validation Error', marker='s', s=50)
        plt.xticks(x_indices, plot_x, rotation=45, ha='right')
    else:
        plt.plot(plot_x, test_errors_mean, 'o-', color='orange', label='Validation Error')
        # Apply log scale if appropriate for known hyperparameters
        if any(isinstance(p, (int, float)) for p in plot_x) and (
                param_name in ['C', 'alpha', 'learning_rate_init', 'gamma']):
            plt.xscale('log')
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('Validation Error (1 - F1_micro)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_nn_learning_curve_error(nn_params, X_train, y_train, X_val, y_val, title, filename, epochs=150):
    """
    Plots the learning curve for a neural network (NN) with epochs on the x-axis.
    This function trains the network iteratively (using partial_fit with warm_start)
    and computes training and validation errors (1 - F1_micro) after each epoch.
    """
    # Initialize a new MLPClassifier (used as an NN) with max_iter=1 and warm_start enabled.
    model = MLPClassifier(
        hidden_layer_sizes=nn_params['hidden_layer_sizes'],
        activation=nn_params['activation'],
        alpha=nn_params['alpha'],
        learning_rate_init=nn_params['learning_rate_init'],
        max_iter=1,
        warm_start=True,
        random_state=RANDOM_SEED
    )
    training_errors = []
    validation_errors = []
    classes = np.unique(y_train)
    for epoch in range(epochs):
        if epoch == 0:
            model.partial_fit(X_train, y_train, classes=classes)
        else:
            model.partial_fit(X_train, y_train)
        # Compute training error
        y_train_pred = model.predict(X_train)
        train_f1 = f1_score(y_train, y_train_pred, average='micro')
        training_errors.append(1 - train_f1)
        # Compute validation error
        y_val_pred = model.predict(X_val)
        val_f1 = f1_score(y_val, y_val_pred, average='micro')
        validation_errors.append(1 - val_f1)
    epochs_range = np.arange(1, epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, training_errors, 'o-', color='blue', label='Training Error')
    plt.plot(epochs_range, validation_errors, 'o-', color='orange', label='Validation Error')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Error (1 - F1_micro)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


# =======================================
# 2. RANDOM SEARCH FOR KNN & SVM (F1 & AUC)
# =======================================
def search_knn_hyperparameters(X_train, y_train):
    """
    Uses RandomizedSearchCV to find the best KNN hyperparameters.
    Searches over multiple metrics, returning both F1 & AUC in 'cv_results_'.
    """
    knn_param_dist = {
        'n_neighbors': randint(1, 31),
        'weights': ['uniform', 'distance'],
        'metric': ['manhattan', 'euclidean', 'chebyshev']
    }

    knn_scoring = {
        'F1': 'f1',
        'AUC': 'roc_auc'
    }

    knn_search = RandomizedSearchCV(
        estimator=KNeighborsClassifier(),
        param_distributions=knn_param_dist,
        n_iter=20,
        cv=3,
        scoring=knn_scoring,
        refit='F1',
        n_jobs=-1,
        random_state=RANDOM_SEED,
        return_train_score=True
    )
    knn_search.fit(X_train, y_train)
    return knn_param_dist, knn_search.best_params_, knn_search.best_estimator_, knn_search.cv_results_


def search_svm_hyperparameters(X_train, y_train):
    """
    Uses RandomizedSearchCV to find the best SVM hyperparameters.
    Searches over multiple metrics, returning both F1 & AUC in 'cv_results_'.
    """
    svm_param_dist = {
        'kernel': ['linear', 'rbf', 'poly'],
        'class_weight': [None, 'balanced'],
        'C': loguniform(1e-3, 1e2),
        'gamma': loguniform(1e-4, 0.1)
    }

    svm_scoring = {
        'F1': 'f1',
        'AUC': 'roc_auc'
    }

    svm_search = RandomizedSearchCV(
        estimator=SVC(probability=True),
        param_distributions=svm_param_dist,
        n_iter=20,
        cv=3,
        scoring=svm_scoring,
        refit='F1',
        n_jobs=-1,
        random_state=RANDOM_SEED,
        return_train_score=True
    )
    svm_search.fit(X_train, y_train)
    return svm_param_dist, svm_search.best_params_, svm_search.best_estimator_, svm_search.cv_results_


# ==========================================
# 3. OPTUNA FOR NN (WITH CLASS IMBALANCE)
# ==========================================
def optimize_nn_parameters(X_train, y_train, X_val, y_val):
    """
    Optuna-based hyperparameter search for NN, optimizing F1 while storing AUC
    in trial.user_attrs to retrieve later.
    """

    def objective(trial):
        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic'])
        hidden_choice = trial.suggest_categorical('hidden_layer_sizes', ['50', '100', '150'])
        hidden_layer = int(hidden_choice)
        alpha = trial.suggest_float('alpha', 1e-6, 1e-2, log=True)
        lr = trial.suggest_float('learning_rate_init', 1e-5, 1e-2, log=True)

        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer,
            activation=activation,
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=150,
            early_stopping=True,
            random_state=RANDOM_SEED
        )
        model.fit(X_train, y_train)
        y_val_pred = model.predict(X_val)
        y_val_prob = model.predict_proba(X_val)[:, 1]

        f1 = f1_score(y_val, y_val_pred)
        auc = roc_auc_score(y_val, y_val_prob)
        trial.set_user_attr('auc', auc)

        return f1

    study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    study.optimize(objective, n_trials=20)

    best = study.best_params
    best['hidden_layer_sizes'] = int(best['hidden_layer_sizes'])

    final_model = MLPClassifier(
        hidden_layer_sizes=best['hidden_layer_sizes'],
        activation=best['activation'],
        alpha=best['alpha'],
        learning_rate_init=best['learning_rate_init'],
        max_iter=150,
        early_stopping=True,
        random_state=RANDOM_SEED
    )
    final_model.fit(X_train, y_train)
    return best, final_model, study


# =======================================
# 4. THRESHOLD OPTIMIZATION
# =======================================
def find_best_threshold(y_prob, y_true):
    thresholds = np.linspace(0, 1, 101)
    f1_scores = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx], f1_scores[best_idx]


def find_cv_best_threshold(model, X, y, folds=5):
    """
    Averages the best threshold across folds by maximizing F1 in each fold.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_SEED)
    thresholds = []
    for train_index, val_index in skf.split(X, y):
        y_train_fold = y.iloc[train_index].reset_index(drop=True)
        y_val = y.iloc[val_index].reset_index(drop=True)
        X_train_fold, X_val_fold = X[train_index], X[val_index]
        model.fit(X_train_fold, y_train_fold)
        y_prob_val = model.predict_proba(X_val_fold)[:, 1]
        t, _ = find_best_threshold(y_prob_val, y_val)
        thresholds.append(t)
    return np.mean(thresholds)


# =============================================
# 5. MAIN TRAINING & EVALUATION PIPELINE
# =============================================
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name):
    output = []
    output.append(f"\n=== {dataset_name} ===\n")

    # Create a directory to save results if it doesn't exist
    results_dir = 'results'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # -------------------
    # 5.1 KNN
    # -------------------
    start_knn = time.time()  # Start timing KNN training & hyperparameter search
    knn_param_dist, knn_best_params, knn_model, knn_cv_results = search_knn_hyperparameters(X_train, y_train)
    end_knn = time.time()  # End timing KNN block
    knn_elapsed = end_knn - start_knn  # Total elapsed time for KNN (in seconds)

    # Report timing in the output
    output.append(f"KNN Training & Hyperparameter Search Time: {knn_elapsed:.3f} seconds")

    knn_hyperparams_considered = {
        'n_neighbors': '1-30',
        'weights': knn_param_dist['weights'],
        'metric': knn_param_dist['metric']
    }
    output.append("KNN Hyperparameters Considered:")
    for param, values in knn_hyperparams_considered.items():
        output.append(f"  - {param}: {values}")
    output.append("")

    knn_thr_cv = find_cv_best_threshold(knn_model, X_val, y_val)
    y_prob_knn = knn_model.predict_proba(X_test)[:, 1]
    y_pred_knn = (y_prob_knn >= knn_thr_cv).astype(int)

    output.append(f"KNN - Best Params: {knn_best_params}")
    output.append(f"KNN Threshold (CV-based): {knn_thr_cv:.3f}")
    output.append(f"KNN Test F1: {f1_score(y_test, y_pred_knn):.4f} | AUC: {roc_auc_score(y_test, y_prob_knn):.4f}")
    output.append(classification_report(y_test, y_pred_knn))

    knn_results_df = pd.DataFrame(knn_cv_results)
    knn_summary = knn_results_df[
        ['param_n_neighbors', 'param_weights', 'param_metric',
         'mean_test_F1', 'mean_test_AUC']
    ]
    knn_summary = knn_summary.sort_values(by='mean_test_F1', ascending=False)
    knn_csv_path = os.path.join(results_dir, f"{dataset_name.replace(' ', '_')}_KNN_Hyperparameter_Search.csv")
    knn_summary.to_csv(knn_csv_path, index=False)
    output.append(f"\nKNN Hyperparameter Search Results saved to {knn_csv_path}")
    output.append("\nKNN Hyperparameter Search Results (Top 15):")
    output.append(tabulate(knn_summary.head(15), headers='keys', tablefmt='pretty', showindex=False))

    # --- Learning curve for KNN (training vs test error) ---
    knn_learning_curve_path = os.path.join(results_dir, f"{dataset_name.replace(' ', '_')}_KNN_LearningCurve_Error.png")
    plot_learning_curve_error(
        model=KNeighborsClassifier(**knn_best_params),
        X=pd.DataFrame(X_train),
        y=y_train,
        title=f"{dataset_name} - KNN Learning Curve (k = {knn_best_params['n_neighbors']})",
        filename=knn_learning_curve_path,
        cv=3
    )
    output.append(f"KNN Learning Curve (Error) Plot saved to {knn_learning_curve_path}\n")

    # --- Explicit Validation Curves for KNN ---
    # Validation curve for 'n_neighbors' (continuous)
    knn_val_curve_neighbors_path = os.path.join(results_dir,
                                                f"{dataset_name.replace(' ', '_')}_KNN_ValidationCurve_n_neighbors.png")
    plot_model_complexity_error(
        estimator=KNeighborsClassifier(**knn_best_params),
        X=pd.DataFrame(X_train),
        y=y_train,
        param_name='n_neighbors',
        param_range=list(range(1, 31)),
        title=f"{dataset_name} - KNN Validation Curve (n_neighbors)",
        filename=knn_val_curve_neighbors_path,
        cv=3
    )
    output.append(f"KNN Validation Curve (n_neighbors) Plot saved to {knn_val_curve_neighbors_path}")

    # Validation curve for 'metric' (discrete)
    knn_val_curve_metric_path = os.path.join(results_dir,
                                             f"{dataset_name.replace(' ', '_')}_KNN_ValidationCurve_metric.png")
    plot_model_complexity_error(
        estimator=KNeighborsClassifier(**knn_best_params),
        X=pd.DataFrame(X_train),
        y=y_train,
        param_name='metric',
        param_range=knn_param_dist['metric'],
        title=f"{dataset_name} - KNN Validation Curve (metric)",
        filename=knn_val_curve_metric_path,
        cv=3,
        force_categorical=True
    )
    output.append(f"KNN Validation Curve (metric) Plot saved to {knn_val_curve_metric_path}\n")

    # -------------------
    # 5.2 SVM
    # -------------------
    start_svm = time.time()  # Start timing SVM training & hyperparameter search
    svm_param_dist, svm_best_params, svm_model, svm_cv_results = search_svm_hyperparameters(X_train, y_train)
    end_svm = time.time()  # End timing SVM block
    svm_elapsed = end_svm - start_svm  # Total elapsed time for SVM
    output.append(f"SVM Training & Hyperparameter Search Time: {svm_elapsed:.3f} seconds")

    svm_hyperparams_considered = {
        'kernel': svm_param_dist['kernel'],
        'class_weight': svm_param_dist['class_weight'],
        'C': '1e-3 to 1e2 (log scale)',
        'gamma': '1e-4 to 0.1 (log scale)'
    }
    output.append("SVM Hyperparameters Considered:")
    for param, values in svm_hyperparams_considered.items():
        output.append(f"  - {param}: {values}")
    output.append("")

    svm_thr_cv = find_cv_best_threshold(svm_model, X_val, y_val)
    y_prob_svm = svm_model.predict_proba(X_test)[:, 1]
    y_pred_svm = (y_prob_svm >= svm_thr_cv).astype(int)

    output.append(f"SVM - Best Params: {svm_best_params}")
    output.append(f"SVM Threshold (CV-based): {svm_thr_cv:.3f}")
    output.append(f"SVM Test F1: {f1_score(y_test, y_pred_svm):.4f} | AUC: {roc_auc_score(y_test, y_prob_svm):.4f}")
    output.append(classification_report(y_test, y_pred_svm))

    svm_results_df = pd.DataFrame(svm_cv_results)
    svm_summary = svm_results_df[
        ['param_kernel', 'param_class_weight', 'param_C', 'param_gamma',
         'mean_test_F1', 'mean_test_AUC']
    ]
    svm_summary = svm_summary.sort_values(by='mean_test_F1', ascending=False)
    svm_csv_path = os.path.join(results_dir, f"{dataset_name.replace(' ', '_')}_SVM_Hyperparameter_Search.csv")
    svm_summary.to_csv(svm_csv_path, index=False)
    output.append(f"\nSVM Hyperparameter Search Results saved to {svm_csv_path}")
    output.append("\nSVM Hyperparameter Search Results (Top 15):")
    output.append(tabulate(svm_summary.head(15), headers='keys', tablefmt='pretty', showindex=False))

    # --- Learning curve for SVM (training vs test error) ---
    svm_learning_curve_path = os.path.join(results_dir, f"{dataset_name.replace(' ', '_')}_SVM_LearningCurve_Error.png")
    best_svc = SVC(probability=True,
                   kernel=svm_best_params['kernel'],
                   class_weight=svm_best_params['class_weight'],
                   C=svm_best_params['C'],
                   gamma=svm_best_params['gamma'],
                   random_state=RANDOM_SEED)
    plot_learning_curve_error(
        model=best_svc,
        X=pd.DataFrame(X_train),
        y=y_train,
        title=f"{dataset_name} - SVM Learning Curve (Kernel: {svm_best_params['kernel']})",
        filename=svm_learning_curve_path,
        cv=3
    )
    output.append(f"SVM Learning Curve (Error) Plot saved to {svm_learning_curve_path}\n")

    # --- Explicit Validation Curves for SVM ---
    # Validation curve for 'kernel' (discrete)
    svm_val_curve_kernel_path = os.path.join(results_dir,
                                             f"{dataset_name.replace(' ', '_')}_SVM_ValidationCurve_kernel.png")
    plot_model_complexity_error(
        estimator=SVC(probability=True, random_state=RANDOM_SEED),
        X=pd.DataFrame(X_train),
        y=y_train,
        param_name='kernel',
        param_range=svm_param_dist['kernel'],
        title=f"{dataset_name} - SVM Validation Curve (kernel)",
        filename=svm_val_curve_kernel_path,
        cv=3,
        force_categorical=True
    )
    output.append(f"SVM Validation Curve (kernel) Plot saved to {svm_val_curve_kernel_path}")

    # Validation curve for 'C' (continuous)
    svm_val_curve_C_path = os.path.join(results_dir, f"{dataset_name.replace(' ', '_')}_SVM_ValidationCurve_C.png")
    svm_C_range = np.logspace(-3, 2, 10)
    plot_model_complexity_error(
        estimator=SVC(probability=True, kernel=svm_best_params['kernel'], class_weight=svm_best_params['class_weight'],
                      gamma=svm_best_params['gamma'], random_state=RANDOM_SEED),
        X=pd.DataFrame(X_train),
        y=y_train,
        param_name='C',
        param_range=svm_C_range,
        title=f"{dataset_name} - SVM Validation Curve (C)",
        filename=svm_val_curve_C_path,
        cv=3
    )
    output.append(f"SVM Validation Curve (C) Plot saved to {svm_val_curve_C_path}")

    # Validation curve for 'gamma' (continuous)
    svm_val_curve_gamma_path = os.path.join(results_dir,
                                            f"{dataset_name.replace(' ', '_')}_SVM_ValidationCurve_gamma.png")
    svm_gamma_range = np.logspace(-4, -1, 10)
    plot_model_complexity_error(
        estimator=SVC(probability=True, kernel=svm_best_params['kernel'], class_weight=svm_best_params['class_weight'],
                      C=svm_best_params['C'], random_state=RANDOM_SEED),
        X=pd.DataFrame(X_train),
        y=y_train,
        param_name='gamma',
        param_range=svm_gamma_range,
        title=f"{dataset_name} - SVM Validation Curve (gamma)",
        filename=svm_val_curve_gamma_path,
        cv=3
    )
    output.append(f"SVM Validation Curve (gamma) Plot saved to {svm_val_curve_gamma_path}\n")

    # -------------------
    # 5.3 Neural Network (NN)
    # -------------------
    start_nn = time.time()  # Start timing NN hyperparameter optimization and training
    nn_best_params, nn_model, nn_study = optimize_nn_parameters(X_train, y_train, X_val, y_val)
    end_nn = time.time()  # End timing NN block
    nn_elapsed = end_nn - start_nn  # Total elapsed time for NN
    output.append(f"Neural Network Training & Hyperparameter Search Time: {nn_elapsed:.3f} seconds")

    nn_hyperparams_considered = {
        'activation': ['relu', 'tanh', 'logistic'],
        'hidden_layer_sizes': '50, 100, 150',
        'alpha': '1e-6 to 1e-2 (log scale)',
        'learning_rate_init': '1e-5 to 1e-2 (log scale)'
    }
    output.append("Neural Network Hyperparameters Considered:")
    for param, values in nn_hyperparams_considered.items():
        output.append(f"  - {param}: {values}")
    output.append("")

    nn_thr_cv = find_cv_best_threshold(nn_model, X_val, y_val)
    y_prob_nn = nn_model.predict_proba(X_test)[:, 1]
    y_pred_nn = (y_prob_nn >= nn_thr_cv).astype(int)

    output.append(f"Neural Network - Best Params: {nn_best_params}")
    output.append(f"NN Threshold (CV-based): {nn_thr_cv:.3f}")
    output.append(f"NN Test F1: {f1_score(y_test, y_pred_nn):.4f} | AUC: {roc_auc_score(y_test, y_prob_nn):.4f}")
    output.append(classification_report(y_test, y_pred_nn))

    nn_trials = nn_study.trials_dataframe()
    nn_summary = nn_trials.rename(columns={'value': 'f1_score'})
    if 'user_attrs_auc' in nn_summary.columns:
        nn_summary.rename(columns={'user_attrs_auc': 'auc'}, inplace=True)
    else:
        nn_summary['auc'] = np.nan
    nn_summary = nn_summary[
        ['params_activation', 'params_hidden_layer_sizes',
         'params_alpha', 'params_learning_rate_init',
         'f1_score', 'auc']
    ].sort_values(by='f1_score', ascending=False)

    nn_csv_path = os.path.join(results_dir, f"{dataset_name.replace(' ', '_')}_NN_Hyperparameter_Search.csv")
    nn_summary.to_csv(nn_csv_path, index=False)
    output.append(f"\nNeural Network Hyperparameter Search Results saved to {nn_csv_path}")
    output.append("\nNeural Network Hyperparameter Search Results (Top 15):")
    output.append(tabulate(nn_summary.head(15), headers='keys', tablefmt='pretty', showindex=False))

    # --- Learning curve for NN (training vs validation error by epoch) ---
    nn_learning_curve_path = os.path.join(results_dir, f"{dataset_name.replace(' ', '_')}_NN_LearningCurve_Error.png")
    plot_nn_learning_curve_error(
        nn_params=nn_best_params,
        X_train=pd.DataFrame(X_train),
        y_train=y_train,
        X_val=pd.DataFrame(X_val),
        y_val=y_val,
        title=f"{dataset_name} - NN Learning Curve (Epochs) (Hidden: {nn_best_params['hidden_layer_sizes']}, Activation: {nn_best_params['activation']})",
        filename=nn_learning_curve_path,
        epochs=150
    )
    output.append(f"NN Learning Curve (Epochs) Error Plot saved to {nn_learning_curve_path}\n")

    # --- Explicit Validation Curves for NN ---
    # Validation curve for 'activation' (discrete)
    nn_val_curve_activation_path = os.path.join(results_dir,
                                                f"{dataset_name.replace(' ', '_')}_NN_ValidationCurve_activation.png")
    plot_model_complexity_error(
        estimator=MLPClassifier(random_state=RANDOM_SEED),
        X=pd.DataFrame(X_train),
        y=y_train,
        param_name='activation',
        param_range=nn_hyperparams_considered['activation'],
        title=f"{dataset_name} - NN Validation Curve (activation)",
        filename=nn_val_curve_activation_path,
        cv=3,
        force_categorical=True
    )
    output.append(f"NN Validation Curve (activation) Plot saved to {nn_val_curve_activation_path}")

    # Validation curve for 'hidden_layer_sizes' (discrete)
    nn_val_curve_hidden_path = os.path.join(results_dir,
                                            f"{dataset_name.replace(' ', '_')}_NN_ValidationCurve_hidden_layer_sizes.png")
    nn_hidden_range = [50, 100, 150]
    plot_model_complexity_error(
        estimator=MLPClassifier(random_state=RANDOM_SEED),
        X=pd.DataFrame(X_train),
        y=y_train,
        param_name='hidden_layer_sizes',
        param_range=nn_hidden_range,
        title=f"{dataset_name} - NN Validation Curve (hidden_layer_sizes)",
        filename=nn_val_curve_hidden_path,
        cv=3,
        force_categorical=True
    )
    output.append(f"NN Validation Curve (hidden_layer_sizes) Plot saved to {nn_val_curve_hidden_path}\n")

    return '\n'.join(output)


# =========================================
# 6. EXECUTION
# =========================================
if __name__ == '__main__':
    # Preprocess Data
    marketing_X, marketing_y = preprocess_marketing_campaign(marketing_campaign_path)
    spotify_X, spotify_y = preprocess_spotify(spotify_path)

    # ---------------------------
    # ADDED CODE: Print Summary Statistics for Each Dataset
    # ---------------------------
    print("=== Marketing Campaign Dataset Summary Statistics ===")
    print("Shape:", marketing_X.shape)
    print("Feature Summary:")
    print(marketing_X.describe(include='all'))
    print("Target Distribution:")
    print(marketing_y.value_counts(normalize=True))
    print("\n")

    print("=== Spotify Dataset Summary Statistics ===")
    print("Shape:", spotify_X.shape)
    print("Feature Summary:")
    print(spotify_X.describe(include='all'))
    print("Target Distribution:")
    print(spotify_y.value_counts(normalize=True))
    print("\n")


    # ---------------------------
    # End of ADDED CODE
    # ---------------------------

    def split_data(X, y):
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=RANDOM_SEED
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_SEED
        )
        return X_train, X_val, X_test, y_train, y_val, y_test


    mk_X_train, mk_X_val, mk_X_test, mk_y_train, mk_y_val, mk_y_test = split_data(marketing_X, marketing_y)
    sp_X_train, sp_X_val, sp_X_test, sp_y_train, sp_y_val, sp_y_test = split_data(spotify_X, spotify_y)

    mk_X_train, mk_y_train = SMOTEENN(random_state=RANDOM_SEED).fit_resample(mk_X_train, mk_y_train)

    scaler_mk = StandardScaler()
    mk_X_train = scaler_mk.fit_transform(mk_X_train)
    mk_X_val = scaler_mk.transform(mk_X_val)
    mk_X_test = scaler_mk.transform(mk_X_test)

    scaler_sp = StandardScaler()
    sp_X_train = scaler_sp.fit_transform(sp_X_train)
    sp_X_val = scaler_sp.transform(sp_X_val)
    sp_X_test = scaler_sp.transform(sp_X_test)

    # Train & Evaluate on Marketing
    marketing_result = train_and_evaluate(
        mk_X_train, mk_y_train, mk_X_val, mk_y_val, mk_X_test, mk_y_test,
        "Marketing Campaign"
    )
    print(marketing_result)

    # Train & Evaluate on Spotify
    spotify_result = train_and_evaluate(
        sp_X_train, sp_y_train, sp_X_val, sp_y_val, sp_X_test, sp_y_test,
        "Spotify Dataset"
    )
    print(spotify_result)
