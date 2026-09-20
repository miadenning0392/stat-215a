from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, make_scorer
from sklearn.tree import plot_tree

from clean import clean_data, load_data

BASE_DIR = Path(__file__).resolve().parent.parent

RANDOM_STATE = 42

DROP_COLUMNS = [
    "patient_id",
    "clinically_important_tbi",
    "ciTBI_1",
    "ciTBI_2",
    "ciTBI_3",
    "ciTBI_4",
    "ct_done",
]


def sensitivity(y_true, y_pred):
    """Calculate sensitivity (true positive rate) from true and predicted labels.

    Args:
        y_true: Array-like of true binary labels (0 or 1).
        y_pred: Array-like of predicted binary labels (0 or 1).

    Returns:
        float: Sensitivity score (TP / (TP + FN)).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return tp / (tp + fn)


def npv(y_true, y_pred):
    """Calculate negative predictive value from true and predicted labels.

    Args:
        y_true: Array-like of true binary labels (0 or 1).
        y_pred: Array-like of predicted binary labels (0 or 1).

    Returns:
        float: Negative predictive value (TN / (TN + FN)).
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return tn / (tn + fn)


def clinical_score(y_true, y_pred):
    """Calculate the clinical score as the harmonic mean of sensitivity and NPV.

    Args:
        y_true: Array-like of true binary labels (0 or 1).
        y_pred: Array-like of predicted binary labels (0 or 1).

    Returns:
        float: Clinical score (harmonic mean of sensitivity and NPV). Returns 0
               if both metrics sum to 0.
    """
    sens = sensitivity(y_true, y_pred)
    neg_pred_value = npv(y_true, y_pred)

    if sens + neg_pred_value == 0:
        return 0

    return 2 * sens * neg_pred_value / (sens + neg_pred_value)


clinical_scorer = make_scorer(clinical_score)


def prepare_data(train_df, test_df):
    """Prepare training and test data for modeling.

    Args:
        train_df: DataFrame with training data.
        test_df: DataFrame with test data.

    Returns:
        tuple: (X_train, X_test, y_train, y_test) where X_train and X_test are
               DataFrames with aligned numeric features, and y_train and y_test
               are Series with the target variable.
    """
    y_train = train_df["clinically_important_tbi"]
    y_test = test_df["clinically_important_tbi"]

    X_train = train_df.drop(columns=DROP_COLUMNS)
    X_test = test_df.drop(columns=DROP_COLUMNS)

    # convert categorical variables to numeric dummy variables. missing categorical variables are kept as their own category.
    X_train = pd.get_dummies(X_train, dummy_na=True)
    X_test = pd.get_dummies(X_test, dummy_na=True)

    # make sure train and test have the same columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    # convert to numeric values
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)

    return X_train, X_test, y_train, y_test


def fit_logistic_regression(X_train, y_train):
    """Fit a logistic regression model with hyperparameter tuning.

    Builds a pipeline with median imputation and logistic regression, then
    performs grid search with 5-fold stratified cross-validation to find
    optimal hyperparameters (C and class_weight) using clinical score.

    Args:
        X_train: DataFrame with training features.
        y_train: Series with training target variable.

    Returns:
        GridSearchCV: Fitted grid search object with best estimator accessible
                      via .best_estimator_ attribute.
    """
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]
    )

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10],
        "model__class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(
        pipeline, param_grid, scoring=clinical_scorer, cv=cv, n_jobs=-1
    )

    search.fit(X_train, y_train)

    return search


def fit_decision_tree(X_train, y_train):
    """Fit a decision tree model with hyperparameter tuning.

    Performs grid search with 5-fold stratified cross-validation to find
    optimal hyperparameters (max_depth, min_samples_leaf, and class_weight)
    using clinical score.

    Args:
        X_train: DataFrame with training features.
        y_train: Series with training target variable.

    Returns:
        GridSearchCV: Fitted grid search object with best estimator accessible
                      via .best_estimator_ attribute.
    """
    model = DecisionTreeClassifier(random_state=RANDOM_STATE)

    param_grid = {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_leaf": [1, 5, 10, 20],
        "class_weight": [None, "balanced"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = GridSearchCV(model, param_grid, scoring=clinical_scorer, cv=cv, n_jobs=-1)

    search.fit(X_train, y_train)

    return search


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance on test data using multiple metrics.

    Computes predictions and probability estimates, then calculates sensitivity,
    negative predictive value, specificity, positive predictive value, accuracy,
    and ROC AUC score.

    Args:
        model: Fitted scikit-learn classifier with predict and predict_proba methods.
        X_test: DataFrame with test features.
        y_test: Series with test target variable.

    Returns:
        dict: Dictionary with keys 'sensitivity', 'npv', 'specificity', 'ppv',
              'accuracy', and 'roc_auc' containing the corresponding metric values.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    return {
        "sensitivity": tp / (tp + fn),
        "npv": tn / (tn + fn),
        "specificity": tn / (tn + fp),
        "ppv": tp / (tp + fp),
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "roc_auc": roc_auc_score(y_test, y_prob),
    }


def run_models(train_df, test_df):
    """Train and evaluate both logistic regression and decision tree models.

    Prepares data, trains logistic regression and decision tree models with
    hyperparameter tuning, evaluates both on test set, and returns aggregated
    results along with the fitted models.

    Args:
        train_df: DataFrame with training data.
        test_df: DataFrame with test data.

    Returns:
        tuple: (results_df, logistic_search, tree_search) where results_df is a
               DataFrame with evaluation metrics for both models, and the other
               elements are GridSearchCV objects for the trained models.
    """
    X_train, X_test, y_train, y_test = prepare_data(train_df, test_df)

    # logistic regression
    logistic_search = fit_logistic_regression(X_train, y_train)

    logistic_results = evaluate_model(logistic_search.best_estimator_, X_test, y_test)

    # decision tree
    tree_search = fit_decision_tree(X_train, y_train)

    tree_results = evaluate_model(tree_search.best_estimator_, X_test, y_test)

    results = pd.DataFrame(
        [logistic_results, tree_results], index=["Logistic Regression", "Decision Tree"]
    )

    return (
        results,
        logistic_search,
        tree_search,
    )


def main():
    """Main entry point for model training and evaluation pipeline."""
    raw = load_data()

    # original data cleaning
    train_df, test_df = clean_data(raw)

    results_original, logistic_original, tree_original = run_models(train_df, test_df)

    print("Original data")
    print("Logistic regression:")
    print(logistic_original.best_params_)

    print("Decision tree:")
    print(tree_original.best_params_)

    print("Test performance:")
    print(results_original.round(3))

    plt.figure(figsize=(20, 10))
    plot_tree(tree_original.best_estimator_, filled=True)
    plt.savefig(BASE_DIR / "figs" / "decision_tree.png", bbox_inches="tight")
    plt.close()

    # perturbed cleaning
    train_perturbed, test_perturbed = clean_data(
        raw, palpable_skull_fracture_handling="palpable_skull_fracture_missing"
    )

    results_perturbed, logistic_perturbed, tree_perturbed = run_models(
        train_perturbed, test_perturbed
    )

    print("Perturbed data")
    print("Logistic regression:")
    print(logistic_perturbed.best_params_)
    print("Decision tree:")
    print(tree_perturbed.best_params_)
    print("Test performance:")
    print(results_perturbed.round(3))

    stability = pd.DataFrame(
        {
            "Logistic Regression": [
                results_original.loc["Logistic Regression", "sensitivity"],
                results_perturbed.loc["Logistic Regression", "sensitivity"],
                results_original.loc["Logistic Regression", "npv"],
                results_perturbed.loc["Logistic Regression", "npv"],
            ],
            "Decision Tree": [
                results_original.loc["Decision Tree", "sensitivity"],
                results_perturbed.loc["Decision Tree", "sensitivity"],
                results_original.loc["Decision Tree", "npv"],
                results_perturbed.loc["Decision Tree", "npv"],
            ],
        },
        index=[
            "Sensitivity - Original",
            "Sensitivity - Perturbed",
            "NPV - Original",
            "NPV - Perturbed",
        ],
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    metrics = ["sensitivity", "npv"]
    titles = [
        "Sensitivity (Before and After Perturbation)",
        "Negative Predictive Value (Before and After Perturbation)",
    ]

    for ax, metric, title in zip(axes, metrics, titles):
        values = pd.DataFrame(
            {
                "Original": [
                    results_original.loc["Logistic Regression", metric],
                    results_original.loc["Decision Tree", metric],
                ],
                "Perturbed": [
                    results_perturbed.loc["Logistic Regression", metric],
                    results_perturbed.loc["Decision Tree", metric],
                ],
            },
            index=["Logistic Regression", "Decision Tree"],
        )

        values.T.plot(kind="bar", ax=ax)

        ax.tick_params(axis="x", rotation=0)

        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3)

        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.set_ylim(0, 1.2)
        ax.legend(title="Model", loc="lower right")

    plt.tight_layout()
    plt.savefig(BASE_DIR / "figs" / "model_stability.pdf", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
