import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def kappa_metric(y_true, y_pred):
    y_pred = np.argmax(y_pred.reshape(len(y_true), -1), axis=1)
    return "kappa", cohen_kappa_score(y_true, y_pred, weights="quadratic"), True


class ChildHealthModel:
    def __init__(self, preprocessor, config):
        """
        Initialize the ChildHealthModel with a preprocessor and configuration.

        :param preprocessor: A ColumnTransformer for preprocessing the data.
        :param config: A dictionary containing model parameters.
        """
        self.config = config
        self.model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "regressor",
                    lgb.LGBMClassifier(
                        n_estimators=config["lgb_boosting_parameters"]["n_estimators"],
                        max_depth=config["lgb_boosting_parameters"]["max_depth"],
                        learning_rate=config["lgb_boosting_parameters"]["learning_rate"],
                        objective=config["lgb_boosting_parameters"]["objective"],
                        num_class=config["lgb_boosting_parameters"]["num_class"],
                        boosting_type=config["lgb_boosting_parameters"]["boosting_type"],
                        num_leaves=config["lgb_boosting_parameters"]["num_leaves"],
                        min_data_in_leaf=config["lgb_boosting_parameters"]["min_data_in_leaf"],
                        feature_fraction=config["lgb_boosting_parameters"]["feature_fraction"],
                        bagging_fraction=config["lgb_boosting_parameters"]["bagging_fraction"],
                        bagging_freq=config["lgb_boosting_parameters"]["bagging_freq"],
                        lambda_l1=config["lgb_boosting_parameters"]["lambda_l1"],
                        lambda_l2=config["lgb_boosting_parameters"]["lambda_l2"],
                        random_state=42,
                        metric="None",  # Disable default metrics
                    ),
                ),
            ]
        )
        self.best_thresholds = [0.5, 1.5, 2.5, 3.5]  # Initial thresholds for rounding

    def quadratic_weighted_kappa(self, y_true, y_pred):
        return cohen_kappa_score(y_true, y_pred, weights="quadratic")

    def threshold_Rounder(self, preds, thresholds):
        return np.where(
            preds < thresholds[0],
            0,
            np.where(
                preds < thresholds[1], 1, np.where(preds < thresholds[2], 2, np.where(preds < thresholds[3], 3, 4))
            ),
        )

    def optimize_thresholds(self, y_true, preds):
        def evaluate_thresholds(thresholds):
            rounded_preds = self.threshold_Rounder(preds, thresholds)
            return -self.quadratic_weighted_kappa(y_true, rounded_preds)

        result = minimize(evaluate_thresholds, self.best_thresholds, method="Nelder-Mead")
        self.best_thresholds = result.x  # Update best thresholds after optimization

    def train(self, X_train, y_train):
        """
        Train the model on the provided training data.

        :param X_train: Training features.
        :param y_train: Training target.
        """
        self.model.named_steps["regressor"].fit(X_train, y_train, eval_metric=kappa_metric)

    def predict(self, X):
        """
        Predict using the trained model.

        :param X: Features to predict.
        :return: Predictions.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities for each class using the trained model.

        :param X: Features to predict.
        :return: Predicted probabilities.
        """
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data.

        :param X_test: Test features.
        :param y_test: Test target.
        :return: Classification report, accuracy score, and kappa score.
        """
        y_pred = self.predict(X_test)
        kappa = self.quadratic_weighted_kappa(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        return report, accuracy, kappa

    def get_feature_importance(self):
        """
        Get feature importance from the trained model.

        :return: Feature importances and feature names.
        """
        feature_importance = self.model.named_steps["regressor"].feature_importances_
        feature_names = self.model.named_steps["preprocessor"].get_feature_names_out()
        return feature_importance, feature_names


# Main function
if __name__ == "__main__":
    # Load configuration
    config_path = "project_config.yml"
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    # Load and preprocess data
    train_path = "train.csv"
    data = pd.read_csv(train_path)
    X = data[config["num_features"] + config["cat_features"]]
    y = data[config["target"]]

    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config["num_features"]),
            ("cat", OneHotEncoder(drop="first"), config["cat_features"]),
        ]
    )

    # Initialize model class
    child_health_model = ChildHealthModel(preprocessor, config)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train model
    child_health_model.train(X_train, y_train)

    # Out-of-Fold predictions for threshold optimization
    oof_preds = child_health_model.predict_proba(X_test)
    child_health_model.optimize_thresholds(y_test, oof_preds)

    # Evaluate model
    report, accuracy, kappa = child_health_model.evaluate(X_test, y_test)
    print("Evaluation Results:")
    print(report)
    print("Accuracy:", accuracy)
    print("Kappa Score:", kappa)

    # Display feature importance
    feature_importance, feature_names = child_health_model.get_feature_importance()
    feature_importance_df = pd.DataFrame({"feature": feature_names, "importance": feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by="importance", ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df["feature"], feature_importance_df["importance"], color="skyblue")
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.gca().invert_yaxis()
    plt.show()

    # Log metrics in MLflow
    with mlflow.start_run():
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("quadratic_weighted_kappa", kappa)
