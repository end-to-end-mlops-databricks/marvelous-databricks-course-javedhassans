import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from childHealth.config import ProjectConfig


def kappa_metric(y_true, y_pred_raw):
    # Use rounding to convert predicted probabilities to the nearest class
    y_pred = np.round(y_pred_raw).astype(int)
    return "kappa", cohen_kappa_score(y_true, y_pred, weights="quadratic"), True


class ChildHealthModel:
    def __init__(self, preprocessor, config):
        """
        Initialize the ChildHealthModel with a preprocessor and configuration.

        :param preprocessor: A ColumnTransformer for preprocessing the data.
        :param config: A dictionary containing model parameters.
        """
        self.config = config
        self.preprocessor = preprocessor
        self.model = lgb.LGBMClassifier(
            learning_rate=config.lgb_parameters["learning_rate"],
            max_depth=config.lgb_parameters["max_depth"],
            num_leaves=config.lgb_parameters["num_leaves"],
            min_data_in_leaf=config.lgb_parameters["min_data_in_leaf"],
            feature_fraction=config.lgb_parameters["feature_fraction"],
            bagging_fraction=config.lgb_parameters["bagging_fraction"],
            bagging_freq=config.lgb_parameters["bagging_freq"],
            lambda_l1=config.lgb_parameters["lambda_l1"],
            lambda_l2=config.lgb_parameters["lambda_l2"],
            n_estimators=config.lgb_parameters["n_estimators"],
            num_class=config.lgb_parameters["num_class"],
            objective=config.lgb_parameters["objective"],
            boosting_type=config.lgb_parameters["boosting_type"],
            random_state=42,
            metric="None",  # Disable default metrics
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

    def train(self, X_train, y_train, X_val, y_val):
        """
        Train the model on the provided training data.

        :param X_train: Training features.
        :param y_train: Training target.
        :param X_val: Validation features.
        :param y_val: Validation target.
        """
        X_train = self.preprocessor.fit_transform(X_train)
        X_val = self.preprocessor.transform(X_val)

        # Using the LightGBM Dataset for early stopping support
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Train with early stopping
        self.model = lgb.train(
            params=self.config.lgb_parameters,
            train_set=train_data,
            valid_sets=[val_data],
            valid_names=["validation"],
            feval=kappa_metric,
            # early_stopping_rounds=10
        )

    def predict(self, X):
        """
        Predict using the trained model.

        :param X: Features to predict.
        :return: Predictions.
        """
        X = self.preprocessor.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict probabilities for each class using the trained model.

        :param X: Features to predict.
        :return: Predicted probabilities.
        """
        X = self.preprocessor.transform(X)
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
        feature_importance = self.model.feature_importances_
        feature_names = self.preprocessor.get_feature_names_out()
        return feature_importance, feature_names


# Main function
if __name__ == "__main__":
    # Load configuration
    config_path = "project_config.yml"
    config = ProjectConfig.from_yaml(config_path)

    # Load and preprocess data
    train_path = "train.csv"
    data = pd.read_csv(train_path)
    X = data[config.num_features + config.cat_features]
    y = data[config.target]

    # Define preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.num_features),
            ("cat", OneHotEncoder(drop="first"), config.cat_features),
        ]
    )

    # Initialize model class
    child_health_model = ChildHealthModel(preprocessor, config)

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

    # Train model
    child_health_model.train(X_train, y_train, X_val, y_val)

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
