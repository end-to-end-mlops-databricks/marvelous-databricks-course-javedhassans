import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb

class ChildHealthModel:
    def __init__(self, preprocessor, config):
        """
        Initialize the ChildHealthModel with a preprocessor and configuration.

        :param preprocessor: A ColumnTransformer for preprocessing the data.
        :param config: A dictionary containing model parameters.
        """
        self.config = config
        self.model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', lgb.LGBMClassifier(
                n_estimators=config['model_parameters']['n_estimators'],
                max_depth=config['model_parameters']['max_depth'],
                learning_rate=config['model_parameters']['learning_rate'],
                objective='multiclass',
                num_class=config['model_parameters']['num_class'],
                random_state=42
            ))
        ])

    def train(self, X_train, y_train):
        """
        Train the model on the provided training data.

        :param X_train: Training features.
        :param y_train: Training target.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X):
        """
        Predict using the trained model.

        :param X: Features to predict.
        :return: Predictions.
        """
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the test data.

        :param X_test: Test features.
        :param y_test: Test target.
        :return: Classification report and accuracy score.
        """
        y_pred = self.predict(X_test)
        report = classification_report(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        return report, accuracy

    def get_feature_importance(self):
        """
        Get feature importance from the trained model.

        :return: Feature importances and feature names.
        """
        feature_importance = self.model.named_steps['regressor'].feature_importances_
        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        return feature_importance, feature_names

#     def save_model(self, model_path='lgbm_model.pkl'):
#         """Save the trained model."""
#         joblib.dump(self.model, model_path)
#         print(f"Model saved as {model_path}")

#     def load_model(self, model_path='lgbm_model.pkl'):
#         """Load a saved model."""
#         self.model = joblib.load(model_path)
#         print(f"Model loaded from {model_path}")

# if __name__ == "__main__":
#     # Load configuration
#     config_path = 'project_config.yml'
#     with open(config_path, 'r') as file:
#         config = yaml.safe_load(file)

#     # Load and preprocess data
#     train_path = 'train.csv'
#     data = pd.read_csv(train_path)
#     X = data[config['num_features'] + config['cat_features']]
#     y = data[config['target']]

#     # Define preprocessor
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', StandardScaler(), config['num_features']),
#             ('cat', OneHotEncoder(drop='first'), config['cat_features'])
#         ]
#     )

#     # Initialize model class
#     child_health_model = ChildHealthModel(preprocessor, config)

#     # Split data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

#     # Train model
#     child_health_model.train(X_train, y_train)

#     # Evaluate model
#     report, accuracy = child_health_model.evaluate(X_test, y_test)
#     print("Evaluation Results:")
#     print(report)
#     print("Accuracy:", accuracy)

#     # Display feature importance
#     feature_importance, feature_names = child_health_model.get_feature_importance()
#     feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': feature_importance})
#     feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

#     plt.figure(figsize=(10, 8))
#     plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='skyblue')
#     plt.xlabel('Importance')
#     plt.title('Feature Importance')
#     plt.gca().invert_yaxis()
#     plt.show()

#     # Save model
#     child_health_model.save_model()