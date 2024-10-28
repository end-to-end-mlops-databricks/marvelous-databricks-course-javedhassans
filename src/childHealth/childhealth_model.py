from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

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
            ('regressor', RandomForestRegressor(
                n_estimators=config['model_parameters']['n_estimators'],
                max_depth=config['model_parameters']['max_depth'],
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
        :return: Mean squared error and R^2 score.
        """
        y_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def get_feature_importance(self):
        """
        Get feature importance from the trained model.

        :return: Feature importances and feature names.
        """
        feature_importance = self.model.named_steps['regressor'].feature_importances_
        feature_names = self.model.named_steps['preprocessor'].get_feature_names_out()
        return feature_importance, feature_names