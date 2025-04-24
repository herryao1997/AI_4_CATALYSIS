"""
models/model_xgb.py

A simple XGBoost regressor wrapper using the scikit-learn API.
"""

from xgboost import XGBRegressor

class XGBRegression:
    """
    XGBoost with reg_alpha, reg_lambda for L1/L2 regularization.
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 random_state=42, reg_alpha=0.0, reg_lambda=1.0):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=0,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda
        )

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_
