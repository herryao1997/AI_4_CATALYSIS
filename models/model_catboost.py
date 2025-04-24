"""
models/model_catboost.py
"""

from catboost import CatBoostRegressor

class CatBoostRegression:
    """
    CatBoost with l2_leaf_reg controlling L2 regularization strength.
    """
    def __init__(self, iterations=100, learning_rate=0.1, depth=6,
                 random_seed=42, l2_leaf_reg=3.0):
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
            verbose=0,
            loss_function="MultiRMSE",
            l2_leaf_reg=l2_leaf_reg
        )

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    # @property
    # def feature_importances_(self):
    #     return self.model.get_feature_importance(type="PredictionValuesChange")

    @property
    def feature_importances_(self):
        """
        Returns normalized feature importance values for CatBoost, ensuring
        the sum of all importances equals 1, making it comparable to
        feature importances from XGBoost, RandomForest, and DecisionTree.
        """
        importances = self.model.get_feature_importance(type="PredictionValuesChange")

        # 避免除零错误
        total_importance = sum(importances)
        if total_importance > 0:
            return importances / total_importance  # 归一化，使总和为 1
        else:
            return importances  # 如果全是 0，就直接返回

