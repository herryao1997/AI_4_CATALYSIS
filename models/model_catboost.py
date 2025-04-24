"""
models/model_catboost.py
"""

from catboost import CatBoostRegressor
from catboost import Pool

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

    def get_shap_values(self, X):
        pool = Pool(X)
        shap_values = self.model.get_feature_importance(type="ShapValues", data=pool)
        # 如果返回的是多输出 (3D 数组)，比如形状为 (n_samples, n_outputs, n_features+1)
        if len(shap_values.shape) == 3:
            # 对每个输出都去掉最后一列基线值（如果列数 == 特征数+1）
            outputs = []
            for i in range(shap_values.shape[1]):
                if shap_values.shape[2] == X.shape[1] + 1:
                    outputs.append(shap_values[:, i, :-1])
                else:
                    outputs.append(shap_values[:, i, :])
            return outputs
        # 如果返回的是二维数组 (n_samples, n_features+1)
        elif len(shap_values.shape) == 2:
            if shap_values.shape[1] == X.shape[1] + 1:
                shap_values = shap_values[:, :-1]
            return shap_values