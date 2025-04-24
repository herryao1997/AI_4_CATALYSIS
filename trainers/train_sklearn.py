"""
trainers/train_sklearn.py

Contains a simple 'fit' function for sklearn-based models,
as they do not require an epoch-based training loop.
"""

def train_sklearn_model(model, X_train, Y_train):
    """
    Train a sklearn model (e.g., RandomForestRegressor).
    :param model: the sklearn model instance
    :param X_train: shape (N, input_dim)
    :param Y_train: shape (N, output_dim)
    :return: the trained model
    """
    model.fit(X_train, Y_train)
    return model
