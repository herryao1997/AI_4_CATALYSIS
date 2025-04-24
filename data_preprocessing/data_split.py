"""
data_preprocessing/data_split.py

Contains the function to split data into train/validation sets
using scikit-learn's train_test_split.
Split data into train/val sets or K folds
"""

from sklearn.model_selection import train_test_split, KFold

def split_data(X, Y, test_size=0.2, random_state=42):
    """
    Split the dataset into train and validation sets.
    :param X: Input features, shape (N, input_dim)
    :param Y: Output targets, shape (N, output_dim)
    :param test_size: Fraction of data for validation
    :param random_state: For reproducibility
    :return: X_train, X_val, Y_train, Y_val
    """
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)


# def kfold_split_data(X, Y, n_splits=5, random_state=42, shuffle=True):
#     kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
#     for train_idx, val_idx in kf.split(X):
#         X_train, X_val = X[train_idx], X[val_idx]
#         Y_train, Y_val = Y[train_idx], Y[val_idx]
#         yield (X_train, X_val, Y_train, Y_val)
