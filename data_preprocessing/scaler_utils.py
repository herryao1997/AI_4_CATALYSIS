"""
data_preprocessing/scaler_utils.py

Contains functions for data standardization (using StandardScaler),
as well as saving/loading scaler objects for future use.

【去掉原先的 logit transform】, 保留/新增 bounded transform(0..100 => -1..1).
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def bounded_transform(y):
    """
    y in [0,100] => z in [-1,1], linear:
        z = 2*(y/100) - 1
    clamp y to [0,100] just in case
    """
    y_ = np.clip(y, 0, 100)
    return 2*(y_/100.0) - 1

def inverse_bounded_transform(z):
    """
    z => clamp in [-1,1], => y=100*(z+1)/2 in [0,100]
    """
    z_ = np.clip(z, -1, 1)
    return 100.0*(z_+1.0)/2.0

def standardize_data(X_train, X_val,
                     Y_train, Y_val,
                     do_input=True,
                     do_output=False,
                     numeric_cols_idx=None,
                     do_output_bounded=False):
    """
    Optionally standardize input features (X) and/or output targets (Y).
    If do_output_bounded=True => 0..100 => -1..1 => standard => model.
    """
    scaler_x = None
    scaler_y = None

    X_train_s = np.copy(X_train)
    X_val_s   = np.copy(X_val)
    Y_train_s = np.copy(Y_train)
    Y_val_s   = np.copy(Y_val)

    if do_input:
        if numeric_cols_idx is None:
            numeric_cols_idx = list(range(X_train.shape[1]))
        scaler_x = StandardScaler()
        scaler_x.fit(X_train_s[:, numeric_cols_idx])
        X_train_s[:, numeric_cols_idx] = scaler_x.transform(X_train_s[:, numeric_cols_idx])
        X_val_s[:, numeric_cols_idx]   = scaler_x.transform(X_val_s[:, numeric_cols_idx])

    if do_output:
        # bounded or not
        if do_output_bounded:
            # 0..100 => -1..1
            for i in range(Y_train_s.shape[1]):
                Y_train_s[:,i] = bounded_transform(Y_train_s[:,i])
                Y_val_s[:,i]   = bounded_transform(Y_val_s[:,i])
            transform_type = "bounded+standard"
        else:
            transform_type = "standard"

        # standard
        scaler_obj = StandardScaler()
        scaler_obj.fit(Y_train_s)
        Y_train_s = scaler_obj.transform(Y_train_s)
        Y_val_s   = scaler_obj.transform(Y_val_s)

        scaler_y = {
            "type": transform_type,
            "scaler": scaler_obj
        }

    return (X_train_s, X_val_s, scaler_x), (Y_train_s, Y_val_s, scaler_y)

def save_scaler(scaler, path):
    if scaler is not None:
        joblib.dump(scaler, path)

def load_scaler(path):
    return joblib.load(path)

def inverse_transform_output(y_pred, scaler_y):
    """
    If scaler_y["type"]=="bounded+standard": inverse standard => inverse_bounded => [0,100].
    If "standard": just inverse standard.
    """
    if scaler_y is None:
        return y_pred
    if not isinstance(scaler_y, dict):
        # older usage => direct standard
        return scaler_y.inverse_transform(y_pred)

    transform_type = scaler_y["type"]
    scaler_obj = scaler_y["scaler"]
    # 1) inverse standard
    y_ = scaler_obj.inverse_transform(y_pred)

    if transform_type.startswith("bounded"):
        # each col => clamp => [0,100]
        for i in range(y_.shape[1]):
            y_[:,i] = inverse_bounded_transform(y_[:,i])

    return y_
