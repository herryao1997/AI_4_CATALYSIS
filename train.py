"""
train.py

- 读取 CSV (前14列)
- 得到 X, Y, numeric_cols_idx, x_col_names, y_col_names, observed_combos, onehot_groups, oh_index_map
- 训练模型并保存
- 将 observed_combos, onehot_groups, oh_index_map 等一起存进 metadata.pkl
"""

import yaml
import os
import numpy as np
import torch
import joblib

from data_preprocessing.data_loader import (
    load_dataset,
    load_raw_data_for_correlation,
    extract_data_statistics
)
from data_preprocessing.data_split import split_data
from data_preprocessing.scaler_utils import (
    standardize_data, inverse_transform_output, save_scaler
)

# 各种模型
from models.model_ann import ANNRegression
from models.model_rf import RFRegression
from models.model_dt import DTRegression
from models.model_catboost import CatBoostRegression
from models.model_xgb import XGBRegression

# 训练 & 评估
from losses.torch_losses import get_torch_loss_fn
from trainers.train_torch import train_torch_model_dataloader
from trainers.train_sklearn import train_sklearn_model
from evaluation.metrics import compute_regression_metrics

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def create_model_by_type(model_type, config, random_seed=42, input_dim=None):
    if model_type == "ANN":
        ann_cfg = config["model"]["ann_params"]
        actual_dim = input_dim if input_dim is not None else ann_cfg["input_dim"]
        model = ANNRegression(
            input_dim=actual_dim,
            output_dim=ann_cfg["output_dim"],
            hidden_dims=ann_cfg["hidden_dims"],
            dropout=ann_cfg.get("dropout", 0.0),
            activation=ann_cfg.get("activation", "ReLU"),
            random_seed=ann_cfg.get("random_seed", 42)
        )
        return model
    elif model_type == "RF":
        rf_cfg = config["model"]["rf_params"]
        return RFRegression(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            random_state=rf_cfg["random_state"],
            ccp_alpha=rf_cfg.get("ccp_alpha", 0.0),
            min_samples_leaf=rf_cfg.get("min_samples_leaf", 1)
        )
    elif model_type == "DT":
        dt_cfg = config["model"]["dt_params"]
        return DTRegression(
            max_depth=dt_cfg["max_depth"],
            random_state=dt_cfg["random_state"],
            ccp_alpha=dt_cfg.get("ccp_alpha", 0.0)
        )
    elif model_type == "CatBoost":
        cat_cfg = config["model"]["catboost_params"]
        return CatBoostRegression(
            iterations=cat_cfg["iterations"],
            learning_rate=cat_cfg["learning_rate"],
            depth=cat_cfg["depth"],
            random_seed=cat_cfg["random_seed"],
            l2_leaf_reg=cat_cfg.get("l2_leaf_reg", 3.0)
        )
    elif model_type == "XGB":
        xgb_cfg = config["model"]["xgb_params"]
        return XGBRegression(
            n_estimators=xgb_cfg["n_estimators"],
            learning_rate=xgb_cfg["learning_rate"],
            max_depth=xgb_cfg["max_depth"],
            random_state=xgb_cfg["random_seed"],
            reg_alpha=xgb_cfg.get("reg_alpha", 0.0),
            reg_lambda=xgb_cfg.get("reg_lambda", 1.0)
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def train_main():
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    csv_path = config["data"]["path"]
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    base_outdir = os.path.join("postprocessing", csv_name, "train")
    ensure_dir(base_outdir)

    # 1) 加载数据
    (X, Y, numeric_cols_idx, x_col_names, y_col_names,
     observed_combos, onehot_groups, oh_index_map) = load_dataset(csv_path)

    # 1.1) 保存 X_onehot.npy => 做 correlation_heatmap_one_hot
    np.save(os.path.join(base_outdir, "X_onehot.npy"), X)
    np.save(os.path.join(base_outdir, "x_onehot_colnames.npy"), x_col_names)

    # 1.2) 若要做 raw correlation
    df_raw_14 = load_raw_data_for_correlation(csv_path, drop_nan=True)
    raw_csv_path = os.path.join(base_outdir, "df_raw_14.csv")
    df_raw_14.to_csv(raw_csv_path, index=False)
    print(f"[INFO] Saved raw 14-col CSV => {raw_csv_path}")

    # 2) 提取统计信息 => metadata.pkl
    stats_dict = extract_data_statistics(
        X, x_col_names, numeric_cols_idx,
        Y=Y, y_col_names=y_col_names
    )
    # 额外记录
    stats_dict["onehot_groups"] = onehot_groups
    stats_dict["oh_index_map"]  = oh_index_map
    stats_dict["observed_onehot_combos"] = observed_combos

    # 保留原始大小写列名
    stats_dict["x_col_names"] = x_col_names
    stats_dict["y_col_names"] = y_col_names

    meta_path = os.path.join("./models", "metadata.pkl")
    joblib.dump(stats_dict, meta_path)
    print(f"[INFO] metadata saved => {meta_path}")

    # 3) 数据拆分 & 标准化
    random_seed = config["data"].get("random_seed", 42)
    X_train, X_val, Y_train, Y_val = split_data(
        X, Y,
        test_size=config["data"]["test_size"],
        random_state=random_seed
    )
    (X_train_s, X_val_s, sx), (Y_train_s, Y_val_s, sy) = standardize_data(
        X_train, X_val, Y_train, Y_val,
        do_input=config["preprocessing"]["standardize_input"],
        do_output=config["preprocessing"]["standardize_output"],
        numeric_cols_idx=numeric_cols_idx,
        do_output_bounded=config["preprocessing"].get("bounded_output", False)
    )

    np.save(os.path.join(base_outdir, "Y_train.npy"), Y_train)
    np.save(os.path.join(base_outdir, "Y_val.npy"), Y_val)

    # 4) 训练 & 保存
    model_types = config["model"]["types"]
    for mtype in model_types:
        print(f"\n=== Train model: {mtype} ===")
        outdir_m = os.path.join(base_outdir, mtype)
        ensure_dir(outdir_m)

        model = create_model_by_type(mtype, config, random_seed, input_dim=X_train_s.shape[1])

        # Torch 或 Sklearn 模型处理
        if mtype == "ANN":
            loss_fn = get_torch_loss_fn(config["loss"]["type"])
            ann_cfg = config["model"]["ann_params"]
            from data_preprocessing.my_dataset import MyDataset
            train_ds = MyDataset(X_train_s, Y_train_s)
            val_ds = MyDataset(X_val_s, Y_val_s)

            model, train_losses, val_losses = train_torch_model_dataloader(
                model, train_ds, val_ds,
                loss_fn=loss_fn,
                epochs=ann_cfg["epochs"],
                batch_size=ann_cfg["batch_size"],
                lr=float(ann_cfg["learning_rate"]),
                weight_decay=float(ann_cfg.get("weight_decay", 0.0)),
                checkpoint_path=ann_cfg["checkpoint_path"],
                log_interval=config["training"]["log_interval"],
                early_stopping=ann_cfg.get("early_stopping", False),
                patience=ann_cfg.get("patience", 5),
                optimizer_name=ann_cfg.get("optimizer", "Adam")
            )
            model.to("cpu")
            np.save(os.path.join(outdir_m, "train_losses.npy"), train_losses)
            np.save(os.path.join(outdir_m, "val_losses.npy"), val_losses)
        else:
            model = train_sklearn_model(model, X_train_s, Y_train_s)

        # 推断(Train/Val)
        if hasattr(model, 'eval') and hasattr(model, 'forward'):
            with torch.no_grad():
                p_tr = model(torch.tensor(X_train_s, dtype=torch.float32)).cpu().numpy()
                p_va = model(torch.tensor(X_val_s, dtype=torch.float32)).cpu().numpy()
            train_pred = p_tr
            val_pred = p_va
        else:
            train_pred = model.predict(X_train_s)
            val_pred = model.predict(X_val_s)

        # 反变换
        if config["preprocessing"]["standardize_output"]:
            train_pred = inverse_transform_output(train_pred, sy)
            val_pred = inverse_transform_output(val_pred, sy)

        train_m = compute_regression_metrics(Y_train, train_pred)
        val_m = compute_regression_metrics(Y_val, val_pred)
        print(f"   => train={train_m}, val={val_m}")

        # 保存预测结果
        np.save(os.path.join(outdir_m, "train_pred.npy"), train_pred)
        np.save(os.path.join(outdir_m, "val_pred.npy"), val_pred)
        joblib.dump({"train_metrics": train_m, "val_metrics": val_m},
                    os.path.join(outdir_m, "metrics.pkl"))

        # 保存 feature_importances（如果存在）
        if config["evaluation"].get("save_feature_importance_bar", False):
            fi_ = None
            if hasattr(model, "feature_importances_"):
                fi_ = model.feature_importances_
            elif hasattr(model, "model") and hasattr(model.model, "feature_importances_"):
                fi_ = model.model.feature_importances_
            if fi_ is not None:
                np.save(os.path.join(outdir_m, "feature_importance.npy"), fi_)

        # 同步保存原始大小写的 x_col_names
        np.save(os.path.join(outdir_m, "x_col_names.npy"), x_col_names)

        # 保存模型 & scaler
        model_dir_ = os.path.join("./models", mtype)
        os.makedirs(model_dir_, exist_ok=True)
        save_scaler(sx, os.path.join(model_dir_, f"scaler_x_{mtype}.pkl"))
        save_scaler(sy, os.path.join(model_dir_, f"scaler_y_{mtype}.pkl"))

        np.save(os.path.join(model_dir_, "x_col_names.npy"), x_col_names)
        np.save(os.path.join(model_dir_, "y_col_names.npy"), y_col_names)

        if mtype != "ANN":
            joblib.dump(model, os.path.join(model_dir_, "trained_model.pkl"))
            print(f"[INFO] saved => {mtype}/trained_model.pkl")

# --------------------- 添加 SHAP 保存逻辑 ---------------------
        if config["evaluation"].get("save_shap", False):
            # 生成存放 SHAP 数据的目录
            shap_dir = os.path.join("evaluation", "figures", csv_name, "model_comparison", mtype, "shap")
            ensure_dir(shap_dir)
            # 利用标准化后的训练集与验证集构成完整数据集
            X_full_s = np.concatenate([X_train_s, X_val_s], axis=0)
            try:
                import shap
                if mtype == "ANN":
                    # 对于 ANN 模型，保持模型在 eval 模式，并选取部分训练数据作为背景数据
                    model.eval()
                    background = torch.tensor(X_train_s[:100], dtype=torch.float32)
                    explainer = shap.DeepExplainer(model, background)
                    shap_values = explainer.shap_values(torch.tensor(X_full_s, dtype=torch.float32))
                elif mtype == "CatBoost":
                    shap_values = model.get_shap_values(X_full_s)
                else:
                    # 对于其他模型，若模型内部包含真实模型对象，则获取该对象
                    base_model = model.model if hasattr(model, "model") else model
                    explainer = shap.TreeExplainer(base_model)
                    shap_values = explainer.shap_values(X_full_s)
                shap_save = {
                    "shap_values": shap_values,
                    "X_full": X_full_s,
                    "x_col_names": x_col_names,
                    "y_col_names": y_col_names
                }
                shap_save_path = os.path.join(shap_dir, "shap_data.pkl")
                joblib.dump(shap_save, shap_save_path)
                print(f"[INFO] SHAP data saved for model {mtype} => {shap_save_path}")
            except Exception as e:
                print(f"[WARN] SHAP computation failed for {mtype}: {e}")
        # -----------------------------------------------------------
    print("\n[INFO] train_main => done.")


if __name__ == "__main__":
    train_main()
