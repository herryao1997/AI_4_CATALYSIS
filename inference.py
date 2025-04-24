"""
inference.py

- 读取 ./models/<model_type>/trained_model.pkl / best_ann.pt
- 读取 metadata.pkl (其中有 onehot_groups, oh_index_map, observed_onehot_combos, etc.)
- 只在实际出现过的 one-hot 组合上做平均 (避免从未见过的组合)
- 输出 heatmap_pred.npy, confusion_pred.npy 等
  (可加权: sum_real += real_pred * freq; avg_real = sum_real / sum_freq)
"""

import yaml
import os
import numpy as np
import torch
import joblib
from tqdm import trange
from itertools import product

from data_preprocessing.scaler_utils import load_scaler, inverse_transform_output

# 各种模型
from models.model_ann import ANNRegression
from models.model_rf import RFRegression
from models.model_dt import DTRegression
from models.model_catboost import CatBoostRegression
from models.model_xgb import XGBRegression

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def load_inference_model(model_type, config):
    model_dir = os.path.join("./models", model_type)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"[ERROR] Directory not found => {model_dir}")

    x_col_path = os.path.join(model_dir, "x_col_names.npy")
    y_col_path = os.path.join(model_dir, "y_col_names.npy")
    if not (os.path.exists(x_col_path) and os.path.exists(y_col_path)):
        raise FileNotFoundError("[ERROR] x_col_names.npy or y_col_names.npy not found.")

    x_col_names = list(np.load(x_col_path, allow_pickle=True))
    y_col_names = list(np.load(y_col_path, allow_pickle=True))

    if model_type == "ANN":
        ann_cfg = config["model"]["ann_params"]
        net = ANNRegression(
            input_dim=len(x_col_names),
            output_dim=len(y_col_names),
            hidden_dims=ann_cfg["hidden_dims"],
            dropout=ann_cfg.get("dropout", 0.0),
            activation=ann_cfg.get("activation", "ReLU"),
            random_seed=ann_cfg.get("random_seed", 42)
        )
        ckpt_path = os.path.join(model_dir, "best_ann.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"[ERROR] {ckpt_path} not found.")
        try:
            state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(ckpt_path, map_location="cpu")
        net.load_state_dict(state_dict)
        net.eval()
        return net, x_col_names, y_col_names
    else:
        pkl_path = os.path.join(model_dir, "trained_model.pkl")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"[ERROR] {pkl_path} not found.")

        model = joblib.load(pkl_path)
        return model, x_col_names, y_col_names

def model_predict(model, X_2d):
    if hasattr(model, 'eval') and hasattr(model, 'forward'):
        # Torch 模型
        with torch.no_grad():
            t_ = torch.tensor(X_2d, dtype=torch.float32)
            out_ = model(t_)
        return out_.cpu().numpy()
    else:
        # Sklearn / CatBoost / XGB
        return model.predict(X_2d)

def get_onehot_global_col_index(local_oh_index, oh_index_map):
    """
    local_oh_index => oh_index_map[local_oh_index]
    """
    return oh_index_map[local_oh_index]

def inference_main():
    with open("./configs/config.yaml","r") as f:
        config = yaml.safe_load(f)

    inf_models = config["inference"].get("models", [])
    if not inf_models:
        print("[INFO] No inference models => exit.")
        return

    meta_path = os.path.join("./models","metadata.pkl")
    if not os.path.exists(meta_path):
        print(f"[ERROR] metadata => {meta_path} missing.")
        return

    stats_dict = joblib.load(meta_path)
    # stats_dict 里包含:
    #   continuous_cols
    #   onehot_groups
    #   oh_index_map
    #   observed_onehot_combos => [((0,1,0...), freq), ...]
    #   x_col_names, y_col_names

    csv_path = config["data"]["path"]
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    base_inf = os.path.join("postprocessing", csv_name, "inference")
    ensure_dir(base_inf)

    observed_combos = stats_dict.get("observed_onehot_combos", [])
    oh_index_map   = stats_dict.get("oh_index_map", [])
    print(f"[INFO] #observed combos => {len(observed_combos)}")

    # Heatmap/Confusion Axes
    heatmap_x_name = config["inference"]["heatmap_axes"]["x_name"]
    heatmap_y_name = config["inference"]["heatmap_axes"]["y_name"]
    # row_name / col_name 仅在可视化时使用，这里不一定需要

    for mtype in inf_models:
        print(f"\n=== Inference => {mtype} ===")
        outdir_m = os.path.join(base_inf, mtype)
        ensure_dir(outdir_m)

        try:
            model, x_col_names, y_col_names = load_inference_model(mtype, config)
        except FileNotFoundError as e:
            print(e)
            continue

        # 加载 scaler
        model_dir = os.path.join("./models", mtype)
        sx_path = os.path.join(model_dir, f"scaler_x_{mtype}.pkl")
        sy_path = os.path.join(model_dir, f"scaler_y_{mtype}.pkl")
        if os.path.exists(sx_path):
            scaler_x = load_scaler(sx_path)
        else:
            scaler_x = None
        if os.path.exists(sy_path):
            scaler_y = load_scaler(sy_path)
        else:
            scaler_y = None

        # numeric cols
        numeric_cols_idx = []
        for cname in stats_dict["continuous_cols"].keys():
            if cname in x_col_names:
                numeric_cols_idx.append(x_col_names.index(cname))

        # ---------- A) 2D Heatmap ----------
        n_points = config["inference"].get("n_points", 50)
        if (heatmap_x_name not in stats_dict["continuous_cols"]) \
           or (heatmap_y_name not in stats_dict["continuous_cols"]):
            print(f"[WARN] heatmap axes not found in stats => skip 2D heatmap.")
        else:
            xinfo = stats_dict["continuous_cols"][heatmap_x_name]
            yinfo = stats_dict["continuous_cols"][heatmap_y_name]

            x_vals = np.linspace(xinfo["min"], xinfo["max"], n_points)
            y_vals = np.linspace(yinfo["min"], yinfo["max"], n_points)
            grid_x, grid_y = np.meshgrid(x_vals, y_vals)

            # base_vec => 连续列=mean, one-hot=0
            base_vec = np.zeros(len(x_col_names), dtype=float)
            for cname, cstat in stats_dict["continuous_cols"].items():
                if cname in x_col_names:
                    base_vec[x_col_names.index(cname)] = cstat["mean"]

            # outdim
            tmp_inp = base_vec.reshape(1,-1)
            if scaler_x:
                tmp_inp[:, numeric_cols_idx] = scaler_x.transform(tmp_inp[:, numeric_cols_idx])
            tmp_out = model_predict(model, tmp_inp)
            outdim = tmp_out.shape[0] if tmp_out.ndim==1 else tmp_out.shape[1]

            H, W = grid_x.shape
            heatmap_pred = np.zeros((H, W, outdim), dtype=float)

            row_iter = trange(H, desc="2DHeatmap Rows", ncols=100)
            for i in row_iter:
                for j in range(W):
                    vec = base_vec.copy()
                    if heatmap_x_name in x_col_names:
                        vec[x_col_names.index(heatmap_x_name)] = grid_x[i, j]
                    if heatmap_y_name in x_col_names:
                        vec[x_col_names.index(heatmap_y_name)] = grid_y[i, j]

                    sum_real = np.zeros(outdim, dtype=float)
                    # 遍历 observed_combos
                    for (oh_tuple, freq) in observed_combos:
                        tmpv = vec.copy()
                        # 设置 one-hot
                        for local_oh_index, val01 in enumerate(oh_tuple):
                            gcol = get_onehot_global_col_index(local_oh_index, oh_index_map)
                            tmpv[gcol] = val01

                        if scaler_x:
                            tmp_inp = tmpv.reshape(1, -1)
                            tmp_inp[:, numeric_cols_idx] = scaler_x.transform(tmp_inp[:, numeric_cols_idx])
                            scaled_pred = model_predict(model, tmp_inp)
                        else:
                            scaled_pred = model_predict(model, tmpv.reshape(1, -1))

                        real_pred = inverse_transform_output(scaled_pred, scaler_y)
                        if real_pred.ndim==2:
                            sum_real += real_pred[0]
                        else:
                            sum_real += real_pred
                    # avg
                    avg_real = sum_real / len(observed_combos)
                    avg_real = np.clip(avg_real, 0, 100)
                    heatmap_pred[i, j, :] = avg_real

            np.save(os.path.join(outdir_m, "heatmap_pred.npy"), heatmap_pred)
            np.save(os.path.join(outdir_m, "grid_x.npy"), grid_x)
            np.save(os.path.join(outdir_m, "grid_y.npy"), grid_y)
            print(f"[INFO] 2D heatmap => shape={heatmap_pred.shape} saved => {outdir_m}")

        # ---------- B) Confusion-like ----------
        if len(stats_dict["onehot_groups"])<2:
            print("[WARN] Not enough onehot groups => skip confusion.")
            continue

        grpA = stats_dict["onehot_groups"][0]
        grpB = stats_dict["onehot_groups"][1]

        base_vec = np.zeros(len(x_col_names), dtype=float)
        for cname, cstat in stats_dict["continuous_cols"].items():
            if cname in x_col_names:
                base_vec[x_col_names.index(cname)] = cstat["mean"]

        # outdim
        tmp_inp = base_vec.reshape(1, -1)
        if scaler_x:
            tmp_inp[:, numeric_cols_idx] = scaler_x.transform(tmp_inp[:, numeric_cols_idx])
        tmp_out = model_predict(model, tmp_inp)
        outdim = tmp_out.shape[0] if tmp_out.ndim==1 else tmp_out.shape[1]

        n_rows = len(grpA)
        n_cols = len(grpB)
        confusion_pred = np.zeros((n_rows, n_cols, outdim), dtype=float)

        row_iter = trange(n_rows, desc="Confusion Rows", ncols=100)
        for i in row_iter:
            rcid = grpA[i]
            for j in range(n_cols):
                ccid = grpB[j]
                sum_real = np.zeros(outdim, dtype=float)
                for (oh_tuple, freq) in observed_combos:
                    tmpv = base_vec.copy()
                    # 设置 oh_tuple
                    for local_oh_index, val01 in enumerate(oh_tuple):
                        gcol = get_onehot_global_col_index(local_oh_index, oh_index_map)
                        tmpv[gcol] = val01

                    # 强行把 groupA[i], groupB[j] => 1
                    tmpv[rcid] = 1
                    tmpv[ccid] = 1

                    if scaler_x:
                        tmp_inp = tmpv.reshape(1, -1)
                        tmp_inp[:, numeric_cols_idx] = scaler_x.transform(tmp_inp[:, numeric_cols_idx])
                        scaled_pred = model_predict(model, tmp_inp)
                    else:
                        scaled_pred = model_predict(model, tmpv.reshape(1, -1))

                    real_pred = inverse_transform_output(scaled_pred, scaler_y)
                    if real_pred.ndim==2:
                        sum_real += real_pred[0]
                    else:
                        sum_real += real_pred

                avg_real = sum_real / len(observed_combos)
                avg_real = np.clip(avg_real, 0, 100)
                confusion_pred[i, j, :] = avg_real

        np.save(os.path.join(outdir_m,"confusion_pred.npy"), confusion_pred)
        print(f"[INFO] confusion => shape={confusion_pred.shape}, saved => {outdir_m}")

if __name__=="__main__":
    inference_main()
