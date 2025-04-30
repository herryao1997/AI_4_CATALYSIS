"""
visualization.py

需求:
1) 从 postprocessing/<csv_name>/train 读取:
   - df_raw_14.csv => 做 correlation (普通) + DataAnalysis
   - X_onehot.npy => 做 correlation_heatmap_one_hot
   - Y_train.npy, Y_val.npy => 用于画散点/残差
   - 对每个模型 => 读取 train_pred.npy, val_pred.npy, metrics.pkl, train_losses.npy, val_losses.npy
     => 画散点、残差、MAE、MSE、Loss曲线、FeatureImportance(若有)
2) 从 postprocessing/<csv_name>/inference/<model_type> 读取:
   - heatmap_pred.npy, grid_x.npy, grid_y.npy => 2D heatmap
   - confusion_pred.npy => confusion-like
3) 输出图到 ./evaluation/figures/<csv_name>/...
4) 已去掉 K-Fold 逻辑.
"""

import os
import yaml
import numpy as np
import pandas as pd
import joblib

from utils import (
    ensure_dir,
    # nonlinear correlation
    plot_nonlinear_correlation_heatmap,
    # data analysis
    plot_kde_distribution,
    plot_catalyst_size_vs_product,
    plot_potential_vs_product_by_electrolyte,
    plot_product_distribution_by_catalyst_and_potential,
    plot_product_vs_potential_bin,
    plot_product_vs_shape,
    plot_product_vs_catalyst,
    plot_potential_vs_product,
    # model metrics
    plot_three_metrics_horizontal,
    plot_overfitting_horizontal,
    plot_loss_curve,
    plot_scatter_3d_outputs_mse,
    plot_scatter_3d_outputs_mae,
    plot_residual_histogram,
    plot_residual_kde,
    # shap analysis
    plot_shap_beeswarm,
    merge_onehot_shap,
    plot_shap_importance,
    # inference
    plot_2d_heatmap_from_npy,
    plot_confusion_from_npy,
    plot_3d_surface_from_heatmap,
    plot_3d_bars_from_confusion,
    ensure_dir,
    safe_filename
)

#####################################################
#  generate_shap_plots
#####################################################
def generate_shap_plots(csv_name, model_types):
    """
    读取 <model>/shap/shap_data.pkl → 合并 one-hot → 画 SHAP 重要性 & beeswarm
    """
    # ---- ① 读取一次 metadata.pkl ----
    meta_path = os.path.join("./models", "metadata.pkl")
    meta = joblib.load(meta_path)
    onehot_groups = meta["onehot_groups"]                 # [[7,8,9], …]
    case_map = {c.lower(): c for c in meta["x_col_names"]}  # 恢复大小写用

    for mtype in model_types:
        shap_dir = os.path.join(
            "./evaluation/figures", csv_name, "model_comparison", mtype, "shap"
        )
        ensure_dir(shap_dir)

        shap_pkl = os.path.join(shap_dir, "shap_data.pkl")
        if not os.path.exists(shap_pkl):
            print(f"[WARN] shap_data not found → {shap_pkl}")
            continue

        # ---- ② 读入 & 合并 one-hot ----
        raw_sd = joblib.load(shap_pkl)
        shap_data = merge_onehot_shap(raw_sd, onehot_groups, case_map)

        # ---- ③ 全局 Mean|SHAP| 条形图 ----
        try:
            plot_shap_importance(
                shap_data, shap_dir,
                top_n_features=15, plot_width=12, plot_height=8
            )
        except Exception as e:
            print(f"[WARN] importance plot failed ({mtype}): {e}")

        # ---- ④ Beeswarm ----
        try:
            plot_shap_beeswarm(
                shap_data, shap_dir,
                top_n_features=15, plot_width=12, plot_height=8
            )
        except Exception as e:
            print(f"[WARN] beeswarm plot failed ({mtype}): {e}")

def visualize_main():
    with open("./configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    csv_path = config["data"]["path"]
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]

    base_train = os.path.join("postprocessing", csv_name, "train")
    if not os.path.isdir(base_train):
        print(f"[WARN] train folder not found => {base_train}")
        return

    # ========== 1.1) df_raw_14.csv & data_corr_dir ==========
    raw_csv_path = os.path.join(base_train, "df_raw_14.csv")
    data_corr_dir = os.path.join("./evaluation/figures", csv_name, "DataCorrelation")
    ensure_dir(data_corr_dir)

    if os.path.exists(raw_csv_path):
        df_raw_14 = pd.read_csv(raw_csv_path)
        # phik correlation
        if config["evaluation"].get("save_correlation", False):
            fn1 = os.path.join(data_corr_dir, "correlation_heatmap.jpg")
            numeric_cols_14 = df_raw_14.select_dtypes(include=[np.number]).columns.tolist()

            plot_nonlinear_correlation_heatmap(
                df_raw_14,
                filename=fn1,
                cmap="ocean",
                vmin=0, vmax=1,
                method = "mic"
            )

        # 数据分析图
        if config["evaluation"].get("save_data_analysis_plots", False):
            possible_cols = ["Potential (V vs. RHE)", "H2", "CO", "C1", "C2+", "Particle size (log scale)", "Particle size (nm)"]
            existing_cols = [c for c in possible_cols if c in df_raw_14.columns]
            if existing_cols:
                out_kde = os.path.join(data_corr_dir, "kde_distribution.jpg")
                plot_kde_distribution(df_raw_14, existing_cols, filename=out_kde)

            out_cat = os.path.join(data_corr_dir, "catalyst_size_vs_product.jpg")
            plot_catalyst_size_vs_product(df_raw_14, filename=out_cat)

            out_pp = os.path.join(data_corr_dir, "potential_vs_product_by_electrolyte.jpg")
            plot_potential_vs_product_by_electrolyte(df_raw_14, filename=out_pp)

            out_prod = os.path.join(data_corr_dir, "product_distribution.jpg")
            plot_product_distribution_by_catalyst_and_potential(df_raw_14, filename=out_prod)

            out_box_pot = os.path.join(data_corr_dir, "box_product_vs_potential_bin.jpg")
            plot_product_vs_potential_bin(df_raw_14, filename=out_box_pot)

            out_box_shape = os.path.join(data_corr_dir, "box_product_vs_shape.jpg")
            plot_product_vs_shape(df_raw_14, filename=out_box_shape)

            out_box_cat = os.path.join(data_corr_dir, "box_product_vs_catalyst.jpg")
            plot_product_vs_catalyst(df_raw_14, filename=out_box_cat)

            out_three = os.path.join(data_corr_dir, "three_dot_potential_vs_product.jpg")
            plot_potential_vs_product(df_raw_14, filename=out_three)
    else:
        print(f"[WARN] df_raw_14.csv not found => {raw_csv_path}")


    # ========== 1.3) Y_train.npy, Y_val.npy ==========
    y_train_path = os.path.join(base_train, "Y_train.npy")
    y_val_path = os.path.join(base_train, "Y_val.npy")
    Y_train = np.load(y_train_path) if os.path.exists(y_train_path) else None
    Y_val = np.load(y_val_path) if os.path.exists(y_val_path) else None

    # ========== 1.4) 针对每个模型 ==========
    model_types = config["model"]["types"]
    for mtype in model_types:
        model_subdir = os.path.join(base_train, mtype)
        if not os.path.isdir(model_subdir):
            print(f"[WARN] no train folder for model type => {model_subdir}")
            continue

        metrics_pkl = os.path.join(model_subdir, "metrics.pkl")
        train_pred_path = os.path.join(model_subdir, "train_pred.npy")
        val_pred_path = os.path.join(model_subdir, "val_pred.npy")
        train_loss_path = os.path.join(model_subdir, "train_losses.npy")
        val_loss_path = os.path.join(model_subdir, "val_losses.npy")

        train_metrics = None
        val_metrics = None
        if os.path.exists(metrics_pkl):
            data_ = joblib.load(metrics_pkl)
            train_metrics = data_.get("train_metrics", None)
            val_metrics = data_.get("val_metrics", None)
            print(f"[{mtype}] train_metrics={train_metrics}, val_metrics={val_metrics}")

        train_pred = np.load(train_pred_path) if os.path.exists(train_pred_path) else None
        val_pred = np.load(val_pred_path) if os.path.exists(val_pred_path) else None
        train_losses = np.load(train_loss_path) if os.path.exists(train_loss_path) else None
        val_losses = np.load(val_loss_path) if os.path.exists(val_loss_path) else None

        # 读取 y_col_names (若存在)
        model_dir = os.path.join("./models", mtype)
        ycol_path = os.path.join(model_dir, "y_col_names.npy")
        if os.path.exists(ycol_path):
            y_cols = list(np.load(ycol_path, allow_pickle=True))
        else:
            y_cols = None

        model_comp_dir = os.path.join("./evaluation/figures", csv_name, "model_comparison", mtype)
        ensure_dir(model_comp_dir)

        # (a) 绘制 Loss
        if train_losses is not None and val_losses is not None and config["evaluation"].get("save_loss_curve", False):
            out_lc = os.path.join(model_comp_dir, f"{mtype}_loss_curve.jpg")
            plot_loss_curve(train_losses, val_losses, filename=out_lc)

        # (b) 绘制散点 & 残差
        if (Y_train is not None and Y_val is not None) and (train_pred is not None and val_pred is not None):
            if config["evaluation"].get("save_scatter_mse_plot", False):
                out_mse_tr = os.path.join(model_comp_dir, "full", "train", f"{mtype}_mse_scatter_train.jpg")
                ensure_dir(os.path.dirname(out_mse_tr))
                plot_scatter_3d_outputs_mse(Y_train, train_pred, y_labels=y_cols, filename=out_mse_tr)

                out_mse_val = os.path.join(model_comp_dir, "full", "valid", f"{mtype}_mse_scatter.jpg")
                ensure_dir(os.path.dirname(out_mse_val))
                plot_scatter_3d_outputs_mse(Y_val, val_pred, y_labels=y_cols, filename=out_mse_val)

            if config["evaluation"].get("save_scatter_mae_plot", False):
                out_mae_tr = os.path.join(model_comp_dir, "full", "train", f"{mtype}_mae_scatter_train.jpg")
                ensure_dir(os.path.dirname(out_mae_tr))
                plot_scatter_3d_outputs_mae(Y_train, train_pred, y_labels=y_cols, filename=out_mae_tr)

                out_mae_val = os.path.join(model_comp_dir, "full", "valid", f"{mtype}_mae_scatter.jpg")
                ensure_dir(os.path.dirname(out_mae_val))
                plot_scatter_3d_outputs_mae(Y_val, val_pred, y_labels=y_cols, filename=out_mae_val)

            if config["evaluation"].get("save_residual_hist", False):
                out_hist_tr = os.path.join(model_comp_dir, "full", "train", f"{mtype}_residual_hist_train.jpg")
                ensure_dir(os.path.dirname(out_hist_tr))
                plot_residual_histogram(Y_train, train_pred, y_labels=y_cols, filename=out_hist_tr)

                out_hist_val = os.path.join(model_comp_dir, "full", "valid", f"{mtype}_residual_hist.jpg")
                ensure_dir(os.path.dirname(out_hist_val))
                plot_residual_histogram(Y_val, val_pred, y_labels=y_cols, filename=out_hist_val)

            if config["evaluation"].get("save_residual_kde", False):
                out_kde_tr = os.path.join(model_comp_dir, "full", "train", f"{mtype}_residual_kde_train.jpg")
                ensure_dir(os.path.dirname(out_kde_tr))
                plot_residual_kde(Y_train, train_pred, y_labels=y_cols, filename=out_kde_tr)

                out_kde_val = os.path.join(model_comp_dir, "full", "valid", f"{mtype}_residual_kde.jpg")
                ensure_dir(os.path.dirname(out_kde_val))
                plot_residual_kde(Y_val, val_pred, y_labels=y_cols, filename=out_kde_val)

    # ========== 2) 汇总多个模型的 metrics ==========
    train_metrics_dict = {}
    val_metrics_dict = {}
    for mtype in model_types:
        mdir = os.path.join(base_train, mtype)
        mpkl = os.path.join(mdir, "metrics.pkl")
        if os.path.exists(mpkl):
            data_ = joblib.load(mpkl)
            train_metrics_dict[mtype] = data_.get("train_metrics", {})
            val_metrics_dict[mtype] = data_.get("val_metrics", {})

    if train_metrics_dict or val_metrics_dict:
        if train_metrics_dict:
            out_3train = os.path.join(data_corr_dir, "three_metrics_horizontal_train.jpg")
            plot_three_metrics_horizontal(train_metrics_dict, save_name=out_3train)

        if val_metrics_dict:
            out_3val = os.path.join(data_corr_dir, "three_metrics_horizontal_val.jpg")
            plot_three_metrics_horizontal(val_metrics_dict, save_name=out_3val)

        if config["evaluation"].get("save_models_evaluation_bar", False):
            if train_metrics_dict and val_metrics_dict:
                overfit_data = {}
                for m in train_metrics_dict:
                    trm = train_metrics_dict[m]
                    vam = val_metrics_dict[m]
                    ms_ratio = float("inf") if trm["MSE"] == 0 else vam["MSE"] / trm["MSE"]
                    r2_diff = trm["R2"] - vam["R2"]
                    overfit_data[m] = {"MSE_ratio": ms_ratio, "R2_diff": r2_diff}

                out_of = os.path.join(data_corr_dir, "overfitting_single.jpg")
                plot_overfitting_horizontal(overfit_data, save_name=out_of)

    # ========== 3) 推理可视化 (Heatmap + Confusion) ==========
    base_inf = os.path.join("postprocessing", csv_name, "inference")
    inf_models = config["inference"].get("models", [])

    # 从 metadata.pkl 中读取 stats_dict["continuous_cols"]
    metadata_path = os.path.join("./models", "metadata.pkl")
    if os.path.exists(metadata_path):
        meta_data = joblib.load(metadata_path)
        stats_dict = meta_data.get("continuous_cols", {})
    else:
        stats_dict = {}

    # 读取 config
    heatmap_x_label = config["inference"]["heatmap_axes"]["x_name"]
    heatmap_y_label = config["inference"]["heatmap_axes"]["y_name"]
    confusion_row_axis = config["inference"]["confusion_axes"]["row_name"]
    confusion_col_axis = config["inference"]["confusion_axes"]["col_name"]

    for mtype in inf_models:
        inf_dir = os.path.join(base_inf, mtype)
        if not os.path.isdir(inf_dir):
            print(f"[WARN] no inference dir => {inf_dir}")
            continue

        heatmap_path = os.path.join(inf_dir, "heatmap_pred.npy")
        gridx_path   = os.path.join(inf_dir, "grid_x.npy")
        gridy_path   = os.path.join(inf_dir, "grid_y.npy")
        confusion_path = os.path.join(inf_dir, "confusion_pred.npy")

        base_out = os.path.join("./evaluation/figures", csv_name, "inference", mtype)
        ensure_dir(base_out)

        # Heatmap
        if os.path.exists(heatmap_path) and os.path.exists(gridx_path) and os.path.exists(gridy_path):
            heatmap_pred = np.load(heatmap_path)
            grid_x = np.load(gridx_path)
            grid_y = np.load(gridy_path)

            model_dir = os.path.join("./models", mtype)
            ycol_path = os.path.join(model_dir, "y_col_names.npy")
            if os.path.exists(ycol_path):
                y_cols = list(np.load(ycol_path, allow_pickle=True))
            else:
                y_cols = None

            out_hm = os.path.join(base_out, "2d_heatmap")
            ensure_dir(out_hm)

            plot_2d_heatmap_from_npy(
                grid_x, grid_y, heatmap_pred,
                out_dir=out_hm,
                x_label=heatmap_x_label,
                y_label=heatmap_y_label,
                y_col_names=y_cols,
                # stats_dict=stats_dict,
                colorbar_extend_ratio=0.02
            )

            # 3D Surface
            plot_3d_surface_from_heatmap(
                grid_x, grid_y, heatmap_pred,
                out_dir=out_hm,
                x_label=heatmap_x_label,
                y_label=heatmap_y_label,
                y_col_names=y_cols,
                # stats_dict=stats_dict,
                colorbar_extend_ratio=0.02,
                cmap_name="viridis"
            )

        # Confusion
        if os.path.exists(confusion_path):
            confusion_pred = np.load(confusion_path)

            meta_path = os.path.join("./models", "metadata.pkl")
            if os.path.exists(meta_path):
                meta_data2 = joblib.load(meta_path)
                oh_groups = meta_data2.get("onehot_groups", [])
            else:
                oh_groups = []

            if len(oh_groups) >= 2:
                grpA, grpB = oh_groups[:2]

                xcol_path = os.path.join("./models", mtype, "x_col_names.npy")
                if os.path.exists(xcol_path):
                    xcols = list(np.load(xcol_path, allow_pickle=True))
                else:
                    xcols = None

                row_labels = [xcols[cid] for cid in grpA] if xcols else [f"Class {i+1}" for i in range(len(grpA))]
                col_labels = [xcols[cid] for cid in grpB] if xcols else [f"Class {i+1}" for i in range(len(grpB))]

                ycol_path = os.path.join("./models", mtype, "y_col_names.npy")
                if os.path.exists(ycol_path):
                    y_cols = list(np.load(ycol_path, allow_pickle=True))
                else:
                    y_cols = None

                out_conf = os.path.join(base_out, "confusion_matrix")
                ensure_dir(out_conf)

                plot_confusion_from_npy(
                    confusion_pred,
                    row_labels, col_labels,
                    out_dir=out_conf,
                    y_col_names=y_cols,
                    # stats_dict=stats_dict,
                    cell_scale=0.25,
                    colorbar_extend_ratio=0.02,
                    row_axis_name=confusion_row_axis,
                    col_axis_name=confusion_col_axis
                )

                plot_3d_bars_from_confusion(
                    confusion_pred,
                    row_labels, col_labels,
                    out_dir=out_conf,
                    y_col_names=y_cols,
                    # stats_dict=stats_dict,
                    colorbar_extend_ratio=0.02,
                    cmap_name="viridis"#"rainbow" "viridis"
                )
            else:
                print("[WARN] Not enough onehot groups => skip confusion matrix.")
    # ========== 4) SHAP 可解释性图 ==========
    # 利用配置中的模型类型（和训练时保持一致）生成每个模型的 SHAP 图
    model_types = config["model"]["types"]
    generate_shap_plots(csv_name, model_types)

    print("\n[INFO] visualize_main => done.")


if __name__ == "__main__":
    visualize_main()
