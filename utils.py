"""
utils.py

包含所有绘图函数 & 一些辅助:
1) correlation_heatmap (含普通 & onehot)
2) 训练可视化: loss_curve, scatter(MAE/MSE), residual, feature_importance, etc.
3) 原始数据分析(kde, scatter, boxplot)
4) 推理可视化(2D Heatmap + ConfusionMatrix)
5) 混淆矩阵中在每个三角形内显示数值 + colorbar范围扩展 + 保持正方形布局.

已去掉K-Fold, 保留注释.
"""

import os
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.collections import PolyCollection
import pandas as pd
import math
import scipy.stats as ss
from matplotlib.patches import Patch
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr, pointbiserialr
from matplotlib.ticker import MaxNLocator, FormatStrFormatter, LinearLocator  # 如果文件顶部已经导入可省略
import matplotlib.ticker as ticker
import shap

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def ensure_dir_for_file(filepath):
    dir_ = os.path.dirname(filepath)
    if dir_:
        os.makedirs(dir_, exist_ok=True)

def normalize_data(data, vmin, vmax):
    """归一化数据到 [0,1] 范围"""
    return (data - vmin) / (vmax - vmin) if vmax > vmin else data

# --------------- correlation ---------------
def safe_filename(s):
    # 简单过滤：去除特殊字符
    return "".join(c if c.isalnum() or c in (' ','.','_') else "_" for c in s).strip()

def short_label_normal(s: str) -> str:
    """
    根据下划线将字符串分割，取最后一段，并做以下处理：
      - 若最后一段全是大写 (如 "CO", "OH")，则原样返回；
      - 对特定化学符号使用 LaTeX 数学模式进行下标渲染，例如将 "cu2s" 转为 "$Cu_{2}S$"；
      - 其余情况：将首字母大写，其他部分保持原状。
    """
    special_chemicals = {
        "cu": "$Cu$",
        "cu(oh)2": "$Cu(OH)_2$",
        "cuxo": "$Cu_{X}O$",  # 如果有多个可能，用你需要的格式
        "cu2s": "$Cu_{2}S$",
        "cu2(oh)2co3": "$Cu_{2}(OH)_{2}CO_{3}$",
        "c2+": "$C_{2+}$",
        "c1": "$C_{1}$",
        "h2": "$H_{2}$"
    }

    parts = s.split('_')
    last_part = parts[-1]  # 取最后一段

    if not last_part:
        return last_part

    lower_last_part = last_part.lower()
    if lower_last_part in special_chemicals:
        return special_chemicals[lower_last_part]

    if last_part.isupper():
        return last_part

    return last_part[0].upper() + last_part[1:]

def short_label_bold(s: str) -> str:
    """
    根据下划线将字符串分割，取最后一段，并做以下处理：
      - 若最后一段全是大写 (如 "CO", "OH")，则原样返回；
      - 对特定化学符号使用 LaTeX 数学模式进行下标渲染，例如将 "cu2s" 转为 "$Cu_{2}S$"；
      - 其余情况：将首字母大写，其他部分保持原状。
    """
    special_chemicals = {
        "cu": "$\\mathbf{Cu}$",
        "cu(oh)2": "$\\mathbf{Cu(OH)}_{2}$",
        "cuxo": "$\\mathbf{Cu_{X}O}$",
        "cu2s": "$\\mathbf{Cu_{2}S}$",
        "cu2(oh)2co3": "$\\mathbf{Cu_{2}(OH)_{2}CO_{3}}$",
        "c2+": "$\\mathbf{C_{2+}}$",
        "c1": "$\\mathbf{C_{1}}$",
        "h2": "$\\mathbf{H_{2}}$"
    }

    parts = s.split('_')
    last_part = parts[-1]  # 取最后一段

    if not last_part:
        return last_part

    lower_last_part = last_part.lower()
    if lower_last_part in special_chemicals:
        return special_chemicals[lower_last_part]

    if last_part.isupper():
        return last_part

    return last_part[0].upper() + last_part[1:]


# --------------- 相关性可视化---------------
# -------------------- reviewer 1 --------------------
# ---------- Non-linear correlation heatmap ----------
def plot_nonlinear_correlation_heatmap(df: pd.DataFrame,
                                       filename: str,
                                       vmin: float = 0.0,
                                       vmax: float = 1.0,
                                       cmap: str = "viridis",
                                       method: str = "distance"):
    """
    计算并绘制非线性相关性热图（Distance Correlation 或 MIC）
    """
    from itertools import combinations
    ensure_dir_for_file(filename)

    # ---------- internal helpers ----------
    def _to_numeric(series: pd.Series):
        if pd.api.types.is_numeric_dtype(series):
            return series.to_numpy(dtype=float)
        return series.astype("category").cat.codes.to_numpy(dtype=float)

    def _distance_corr(x, y):
        try:
            import dcor
            return float(dcor.distance_correlation(x, y))
        except ImportError:
            import numpy as np
            n = x.size
            a = np.abs(x[:, None] - x[None, :])
            b = np.abs(y[:, None] - y[None, :])
            A = a - a.mean(0)[None, :] - a.mean(1)[:, None] + a.mean()
            B = b - b.mean(0)[None, :] - b.mean(1)[:, None] + b.mean()
            dcov_xy = (A * B).mean()
            dcov_xx = (A * A).mean()
            dcov_yy = (B * B).mean()
            return 0.0 if dcov_xx == 0 or dcov_yy == 0 else np.sqrt(dcov_xy) / np.sqrt(np.sqrt(dcov_xx * dcov_yy))

    def _mic(x, y):
        try:
            from minepy import MINE
        except ImportError as e:
            raise ImportError("计算 MIC 需先安装 minepy：pip install minepy") from e
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(x, y)
        return mine.mic()

    # ---------- correlation matrix ----------
    cols = df.columns.tolist()
    n = len(cols)
    corr_mat = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        corr_mat[i, i] = 1.0
        xi = _to_numeric(df[cols[i]].dropna())
        for j in range(i + 1, n):
            xj = _to_numeric(df[cols[j]].dropna())
            m = min(len(xi), len(xj))
            x_arr, y_arr = xi[:m], xj[:m]
            val = _mic(x_arr, y_arr) if method == "mic" else _distance_corr(x_arr, y_arr)
            corr_mat[i, j] = corr_mat[j, i] = max(min(val, 1.0), 0.0)

    # ---------- plot ----------
    labels = [short_label_bold(c) for c in cols]
    fig, ax = plt.subplots(figsize=(max(10, 0.5 * n), max(8, 0.5 * n)))
    hm = sns.heatmap(
        corr_mat,
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        vmin=vmin, vmax=vmax,
        square=True,
        cbar_kws={"shrink": 0.8, "aspect": 30,
                  "label": f"{'Distance' if method=='distance' else 'MIC'} Correlation"},
        ax=ax
    )

    ax.set_title(f"{'Distance' if method=='distance' else 'MIC'} Correlation Heatmap",
                 fontsize=16, fontweight='bold')                                    # ← font
    plt.xticks(rotation=45, ha="right", fontsize=15, fontweight='bold')             # ← font
    plt.yticks(fontsize=15, fontweight='bold')                                      # ← font

    # colour-bar font tweaks
    cbar = hm.collections[0].colorbar
    cbar.set_label(cbar.ax.get_ylabel(), fontsize=16, fontweight='bold')            # ← font
    for t in cbar.ax.get_yticklabels():                                             # ← font
        t.set_fontsize(15)
        t.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()
    print(f"[plot_nonlinear_correlation_heatmap] => {filename}")


# --------------- Training visualisation ---------------
def plot_loss_curve(train_losses, val_losses, filename):
    ensure_dir_for_file(filename)
    plt.figure()
    plt.plot(train_losses, label='Train Loss', linewidth=2)
    plt.plot(val_losses, label='Val Loss',   linewidth=2)
    plt.xlabel("Epoch", fontsize=15, fontweight='bold')                            # ← font
    plt.ylabel("Loss",  fontsize=15, fontweight='bold')                            # ← font
    plt.title("Training / Validation Loss", fontsize=15, fontweight='bold')        # ← font
    plt.legend(prop={'size': 13, 'weight': 'bold'})                                # ← font
    ax = plt.gca()
    for t in ax.get_xticklabels() + ax.get_yticklabels():                          # ← font
        t.set_fontsize(14)
        t.set_fontweight('bold')
    plt.savefig(filename, dpi=700, format='jpg')
    plt.close()


def _set_axis_tick_fonts(ax):                                                      # helper
    for t in ax.get_xticklabels():
        t.set_fontsize(15)
        t.set_fontweight('bold')
    for t in ax.get_yticklabels():
        t.set_fontsize(15)
        t.set_fontweight('bold')


def plot_scatter_3d_outputs_mse(y_true, y_pred, y_labels=None,
                                filename="scatter_3d_mse.jpg"):
    ensure_dir_for_file(filename)
    if y_pred.ndim != 2:
        raise ValueError("y_pred must be 2D (N, out_dim)")
    _, out_dim = y_pred.shape
    fig, axes = plt.subplots(1, out_dim, figsize=(4 * out_dim, 4), squeeze=False)

    for i in range(out_dim):
        errors = (y_true[:, i] - y_pred[:, i]) ** 2
        r2_val = r2_score(y_true[:, i], y_pred[:, i])
        ax = axes[0, i]
        sc = ax.scatter(y_true[:, i], y_pred[:, i], c=errors, alpha=0.5, cmap='brg')
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

        if y_labels and i < len(y_labels):
            label = short_label_bold(y_labels[i])
            ax.set_title(f"{label} (MSE)\nR²={r2_val:.3f}", fontsize=14, fontweight='bold')
            ax.set_xlabel(f"True {label}",  fontsize=16, fontweight='bold')
            ax.set_ylabel(f"Pred {label}",  fontsize=16, fontweight='bold')
        else:
            ax.set_title(f"Out {i} (MSE)\nR²={r2_val:.3f}", fontsize=14, fontweight='bold')

        _set_axis_tick_fonts(ax)                                                   # ← font

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Squared Error", fontsize=16, fontweight='bold')            # ← font
        for tick in cbar.ax.get_yticklabels():                                     # ← font
            tick.set_fontsize(15)
            tick.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()


def plot_scatter_3d_outputs_mae(y_true, y_pred, y_labels=None,
                                filename="scatter_3d_mae.jpg"):
    ensure_dir_for_file(filename)
    if y_pred.ndim != 2:
        raise ValueError("y_pred must be 2D (N, out_dim)")
    _, out_dim = y_pred.shape
    fig, axes = plt.subplots(1, out_dim, figsize=(4 * out_dim, 4), squeeze=False)

    for i in range(out_dim):
        errors = np.abs(y_true[:, i] - y_pred[:, i])
        r2_val = r2_score(y_true[:, i], y_pred[:, i])
        ax = axes[0, i]
        sc = ax.scatter(y_true[:, i], y_pred[:, i], c=errors, alpha=0.5, cmap='ocean')
        min_val = min(y_true[:, i].min(), y_pred[:, i].min())
        max_val = max(y_true[:, i].max(), y_pred[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5)

        if y_labels and i < len(y_labels):
            label = short_label_bold(y_labels[i])
            ax.set_title(f"{label} (MAE)\nR²={r2_val:.3f}", fontsize=14, fontweight='bold')
            ax.set_xlabel(f"True {label}",  fontsize=16, fontweight='bold')
            ax.set_ylabel(f"Pred {label}",  fontsize=16, fontweight='bold')
        else:
            ax.set_title(f"Out {i} (MAE)\nR²={r2_val:.3f}", fontsize=14, fontweight='bold')

        _set_axis_tick_fonts(ax)                                                   # ← font

        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("Absolute Error", fontsize=16, fontweight='bold')           # ← font
        for tick in cbar.ax.get_yticklabels():                                     # ← font
            tick.set_fontsize(15)
            tick.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()


# ---- Residual histogram ------------------------------------------------
def plot_residual_histogram(y_true, y_pred, y_labels=None,
                            cmap_name="coolwarm",
                            vmin=-70, vmax=70,
                            filename="residual_hist_bottom.jpg"):
    ensure_dir_for_file(filename)
    residuals = y_true - y_pred
    n_outputs = residuals.shape[1]

    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4.5))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88, wspace=0.3)

    num_bins = 30
    bins_array = np.linspace(vmin, vmax, num_bins + 1)

    cmap_obj = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(n_outputs):
        ax = axes[i] if n_outputs > 1 else axes
        hist_data, bin_edges, patches = ax.hist(
            residuals[:, i], bins=bins_array, alpha=0.9, edgecolor='none'
        )
        for b_idx, patch in enumerate(patches):
            bin_center = 0.5 * (bin_edges[b_idx] + bin_edges[b_idx + 1])
            patch.set_facecolor(cmap_obj(norm(bin_center)))

        title_txt = (f"Residuals of {short_label_bold(y_labels[i])}"
                     if y_labels and i < len(y_labels)
                     else f"Output {i} Residual")
        ax.set_title(title_txt, fontsize=14, fontweight='bold')

        ax.set_xlabel("Residual", fontsize=16, fontweight='bold')
        ax.set_ylabel("Count",    fontsize=16, fontweight='bold')
        ax.set_xlim(vmin, vmax)
        _set_axis_tick_fonts(ax)

    # ---- colour-bar ----
    sm = cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        orientation="horizontal", fraction=0.07,
                        pad=0.20, shrink=0.9)
    cbar.set_label("Residual Value", fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=15, width=2)

    # Make colour-bar tick labels **bold**
    for t in cbar.ax.get_xticklabels():
        t.set_fontweight('bold')

    plt.savefig(filename, dpi=700)
    plt.close()
    print(f"[plot_residual_histogram] => {filename}")


# ---- Residual KDE ------------------------------------------------------
class MyScalarFormatter(ticker.ScalarFormatter):
    def __init__(self, useMathText=True):
        super().__init__(useMathText=useMathText)
    def _set_format(self):
        self.format = '%.1f'


def plot_residual_kde(
    y_true, y_pred, y_labels=None,
    cmap_name="coolwarm",
    vmin=-70, vmax=70,
    filename="residual_kde_bottom.jpg"
):
    ensure_dir_for_file(filename)
    residuals = y_true - y_pred
    n_outputs = residuals.shape[1]

    fig, axes = plt.subplots(1, n_outputs, figsize=(4 * n_outputs, 4.5))
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.88, wspace=0.3)

    cmap_obj = cm.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    for i in range(n_outputs):
        ax = axes[i] if n_outputs > 1 else axes
        sns.kdeplot(residuals[:, i], ax=ax, fill=False, color="black")
        line = ax.get_lines()[-1] if ax.get_lines() else None
        if line is None:
            continue
        x_plot, y_plot = line.get_xdata(), line.get_ydata()
        idxsort = np.argsort(x_plot)
        x_plot, y_plot = x_plot[idxsort], y_plot[idxsort]

        for j in range(len(x_plot) - 1):
            x0, x1 = x_plot[j], x_plot[j + 1]
            y0, y1 = y_plot[j], y_plot[j + 1]
            verts = np.array([[x0, 0], [x0, y0], [x1, y1], [x1, 0]])
            poly = PolyCollection([verts],
                                  facecolors=[cmap_obj(norm(0.5 * (x0 + x1)))],
                                  edgecolor='none', alpha=0.6)
            ax.add_collection(poly)

        title_txt = (f"Residual KDE of {short_label_bold(y_labels[i])}"
                     if y_labels and i < len(y_labels)
                     else f"KDE - Output {i}")
        ax.set_title(title_txt, fontsize=14, fontweight='bold')

        ax.set_xlabel("Residual", fontsize=16, fontweight='bold')
        ax.set_ylabel("Density",  fontsize=16, fontweight='bold')
        ax.set_xlim(vmin, vmax)
        _set_axis_tick_fonts(ax)

        ax.yaxis.set_major_locator(ticker.LinearLocator(5))
        my_formatter = MyScalarFormatter(useMathText=True)
        my_formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(my_formatter)
        ax.yaxis.get_offset_text().set_fontsize(16)
        ax.yaxis.get_offset_text().set_fontweight('bold')

    # ---- colour-bar ----
    sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(),
                        orientation="horizontal", fraction=0.07,
                        pad=0.20, shrink=0.9)
    cbar.set_label("Residual Value", fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=15, width=2)

    # Make colour-bar tick labels **bold**
    for t in cbar.ax.get_xticklabels():
        t.set_fontweight('bold')

    plt.savefig(filename, dpi=700)
    plt.close()
    print(f"[plot_residual_kde] => {filename}")



# --------------- Three-metric comparison ---------------
# --------------- Three-metric comparison ---------------
def plot_three_metrics_horizontal(metrics_data, save_name="three_metrics.jpg"):
    ensure_dir_for_file(save_name)
    model_names = list(metrics_data.keys())
    mse_vals = [metrics_data[m]["MSE"] for m in model_names]
    mae_vals = [metrics_data[m]["MAE"] for m in model_names]
    r2_vals  = [metrics_data[m]["R2"]  for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    def plot_hbar_with_mean(ax, model_names, values, subplot_label,
                            metric_label, bigger_is_better=False):
        arr = np.array(values)
        best_idx  = arr.argmax() if bigger_is_better else arr.argmin()
        worst_idx = arr.argmin() if bigger_is_better else arr.argmax()

        colors = ["red" if i == best_idx else
                  "blue" if i == worst_idx else
                  "green" for i in range(len(arr))]

        mean_val    = arr.mean()
        y_positions = np.arange(len(arr))
        ax.barh(y_positions, arr, color=colors, alpha=0.8, height=0.4)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(model_names, fontsize=15, fontweight='bold')
        ax.invert_yaxis()

        ax.text(-0.08, 1.05, subplot_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=16, fontweight="bold")
        ax.set_title(metric_label, fontsize=14, fontweight='bold')

        for i, vv in enumerate(arr):
            ax.text(vv, i, f"{vv:.2f}",
                    ha="left" if vv >= 0 else "right",
                    va="center", fontsize=10)

        ax.axvline(mean_val, color='gray', linestyle='--', linewidth=2)
        ax.axvspan(*sorted([0, mean_val]), facecolor='gray', alpha=0.2)

        ax.set_xlim(arr.min() * 1.1 if arr.min() < 0 else 0, arr.max() * 1.79)

        # ---- NEW: cap major x-ticks to ≤ 6 ----
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

        _set_axis_tick_fonts(ax)

        legend_e = [
            Patch(facecolor="red",   label="Best"),
            Patch(facecolor="blue",  label="Worst"),
            Patch(facecolor="green", label="Ordinary"),
            Patch(facecolor="gray", alpha=0.2, label="Under Mean"),
        ]
        ax.legend(handles=legend_e, loc="lower right",
                  prop={'size': 11, 'weight': 'bold'})

    plot_hbar_with_mean(axes[0], model_names, mse_vals, "a",
                        "MSE (Lower=Better)", bigger_is_better=False)
    plot_hbar_with_mean(axes[1], model_names, mae_vals, "b",
                        "MAE (Lower=Better)", bigger_is_better=False)
    plot_hbar_with_mean(axes[2], model_names, r2_vals, "c",
                        "R² (Higher=Better)", bigger_is_better=True)

    plt.tight_layout()
    plt.savefig(save_name, dpi=700)
    plt.close()
    print(f"[plot_three_metrics_horizontal] => {save_name}")


# --------------- Over-fitting diagnostics ---------------
def plot_overfitting_horizontal(overfit_data, save_name="overfitting_horizontal.jpg"):
    ensure_dir_for_file(save_name)
    model_names = list(overfit_data.keys())
    msr_vals = [overfit_data[m]["MSE_ratio"] for m in model_names]
    r2d_vals = [overfit_data[m]["R2_diff"]   for m in model_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def plot_hbar_threshold(ax, model_names, values, subplot_label, metric_label,
                            bigger_is_better=False, threshold_h=0.5, threshold_l=0.0):
        arr = np.array(values)
        best_idx  = arr.argmax() if bigger_is_better else arr.argmin()
        worst_idx = arr.argmin() if bigger_is_better else arr.argmax()

        colors = ["red" if i == best_idx else
                  "blue" if i == worst_idx else
                  "green" for i in range(len(arr))]

        y_positions = np.arange(len(arr))
        ax.barh(y_positions, arr, color=colors, alpha=0.8, height=0.4)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(model_names, fontsize=15, fontweight='bold')
        ax.invert_yaxis()

        ax.text(-0.08, 1.05, subplot_label, transform=ax.transAxes,
                ha="left", va="top", fontsize=16, fontweight="bold")
        ax.set_title(metric_label, fontsize=14, fontweight='bold')

        for i, vv in enumerate(arr):
            ax.text(vv, i, f"{vv:.2f}",
                    ha="left" if vv >= 0 else "right",
                    va="center", fontsize=10)

        # Threshold shading
        if threshold_l == 0.0:
            ax.axvspan(threshold_l, threshold_h, facecolor='gray', alpha=0.2)
            legend_e = [Patch(facecolor='gray', alpha=0.2, label="Acceptable")]
        else:
            ax.axvspan(0, threshold_l,              facecolor='gray',       alpha=0.2)
            ax.axvspan(threshold_l, threshold_h,    facecolor='lightcoral', alpha=0.3)
            ax.axvline(threshold_l, color='gray', linestyle='--', linewidth=2)
            ax.axvline(threshold_h, color='gray', linestyle='--', linewidth=2)
            legend_e = [
                Patch(facecolor='gray',      alpha=0.2, label="Acceptable"),
                Patch(facecolor='lightcoral', alpha=0.3, label="Overfitting Risk")
            ]

        max_val, min_val = arr.max(), arr.min()
        ax.set_xlim(min_val * 1.1 if min_val < 0 else 0, max_val * 2.5)

        # ---- NEW: cap major x-ticks to ≤ 6 ----
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

        for t in ax.get_xticklabels():
            t.set_fontsize(15)
            t.set_fontweight('bold')

        legend_e.extend([
            Patch(facecolor="red",   label="Best"),
            Patch(facecolor="blue",  label="Worst"),
            Patch(facecolor="green", label="Ordinary")
        ])
        ax.legend(handles=legend_e, loc="lower right",
                  prop={'size': 11, 'weight': 'bold'})

    plot_hbar_threshold(axes[0], model_names, msr_vals, "a",
                        "MSE Ratio (Val/Train)\n(Lower=Better)",
                        bigger_is_better=False, threshold_h=10, threshold_l=5)
    plot_hbar_threshold(axes[1], model_names, r2d_vals, "b",
                        "R2 diff (Train - Val)\n(Lower=Better)",
                        bigger_is_better=False, threshold_h=0.2, threshold_l=0.15)

    plt.tight_layout()
    plt.savefig(save_name, dpi=700)
    plt.close()
    print(f"[plot_overfitting_horizontal] => {save_name}")

# --------------- 原始数据分析 ---------------
def plot_kde_distribution(df, columns, filename):
    ensure_dir_for_file(filename)
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 5))
    if len(columns) == 1:
        axes = [axes]

    for i, col in enumerate(columns):
        ax = axes[i]
        col_label = short_label_bold(col)
        if col not in df.columns:
            ax.text(0.5, 0.5, f"'{col_label}' not in df", ha='center',
                    va='center', fontsize=16, fontweight='bold')
            continue

        sns.kdeplot(df[col], ax=ax, fill=False, color="black",
                    clip=(df[col].min(), df[col].max()))
        lines = ax.get_lines()
        if not lines:
            ax.set_title(f"No Data for {col_label}",
                         fontsize=14, fontweight='bold')
            continue

        # Colour-fill
        line = lines[-1]
        x_plot, y_plot = line.get_xdata(), line.get_ydata()
        idxsort = np.argsort(x_plot)
        x_plot, y_plot = x_plot[idxsort], y_plot[idxsort]

        vmin = max(np.min(x_plot), df[col].min())
        vmax = min(np.max(x_plot), df[col].max())
        cmap_ = cm.get_cmap("coolwarm")
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

        for j in range(len(x_plot) - 1):
            x0, x1 = x_plot[j], x_plot[j + 1]
            y0, y1 = y_plot[j], y_plot[j + 1]
            color = cmap_(norm((x0 + x1) * 0.5))
            verts = np.array([[x0, 0], [x0, y0], [x1, y1], [x1, 0]])
            poly = PolyCollection([verts], facecolors=[color],
                                  edgecolor='none', alpha=0.6)
            ax.add_collection(poly)

        ax.set_title(f"KDE of {col_label}", fontsize=14, fontweight='bold')
        ax.set_xlabel(col_label, fontsize=16, fontweight='bold')
        ax.set_ylabel("Density", fontsize=16, fontweight='bold')
        ax.set_xlim(df[col].min(), df[col].max())

        # Tick fonts
        for t in ax.get_xticklabels():                                       # ← font
            t.set_fontsize(15)
            t.set_fontweight('bold')
        for t in ax.get_yticklabels():                                       # ← font
            t.set_fontsize(15)
            t.set_fontweight('bold')

        # Y-axis formatting
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        my_formatter = MyScalarFormatter(useMathText=True)
        my_formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(my_formatter)
        ax.yaxis.get_offset_text().set_fontsize(16)
        ax.yaxis.get_offset_text().set_fontweight('bold')

        # Colour-bar
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax)
        cb.set_label("Value Range", fontsize=16, fontweight='bold')
        for tick in cb.ax.get_yticklabels():                                  # ← font
            tick.set_fontsize(15)
            tick.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()
    print(f"[plot_kde_distribution] => {filename}")


def plot_catalyst_size_vs_product(df, filename):
    ensure_dir_for_file(filename)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    products = ['H2', 'CO', 'C1', 'C2+']

    for i, product in enumerate(products):
        ax = axes[i // 2, i % 2]
        needed = ['Particle size (log scale)', 'Active metal', product]
        if all(c in df.columns for c in needed):
            prod_label = short_label_bold(product)
            sns.scatterplot(x='Particle size (log scale)', y=product,
                            hue='Active metal', data=df,
                            ax=ax, alpha=0.7)
            ax.set_title(f'Particle size vs {prod_label} Yield',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel('Particle size (log scale)',
                          fontsize=16, fontweight='bold')
            ax.set_ylabel(f'{prod_label} Yield (%)',
                          fontsize=16, fontweight='bold')

            # Tick fonts
            for t in ax.get_xticklabels():                                   # ← font
                t.set_fontsize(15)
                t.set_fontweight('bold')
            for t in ax.get_yticklabels():                                   # ← font
                t.set_fontsize(15)
                t.set_fontweight('bold')

            handles, labels = ax.get_legend_handles_labels()
            new_labels = [short_label_bold(label) for label in labels]
            ax.legend(handles=handles, labels=new_labels,
                      prop={'size': 12, 'weight': 'bold'})                  # ← font
        else:
            ax.text(0.5, 0.5, f"Cols not found => {short_label_bold(product)}",
                    ha='center', va='center', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()


def plot_potential_vs_product_by_electrolyte(df, filename):
    ensure_dir_for_file(filename)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    products = ['H2', 'CO', 'C1', 'C2+']

    for i, product in enumerate(products):
        ax = axes[i // 2, i % 2]
        needed = ['Potential (V vs. RHE)', 'Electrode support', product]
        if all(c in df.columns for c in needed):
            prod_label = short_label_bold(product)
            sns.scatterplot(x='Potential (V vs. RHE)', y=product,
                            hue='Electrode support', data=df,
                            ax=ax, alpha=0.7)
            ax.set_title(f'Potential vs {prod_label}', fontsize=14,
                         fontweight='bold')
            ax.set_xlabel('Potential (V vs. RHE)', fontsize=16,
                          fontweight='bold')
            ax.set_ylabel(f'{prod_label} Yield (%)', fontsize=16,
                          fontweight='bold')

            # Tick fonts
            for t in ax.get_xticklabels():                                   # ← font
                t.set_fontsize(15)
                t.set_fontweight('bold')
            for t in ax.get_yticklabels():                                   # ← font
                t.set_fontsize(15)
                t.set_fontweight('bold')

            handles, labels = ax.get_legend_handles_labels()
            new_labels = [short_label_bold(label) for label in labels]
            ax.legend(handles=handles, labels=new_labels,
                      prop={'size': 12, 'weight': 'bold'})                  # ← font
        else:
            ax.text(0.5, 0.5, f"Cols not found => {short_label_bold(product)}",
                    ha='center', va='center', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()


def plot_product_distribution_by_catalyst_and_potential(df, filename):
    ensure_dir_for_file(filename)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    products = ['H2', 'CO', 'C1', 'C2+']

    if 'Potential (V vs. RHE)' in df.columns:
        df['Potential_bin'] = pd.cut(df['Potential (V vs. RHE)'], bins=5)
    else:
        df['Potential_bin'] = "Unknown"

    for i, product in enumerate(products):
        ax = axes[i]
        needed = ['Active metal', product, 'Potential_bin']
        if all(c in df.columns for c in needed):
            prod_label = short_label_bold(product)
            sns.boxplot(x='Active metal', y=product, hue='Potential_bin',
                        data=df, ax=ax)
            ax.set_title(f'{prod_label} by Active metal & Potential',
                         fontsize=14, fontweight='bold')
            ax.set_xlabel("Active metal", fontsize=16, fontweight='bold')  # NEW
            ax.set_ylabel(f'{prod_label} Yield (%)', fontsize=16, fontweight='bold')  # NEW
            ax.tick_params(axis='x', rotation=45)

            # Tick fonts
            for t in ax.get_xticklabels():                                   # ← font
                t.set_fontsize(15)
                t.set_fontweight('bold')
            for t in ax.get_yticklabels():                                   # ← font
                t.set_fontsize(15)
                t.set_fontweight('bold')

            new_xticks = [short_label_bold(t.get_text())
                          for t in ax.get_xticklabels()]
            ax.set_xticklabels(new_xticks)
            ax.legend(prop={'size': 12, 'weight': 'bold'})                  # ← font
        else:
            ax.text(0.5, 0.5, f"Cols not found => {short_label_bold(product)}",
                    ha='center', va='center', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()


def plot_product_vs_potential_bin(df, filename):
    ensure_dir_for_file(filename)

    products = ['H2', 'CO', 'C1', 'C2+']
    if 'Potential (V vs. RHE)' not in df.columns:
        print("[WARN] no Potential => skip")
        return

    df['Potential bin custom'] = pd.cut(df['Potential (V vs. RHE)'], bins=5)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, product in enumerate(products):
        ax = axes[i]
        if product not in df.columns:
            ax.text(0.5, 0.5, f"No col => {short_label_bold(product)}",
                    ha='center', va='center', fontsize=16, fontweight='bold')
            continue
        prod_label = short_label_bold(product)
        sns.boxplot(x='Potential bin custom', y=product, data=df, ax=ax)
        ax.set_title(f"{prod_label} vs Potential Bin",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Potential Bin", fontsize=16, fontweight='bold')        # NEW
        ax.set_ylabel(f"{prod_label} Yield (%)",
                      fontsize=16, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Tick fonts → 13 pt
        for t in ax.get_xticklabels():
            t.set_fontsize(13)
            t.set_fontweight('bold')
        for t in ax.get_yticklabels():
            t.set_fontsize(13)
            t.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()


def plot_product_vs_shape(df, filename):
    ensure_dir_for_file(filename)

    products = ['H2', 'CO', 'C1', 'C2+']
    if 'Shape' not in df.columns:
        print("[WARN] no 'Shape' => skip")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, product in enumerate(products):
        ax = axes[i]
        if product not in df.columns:
            ax.text(0.5, 0.5, f"No col => {short_label_bold(product)}",
                    ha='center', va='center', fontsize=16, fontweight='bold')
            continue
        prod_label = short_label_bold(product)
        sns.boxplot(x='Shape', y=product, data=df, ax=ax)
        ax.set_title(f"{prod_label} vs Shape",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("Shape", fontsize=16, fontweight='bold')               # NEW
        ax.set_ylabel(f"{prod_label} Yield (%)",
                      fontsize=16, fontweight='bold')
        ax.tick_params(axis='x', rotation=48)

        # Tick fonts → 13 pt
        for t in ax.get_xticklabels():
            t.set_fontsize(11)
            t.set_fontweight('bold')
        for t in ax.get_yticklabels():
            t.set_fontsize(13)
            t.set_fontweight('bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()


def plot_product_vs_catalyst(df, filename):
    ensure_dir_for_file(filename)

    products = ['H2', 'CO', 'C1', 'C2+']
    if 'Active metal' not in df.columns:
        print("[WARN] no 'Active metal' => skip")
        return

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for i, product in enumerate(products):
        ax = axes[i]
        if product not in df.columns:
            ax.text(0.5, 0.5, f"No col => {short_label_bold(product)}",
                    ha='center', va='center', fontsize=16, fontweight='bold')
            continue
        prod_label = short_label_bold(product)
        sns.boxplot(x='Active metal', y=product, data=df, ax=ax)
        ax.set_title(f"{prod_label} vs Active metal",
                     fontsize=16, fontweight='bold')
        ax.set_xlabel("Active metal", fontsize=14, fontweight='bold')        # NEW
        ax.set_ylabel(f"{prod_label} Yield (%)",
                      fontsize=16, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)

        # Tick fonts → 13 pt
        for t in ax.get_xticklabels():
            t.set_fontsize(13)
            t.set_fontweight('bold')
        for t in ax.get_yticklabels():
            t.set_fontsize(13)
            t.set_fontweight('bold')

        # Re-label x-tick text with bold chemical symbols
        ax.set_xticklabels([short_label_bold(t.get_text())
                            for t in ax.get_xticklabels()])
    plt.tight_layout()
    plt.savefig(filename, dpi=700)
    plt.close()

def plot_potential_vs_product(df, filename):
    ensure_dir_for_file(filename)

    products = ['H2', 'CO', 'C1', 'C2+']
    if 'Potential (V vs. RHE)' not in df.columns:
        print("[WARN] no 'Potential (V vs. RHE)' => skip")
        return

    plt.figure(figsize=(7, 6))
    for product in products:
        if product in df.columns:
            prod_label = short_label_bold(product)
            plt.scatter(df['Potential (V vs. RHE)'], df[product],
                        label=prod_label, alpha=0.7)

    plt.title("Potential vs Products", fontsize=16, fontweight='bold')
    plt.xlabel("Potential (V vs. RHE)", fontsize=16, fontweight='bold')
    plt.ylabel("Yield (%)", fontsize=16, fontweight='bold')
    plt.legend(fontsize=15, prop={'weight': 'bold'})

    ax = plt.gca()

    # ---- NEW: limit major x-ticks to ≤ 6 ----
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=6))

    # Tick-label fonts
    for t in ax.get_xticklabels():
        t.set_fontsize(15)
        t.set_fontweight('bold')
    for t in ax.get_yticklabels():
        t.set_fontsize(15)
        t.set_fontweight('bold')

    plt.savefig(filename, dpi=700)
    plt.close()



# ================ 2D Heatmap 绘制 =====================
def plot_2d_heatmap_from_npy(grid_x, grid_y, heatmap_pred,
                             out_dir,
                             x_label="X-axis",
                             y_label="Y-axis",
                             y_col_names=None,
                             stats_dict=None,
                             colorbar_extend_ratio=0.25):
    """
    heatmap_pred shape=(H,W,out_dim).
    若 stats_dict 和 y_col_names 对应上，则使用 stats_dict[y_name]["min"/"max"] 做颜色范围；
    否则自动从 heatmap_pred 的数据范围 (z_.min, z_.max) 中获取。
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W, out_dim = heatmap_pred.shape

    for odx in range(out_dim):
        z_ = heatmap_pred[:, :, odx]
        # 自动数据范围
        auto_min, auto_max = z_.min(), z_.max()

        # 若可用统计信息，则设置颜色范围
        if (stats_dict is not None) and (y_col_names is not None) and (odx < len(y_col_names)) and (y_col_names[odx] in stats_dict):
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
            vmin_ = max(0, real_min * (1 - colorbar_extend_ratio))
            vmax_ = min(100, real_max * (1 + colorbar_extend_ratio))
        else:
            vmin_ = auto_min
            vmax_ = auto_max

        norm_ = mcolors.Normalize(vmin=vmin_, vmax=vmax_)
        plt.figure(figsize=(6, 5))
        cm_ = plt.pcolormesh(grid_x, grid_y, z_, shading='auto', cmap='viridis', norm=norm_)
        cb_ = plt.colorbar(cm_)
        # 设置 colorbar 标签，调用 short_label_bold 对 y_col_names 中对应标签进行转换
        if y_col_names and odx < len(y_col_names):
            cb_.set_label(short_label_bold(y_col_names[odx]), fontsize=16, fontweight='bold')
        else:
            cb_.set_label(f"Output_{odx}", fontsize=16, fontweight='bold')
        # 遍历 colorbar 刻度，设置字体 15 号且加粗
        for tick in cb_.ax.get_yticklabels():
            tick.set_fontsize(15)
            tick.set_fontweight('bold')
        # 坐标轴标签
        plt.xlabel(x_label, fontsize=16, fontweight='bold')
        plt.ylabel(y_label, fontsize=16, fontweight='bold')
        # 设置刻度文字：fontsize 15 且加粗
        plt.xticks(fontsize=15, fontweight='bold')
        plt.yticks(fontsize=15, fontweight='bold')
        out_jpg = os.path.join(out_dir, f"heatmap_output_{odx+1}.jpg")
        plt.savefig(out_jpg, dpi=700, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 2D Heatmap saved => {out_jpg}")


# ================ 3D Surface 绘制 =====================
def plot_3d_surface_from_heatmap(grid_x, grid_y, heatmap_pred,
                                 out_dir,
                                 x_label="X-axis",
                                 y_label="Y-axis",
                                 y_col_names=None,
                                 stats_dict=None,
                                 colorbar_extend_ratio=0.25,
                                 cmap_name="viridis"):
    """
    绘制三维曲面图 (surface plot)。若 stats_dict 存在，则用统计区间；否则用 heatmap_pred 的数据区间。
    """
    os.makedirs(out_dir, exist_ok=True)
    H, W, out_dim = heatmap_pred.shape

    for odx in range(out_dim):
        Z = heatmap_pred[:, :, odx]
        auto_min, auto_max = Z.min(), Z.max()
        if (stats_dict is not None) and (y_col_names is not None) and (odx < len(y_col_names)) and (y_col_names[odx] in stats_dict):
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
            vmin_ = max(0, real_min * (1 - colorbar_extend_ratio))
            vmax_ = min(100, real_max * (1 + colorbar_extend_ratio))
        else:
            vmin_ = auto_min
            vmax_ = auto_max

        norm_ = mcolors.Normalize(vmin=vmin_, vmax=vmax_)
        cmap_ = plt.get_cmap(cmap_name)
        Z_flat = Z.flatten()
        colors_rgba = cmap_(norm_(Z_flat))
        colors_rgba = colors_rgba.reshape((H, W, 4))
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(grid_x, grid_y, Z,
                               facecolors=colors_rgba,
                               rstride=1, cstride=1,
                               linewidth=0, antialiased=False,
                               shade=False)
        sm = matplotlib.cm.ScalarMappable(norm=norm_, cmap=cmap_)
        sm.set_array([])
        cb = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.1, aspect=15)
        if (y_col_names is not None) and (odx < len(y_col_names)):
            cb.set_label(short_label_bold(y_col_names[odx]), fontsize=16, fontweight='bold')
        else:
            cb.set_label(f"Output_{odx}", fontsize=16, fontweight='bold')
        for tick in cb.ax.get_yticklabels():
            tick.set_fontsize(15)
            tick.set_fontweight('bold')
        ax.set_xlabel(x_label, fontsize=16, fontweight='bold')
        ax.set_ylabel(y_label, fontsize=16, fontweight='bold')
        ax.set_zlabel("Value", fontsize=16, fontweight='bold')
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)
            tick.label.set_fontweight('bold')
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15)
            tick.label.set_fontweight('bold')
        for tick in ax.zaxis.get_major_ticks():
            tick.label.set_fontsize(15)
            tick.label.set_fontweight('bold')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.zaxis.set_major_locator(LinearLocator(5))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.d'))
        ax.grid(False)
        out_jpg = os.path.join(out_dir, f"heatmap_3d_surface_output_{odx+1}.jpg")
        plt.savefig(out_jpg, dpi=700, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        print(f"[INFO] 3D Surface saved => {out_jpg}")


# ================ 混淆矩阵 (Confusion) 绘制 =====================
def plot_confusion_from_npy(confusion_pred,
                            row_labels, col_labels,
                            out_dir,
                            y_col_names=None,
                            stats_dict=None,
                            cell_scale=1/5,
                            colorbar_extend_ratio=0.25,
                            row_axis_name="Row Axis",
                            col_axis_name="Col Axis"):
    """
    confusion_pred shape=(n_rows,n_cols,out_dim)，MIMO 的“混淆矩阵”可视化。
    - 如果 stats_dict 中有该维度对应的统计信息，则用它来确定最小/最大值；
      否则根据该维度的最小值和最大值进行归一化。
    - 最多显示 4 个维度 (odx in [0..3])，并各画一个色条。
    """
    os.makedirs(out_dir, exist_ok=True)
    n_rows, n_cols, out_dim = confusion_pred.shape
    row_labels = [short_label_bold(lbl) for lbl in row_labels]
    col_labels = [short_label_bold(lbl) for lbl in col_labels]
    if y_col_names:
        y_col_names = [short_label_bold(name) for name in y_col_names]
    dim_used = min(4, out_dim)
    cmaps = [plt.get_cmap("Purples"), plt.get_cmap("Blues"),
             plt.get_cmap("Greens"), plt.get_cmap("Oranges")]
    norms = []
    for odx in range(dim_used):
        all_vals_dim = confusion_pred[:, :, odx]
        auto_min, auto_max = all_vals_dim.min(), all_vals_dim.max()
        if (stats_dict is not None) and (y_col_names is not None) and (odx < len(y_col_names)) and (y_col_names[odx] in stats_dict):
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
        else:
            real_min = auto_min
            real_max = auto_max
        confusion_pred[:, :, odx] = normalize_data(all_vals_dim, real_min, real_max)
        norm_ = mcolors.Normalize(vmin=0, vmax=1)
        norms.append(norm_)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Confusion-like MIMO (No numeric)", fontsize=16, fontweight='bold')
    ax.set_aspect("equal", "box")
    fig.subplots_adjust(left=0.08, right=0.92, top=0.88, bottom=0.1)
    for rr in range(n_rows + 1):
        ax.axhline(rr * cell_scale, color='black', linewidth=1)
    for cc in range(n_cols + 1):
        ax.axvline(cc * cell_scale, color='black', linewidth=1)
    for i in range(n_rows):
        for j in range(n_cols):
            vals = confusion_pred[i, j, :]
            BL = (j * cell_scale, i * cell_scale)
            BR = ((j + 1) * cell_scale, i * cell_scale)
            TL = (j * cell_scale, (i + 1) * cell_scale)
            TR = ((j + 1) * cell_scale, (i + 1) * cell_scale)
            Cx = j * cell_scale + cell_scale / 2
            Cy = i * cell_scale + cell_scale / 2
            for odx in range(dim_used):
                val_ = vals[odx]
                norm_ = norms[odx]
                color_ = cmaps[odx](norm_(val_))
                if odx == 0:
                    poly = [TL, (Cx, Cy), TR]
                elif odx == 1:
                    poly = [TR, (Cx, Cy), BR]
                elif odx == 2:
                    poly = [BR, (Cx, Cy), BL]
                else:
                    poly = [BL, (Cx, Cy), TL]
                ax.add_patch(plt.Polygon(poly, facecolor=color_, alpha=0.9))
    ax.set_xlim(0, n_cols * cell_scale)
    ax.set_ylim(0, n_rows * cell_scale)
    ax.invert_yaxis()
    ax.set_xticks([(j + 0.5) * cell_scale for j in range(n_cols)])
    ax.set_yticks([(i + 0.5) * cell_scale for i in range(n_rows)])
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=16, fontweight='bold')
    ax.set_yticklabels(row_labels, fontsize=16, fontweight='bold')
    ax.set_ylabel(row_axis_name, fontsize=16, fontweight='bold')
    ax.set_xlabel(col_axis_name, fontsize=16, fontweight='bold')
    cbar_width = 0.21
    cbar_height = 0.02
    cbar_bottom = 0.93
    cbar_left_start = 0.08
    for odx in range(dim_used):
        sm = plt.cm.ScalarMappable(norm=norms[odx], cmap=cmaps[odx])
        sm.set_array([])
        cax_left = cbar_left_start + odx * cbar_width
        cax = fig.add_axes([cax_left, cbar_bottom, cbar_width, cbar_height])
        cb_ = plt.colorbar(sm, cax=cax, orientation='horizontal', pad=0.2)
        if (y_col_names is not None) and (odx < len(y_col_names)):
            short_lbl = y_col_names[odx]
        else:
            short_lbl = f"Out {odx}"
        cb_.set_label(short_lbl, fontsize=16, fontweight='bold', labelpad=4)
        cb_.set_ticks([])
        cb_.ax.xaxis.set_label_position('bottom')
        cb_.ax.xaxis.set_ticks_position('top')
    outfn = os.path.join(out_dir, "confusion_matrix_mimo.jpg")
    plt.savefig(outfn, dpi=700, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Confusion => {outfn}")


# ================ 3D Bars from Confusion 绘制 =====================
def plot_3d_bars_from_confusion(confusion_pred,
                                row_labels, col_labels,
                                out_dir,
                                y_col_names=None,
                                stats_dict=None,
                                colorbar_extend_ratio=0.25,
                                cmap_name="viridis"):
    """
    绘制三维柱状图(Bar3D)的 “confusion-like” 图。
    - 若 stats_dict 存在且含有对应维度的统计范围，则用其 min/max；
      否则用该维度数据的最小值、最大值。
    - 将 x/y 刻度对准柱体中心，并使得刻度标签居中。
    - 每个维度单独输出一个 3D 柱状图。
    """
    os.makedirs(out_dir, exist_ok=True)
    n_rows, n_cols, out_dim = confusion_pred.shape
    row_labels = [short_label_bold(lbl) for lbl in row_labels]
    col_labels = [short_label_bold(lbl) for lbl in col_labels]
    if y_col_names:
        y_col_names = [short_label_bold(name) for name in y_col_names]

    for odx in range(out_dim):
        all_vals_dim = confusion_pred[:, :, odx]
        auto_min, auto_max = all_vals_dim.min(), all_vals_dim.max()
        if (stats_dict is not None) and (y_col_names is not None) and (odx < len(y_col_names)) and (y_col_names[odx] in stats_dict):
            real_min = stats_dict[y_col_names[odx]]["min"]
            real_max = stats_dict[y_col_names[odx]]["max"]
        else:
            real_min = auto_min
            real_max = auto_max
        Z = normalize_data(all_vals_dim, real_min, real_max)
        norm_ = mcolors.Normalize(vmin=0, vmax=1)
        cmap_ = plt.get_cmap(cmap_name)
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        dx = dy = 0.5
        x_vals, y_vals, z_vals = [], [], []
        dz_vals, facecolors = [], []
        for i in range(n_rows):
            for j in range(n_cols):
                val_ = Z[i, j]
                x_vals.append(j)
                y_vals.append(i)
                z_vals.append(0)
                dz_vals.append(val_)
                facecolors.append(cmap_(norm_(val_)))
        x_vals = np.array(x_vals)
        y_vals = np.array(y_vals)
        z_vals = np.array(z_vals)
        dz_vals = np.array(dz_vals)
        ax.bar3d(x_vals, y_vals, z_vals, dx, dy, dz_vals, color=facecolors, alpha=0.75, shade=True)
        ax.grid(False)
        # 设置x、y轴刻度居中
        ax.set_xticks(np.arange(n_cols) + dx / 2)
        ax.set_yticks(np.arange(n_rows) + dy / 2)
        ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=15)
        ax.set_yticklabels(row_labels, rotation=-15, ha='left', va='center', fontsize=15)
        # bold
        # ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=16, fontweight='bold')
        # ax.set_yticklabels(row_labels, rotation=-15, ha='left', va='center', fontsize=16, fontweight='bold')
        # 设置z轴刻度：遍历所有z轴主刻度，设置 fontsize=16、加粗
        for tick in ax.zaxis.get_major_ticks():
            tick.label.set_fontsize(15)
            tick.label.set_fontweight('bold')
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')

        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

        # 坐标轴标签（x、y、z）
        ax.set_xlabel("", fontsize=16, fontweight='bold')
        ax.set_ylabel("", fontsize=16, fontweight='bold')
        ax.set_zlabel("Value", fontsize=16, fontweight='bold')

        # 设置colorbar
        sm = plt.cm.ScalarMappable(norm=norm_, cmap=cmap_)
        sm.set_array([])
        cb_ = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.1, aspect=15)
        if y_col_names and odx < len(y_col_names):
            var_name = y_col_names[odx]
            ax.set_title(f"3D Bars Confusion - {var_name}", fontsize=18, fontweight='bold')
            # 将 colorbar 标签字体设为 fontsize=15 且加粗
            cb_.set_label(var_name, fontsize=15, fontweight='bold')
        else:
            var_name = f"Output_{odx}"
            ax.set_title(f"3D Bars Confusion - out {odx}", fontsize=18, fontweight='bold')
            cb_.set_label(var_name, fontsize=15, fontweight='bold')
        # 遍历colorbar刻度，设置 fontsize=15 且加粗
        for tick in cb_.ax.get_yticklabels():
            tick.set_fontsize(15)
            tick.set_fontweight('bold')
        out_jpg = os.path.join(out_dir, f"3d_bars_confusion_output_{odx+1}.jpg")
        plt.savefig(out_jpg, dpi=700, bbox_inches='tight')
        plt.close()
        print(f"[INFO] 3D Bars Confusion saved => {out_jpg}")


# ---------------------------------------------
#  One‑hot merge utility
# ---------------------------------------------
def merge_onehot_shap(shap_data, onehot_groups, case_map=None):
    """
    将同一类别的 one-hot dummy 列合并成单列，并返回新的 shap_data dict。
    """
    import copy
    shap_values = shap_data["shap_values"]
    X_full      = shap_data["X_full"]
    col_names   = shap_data["x_col_names"]

    # ------------- 1) 统一成 list -------------
    shap_is_list = isinstance(shap_values, list)
    shap_values = shap_values if shap_is_list else [shap_values]

    # ------------- 2) 建立“保留列”索引 -------------
    flat_oh = {i for g in onehot_groups for i in g}
    keep_idx = [i for i in range(len(col_names)) if i not in flat_oh]

    # ------------- 3) 构造新列名 -------------
    new_col_names = [col_names[i] for i in keep_idx]
    for g in onehot_groups:
        pref = col_names[g[0]].split('_')[0]
        if case_map is not None:
            pref = case_map.get(pref.lower(), pref)
        new_col_names.append(pref)

    # ------------- 4) 合并 SHAP & X_full -------------
    new_shap_list = []
    for sv in shap_values:
        parts = [sv[:, keep_idx]]
        for g in onehot_groups:
            parts.append(sv[:, g].sum(axis=1, keepdims=True))
        new_shap_list.append(np.hstack(parts))

    if X_full is not None:
        parts_d = [X_full[:, keep_idx]]
        for g in onehot_groups:
            chosen = (X_full[:, g].argmax(axis=1)).reshape(-1, 1)
            parts_d.append(chosen)
        new_data = np.hstack(parts_d)
    else:
        new_data = None

    # ------------- 5) 封装并返回 -------------
    new_sd = copy.deepcopy(shap_data)
    new_sd["shap_values"] = new_shap_list if shap_is_list else new_shap_list[0]
    new_sd["X_full"]      = new_data
    new_sd["x_col_names"] = new_col_names
    return new_sd

# ---------------------------------------------
#  Plot – SHAP importance
# ---------------------------------------------

def plot_shap_importance(
    shap_data,
    output_path,
    top_n_features: int = 15,
    plot_width: int = 12,
    plot_height: int = 8,
):
    ensure_dir_for_file(os.path.join(output_path, "dummy.txt"))

    shap_values   = shap_data["shap_values"]
    x_col_names   = [short_label_bold(c) for c in shap_data["x_col_names"]]
    multi_output  = isinstance(shap_values, list)
    if not multi_output:
        shap_values = [shap_values]

    for idx, sv in enumerate(shap_values):
        # ---- importance ranking ----
        mean_abs   = np.mean(np.abs(sv), axis=0)
        order      = np.argsort(mean_abs)[::-1][:top_n_features]
        imps       = mean_abs[order]
        feats      = [x_col_names[i] for i in order]
        threshold  = imps.mean()
        colors     = ["blue" if v > threshold else "red" for v in imps]

        # ---- fig ----
        fig, ax = plt.subplots(figsize=(plot_width, plot_height))
        ax.barh(range(len(imps)), imps, color=colors, align="center")
        ax.set_yticks(range(len(imps)))
        ax.set_yticklabels(feats, fontsize=15, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlabel("Mean(|SHAP|)", fontsize=16, fontweight="bold")

        # ---- title (pretty) ----
        pretty_y   = short_label_bold(shap_data["y_col_names"][idx]) if idx < len(shap_data["y_col_names"]) else f"Output{idx}"
        ax.set_title(f"Mean |SHAP| (Top-{top_n_features}) – {pretty_y}", fontsize=14, fontweight="bold")

        # ---- threshold marker ----
        ax.axvspan(0, threshold, facecolor="lightgray", alpha=0.5)
        ax.axvline(threshold, color="gray", linestyle="dashed", linewidth=2)

        legend_e = [
            Patch(facecolor="blue", label="Above Mean"),
            Patch(facecolor="red", label="Below/Equal Mean")
        ]
        ax.legend(
            handles=legend_e,
            loc="lower right",
            prop={'size': 12, 'weight': 'bold'}
        )

        plt.tight_layout()

        # ---- filename (RAW label) ----
        raw_label = shap_data["y_col_names"][idx] if idx < len(shap_data["y_col_names"]) else f"Output{idx}"
        fname     = f"shap_importance_{safe_filename(raw_label)}.jpg"
        plt.savefig(os.path.join(output_path, fname), dpi=700)
        plt.close()
        print(f"[INFO] SHAP importance saved → {fname}")

# ---------------------------------------------
#  Plot – SHAP beeswarm
# ---------------------------------------------

def plot_shap_beeswarm(
    shap_data,
    output_path,
    top_n_features: int = 15,
    plot_width: int = 12,
    plot_height: int = 8,
):
    ensure_dir_for_file(os.path.join(output_path, "dummy.txt"))

    shap_values   = shap_data["shap_values"]
    X_full        = shap_data["X_full"]
    x_col_names   = [short_label_bold(c) for c in shap_data["x_col_names"]]
    multi_output  = isinstance(shap_values, list)
    if not multi_output:
        shap_values = [shap_values]

    for idx, sv in enumerate(shap_values):
        # ---- figure ----
        shap.summary_plot(
            sv,
            features      = X_full,
            feature_names = x_col_names,
            show          = False,
            max_display   = top_n_features,
            plot_size     = (plot_width, plot_height),
        )

        ax = plt.gca()
        for tick in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            tick.set_fontsize(15)
            tick.set_fontweight("bold")
        ax.set_xlabel(ax.get_xlabel(), fontsize=16, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_visible(True)
        plt.tight_layout()

        # ---- filename ----
        raw_label = shap_data["y_col_names"][idx] if idx < len(shap_data["y_col_names"]) else f"Output{idx}"
        fname     = f"shap_beeswarm_{safe_filename(raw_label)}.jpg"
        plt.savefig(os.path.join(output_path, fname), dpi=700)
        plt.close()
        print(f"[INFO] SHAP beeswarm saved → {fname}")
