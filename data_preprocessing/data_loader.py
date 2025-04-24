"""
data_preprocessing/data_loader.py

1) 读取 CSV 前14列: 前10 => X, 后4 => Y
2) 对 X 做 one-hot (pandas.get_dummies)
3) 为分类列统一转小写；然后做完 get_dummies 后恢复列名大小写
4) 返回:
   X, Y, numeric_cols_idx, x_col_names, y_col_names,
   observed_combos, onehot_groups, oh_index_map
   （其中 observed_combos、onehot_groups、oh_index_map 用于后续 inference）
"""

import pandas as pd
import numpy as np
from collections import defaultdict

def _to_lower_dict(lst):
    """记录原始列名与其小写形式的映射: {lower_name: original_name}"""
    return {col.lower(): col for col in lst}

def _restore_original_case(mapped_dict, lower_names):
    """根据 mapped_dict(小写->原始)，将 lower_names 里的每个列名恢复为原始大小写"""
    restored = []
    for name in lower_names:
        # 先把当前 name 转小写，然后在字典中找
        # 如果找不到，就用 name 原样
        orig = mapped_dict.get(name.lower(), name)
        restored.append(orig)
    return restored

def load_dataset(csv_path, drop_nan=True):
    df_raw = pd.read_csv(csv_path)
    df = df_raw.iloc[:, :14].copy()

    if drop_nan:
        df.dropna(subset=df.columns, how='any', inplace=True)

    if df.shape[1] < 14:
        raise ValueError("After dropping/cleaning, not enough columns (need >=14).")

    # 前10列 => X, 后4列 => Y
    X_df = df.iloc[:, :10].copy()
    Y_df = df.iloc[:, 10:14].copy()
    y_col_names = list(Y_df.columns)
    Y = Y_df.values

    # 可选：检查 Y 是否有 <0 或 >100
    for i, cname in enumerate(y_col_names):
        below0 = (Y[:, i] < 0).sum()
        above100 = (Y[:, i] > 100).sum()
        if below0 > 0 or above100 > 0:
            print(f"[WARN] Y-col '{cname}' has {below0} <0, {above100} >100")

    # ---- 记录原始 X 列名(仅10列)
    original_x_col_names = list(X_df.columns)

    # ---- 区分数值列 & 分类列
    categorical_cols = X_df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols_original = [col for col in X_df.columns if col not in categorical_cols]

    # ---- 对分类列统一转小写
    #      如果某列是object类型，就把其转为小写
    for col in categorical_cols:
        X_df[col] = X_df[col].apply(lambda x: x.lower() if isinstance(x, str) else x)

    # ---- 做 One-Hot
    X_encoded = pd.get_dummies(X_df, columns=categorical_cols)
    all_cols = X_encoded.columns.tolist()
    X = X_encoded.values

    # ---- numeric_cols_idx
    numeric_cols_idx = []
    for i, colname in enumerate(all_cols):
        if colname in numeric_cols_original:
            numeric_cols_idx.append(i)

    x_col_names_lower = list(X_encoded.columns)  # 此时的列名是小写 + 下划线
    # x_col_names_lower 形如 ["shape_nanowire", "shape_nanoparticle", ...]

    # ---- 分析 one-hot 分组
    onehot_groups, oh_index_map = _extract_onehot_info(x_col_names_lower, numeric_cols_idx)

    # ---- 统计在训练集出现过的 one-hot 组合
    observed_combos = _get_observed_onehot_combos(X, onehot_groups)

    # ---- 恢复列名大小写
    #      首先构造一个字典: { lower_name: original_name } 用于映射
    x_col_map = _to_lower_dict(original_x_col_names)
    # 由于 one-hot 生成的新列名 e.g. "shape_nanowire" 并不在 original_x_col_names 里，
    # 这部分会恢复不到「原始」(因为是新增列)。但是 numeric 列可以恢复到原大小写。
    # 对于 one-hot 生成的新列，只能保持它的小写+下划线了。
    # 如果你确实要恢复"大写/小写"的分类值，这里要更复杂的映射; 简单处理就维持 one-hot 列名即可。
    # (示例： "Shape_nanowire" => "shape_nanowire" -> 你或许并不一定想恢复?)
    # 这里按需求，只恢复 numeric 的列名大小写即可:
    x_col_names_final = _restore_original_case(x_col_map, x_col_names_lower)

    return X, Y, numeric_cols_idx, x_col_names_final, y_col_names, observed_combos, onehot_groups, oh_index_map


def _extract_onehot_info(x_col_names, numeric_cols_idx):
    """
    构造:
      onehot_groups: [[colid7, colid8], [colid9, colid10,...], ...]
      oh_index_map: flatten 后 => 全局列号
    """
    prefix_map = {}
    for i, cname in enumerate(x_col_names):
        if i not in numeric_cols_idx:
            prefix = cname.split('_')[0]  # 以'_'前缀区分
            prefix_map.setdefault(prefix, []).append(i)

    onehot_groups = []
    for pref, idxs in prefix_map.items():
        idxs_sorted = sorted(idxs)
        onehot_groups.append(idxs_sorted)

    # flatten
    oh_index_map = []
    for group in onehot_groups:
        for colid in group:
            oh_index_map.append(colid)
    return onehot_groups, oh_index_map


def _get_observed_onehot_combos(X, onehot_groups):
    """
    从 X 中统计实际出现过的 one-hot 组合 (flatten后) => freq
    返回: [ (tuple0_1, freq), ... ]  其中 tuple0_1 形如 (0,1,0,1,...)
    """
    combo_count = {}
    for row in X:
        flattened = []
        for g in onehot_groups:
            for colid in g:
                flattened.append(int(row[colid]))
        key = tuple(flattened)
        combo_count[key] = combo_count.get(key, 0) + 1

    combos_sorted = sorted(combo_count.items(), key=lambda x: -x[1])
    return combos_sorted


def load_raw_data_for_correlation(csv_path, drop_nan=True):
    """
    如果要做混合变量相关性分析，而不想做 One-Hot，可用这个读取前14列。
    """
    df_raw = pd.read_csv(csv_path)
    df = df_raw.iloc[:, :14].copy()
    if drop_nan:
        df.dropna(subset=df.columns, how='any', inplace=True)
    return df


def extract_data_statistics(X, x_col_names, numeric_cols_idx,
                            Y=None, y_col_names=None):
    """
    提取并返回一个 dict:
      {
        "continuous_cols": {
            <colname>: { "min":..., "max":..., "mean":... }, ...
        },
        "onehot_groups": [...]
      }
    """
    stats = {
        "continuous_cols": {},
        "onehot_groups": []
    }

    # 1) X 的数值列 min/max/mean
    for idx in numeric_cols_idx:
        cname = x_col_names[idx]
        col_data = X[:, idx]
        stats["continuous_cols"][cname] = {
            "min": float(col_data.min()),
            "max": float(col_data.max()),
            "mean": float(col_data.mean())
        }

    # 2) 若有 Y
    if (Y is not None) and (y_col_names is not None):
        for i, cname in enumerate(y_col_names):
            col_data = Y[:, i]
            stats["continuous_cols"][cname] = {
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "mean": float(col_data.mean())
            }

    # 3) 找到 one-hot groups
    prefix_map = {}
    for i, cname in enumerate(x_col_names):
        # 若 i 不在 numeric_cols_idx => 说明是 one-hot
        if i not in numeric_cols_idx:
            prefix = cname.split('_')[0]
            prefix_map.setdefault(prefix, []).append(i)

    for pref, idxs in prefix_map.items():
        idxs_sorted = sorted(idxs)
        stats["onehot_groups"].append(idxs_sorted)

    return stats
