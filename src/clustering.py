"""学生聚类分群模块。

基于 K-Means 对学生成绩进行聚类分析。

功能：
- prepare_features(): 特征工程（标准化）
- find_optimal_k(): 肘部法则 + 轮廓系数选择最优 K
- run_kmeans(): 执行 K-Means 聚类
- pca_transform(): PCA 降维到 2D
- cluster_profiles(): 聚类画像描述
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .utils import SUBJECT_LABELS, ALL_SUBJECTS, SUBJECT_GROUPS

# 预设聚类命名参考
CLUSTER_NAMING_HINTS = [
    "全能型",
    "理科优势型",
    "文科优势型",
    "基础薄弱型",
    "中等偏科型",
]


def prepare_features(
    df: pd.DataFrame,
    max_scores: Dict[str, float],
) -> Tuple[np.ndarray, List[str], StandardScaler]:
    """准备聚类特征。

    选择所有成绩列，使用 StandardScaler 标准化。

    Args:
        df: 清洗后的 DataFrame
        max_scores: 各科满分

    Returns:
        (X_scaled, feature_names, scaler)
    """
    # 可选特征：各科成绩列
    feature_cols = [s for s in ALL_SUBJECTS if s in df.columns]

    # 提取特征矩阵
    X = df[feature_cols].copy()

    # 替换 NaN 为列均值（聚类需要完整数据）
    for col in feature_cols:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].mean())

    # 标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, feature_cols, scaler


def find_optimal_k(
    X_scaled: np.ndarray,
    k_range: range = None,
) -> Dict[str, Any]:
    """寻找最优 K 值。

    双指标：
    1. Elbow Method (WCSS)
    2. Silhouette Score (轮廓系数)

    Args:
        X_scaled: 标准化后的特征矩阵
        k_range: K 值范围，默认 range(3, 9)

    Returns:
        {
            "k_values": [3, 4, 5, 6, 7, 8],
            "wcss": [1234.5, 987.3, ...],
            "silhouette": [0.35, 0.41, ...],
            "optimal_k": 4,
        }
    """
    if k_range is None:
        k_range = range(3, min(9, len(X_scaled)))

    k_values = []
    wcss_list = []
    silhouette_list = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        k_values.append(k)
        wcss_list.append(round(float(kmeans.inertia_), 2))
        if len(set(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)
            silhouette_list.append(round(float(sil), 4))
        else:
            silhouette_list.append(0.0)

    # 选最优 K（轮廓系数最大）
    if silhouette_list:
        max_sil_idx = silhouette_list.index(max(silhouette_list))
        optimal_k = k_values[max_sil_idx]
    else:
        optimal_k = 4  # 默认

    return {
        "k_values": k_values,
        "wcss": wcss_list,
        "silhouette": silhouette_list,
        "optimal_k": optimal_k,
    }


def run_kmeans(
    X_scaled: np.ndarray,
    n_clusters: int = 4,
) -> Tuple[np.ndarray, KMeans]:
    """执行 K-Means 聚类。

    Args:
        X_scaled: 标准化特征矩阵
        n_clusters: 聚类数

    Returns:
        (labels, kmeans_model)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return labels, kmeans


def pca_transform(X_scaled: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, PCA]:
    """PCA 降维到 2D 用于可视化。

    Returns:
        (X_pca, pca_model)
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


def cluster_profiles(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: List[str],
    max_scores: Dict[str, float],
    scaler: StandardScaler,
) -> List[Dict[str, Any]]:
    """生成各聚类的画像描述。

    Returns:
        每个聚类的信息列表:
        [{
            "cluster_id": 0,
            "name": "全能型",
            "size": 125,
            "percent": 15.2,
            "subject_means": {"语文": 110.5, ...},
            "subject_z_means": {"语文": 0.85, ...},
            "total_mean": 720.5,
        }, ...]
    """
    # 计算每个聚类的各科均分
    unique_labels = sorted(set(labels))
    profiles = []

    for label in unique_labels:
        mask = labels == label
        cluster_df = df.iloc[mask]
        n = len(cluster_df)

        subject_means = {}
        for col in feature_cols:
            if col in cluster_df.columns:
                subject_means[SUBJECT_LABELS.get(col, col)] = round(float(cluster_df[col].mean()), 1)

        # 总分均分
        total_mean = None
        if "total_score" in cluster_df.columns:
            total_mean = round(float(cluster_df["total_score"].mean()), 1)

        profiles.append({
            "cluster_id": int(label),
            "name": _auto_name_cluster(cluster_df, feature_cols, max_scores),
            "size": n,
            "percent": round(n / len(df) * 100, 1),
            "subject_means": subject_means,
            "total_mean": total_mean,
        })

    # 按总分排序
    profiles.sort(key=lambda x: x.get("total_mean") or 0, reverse=True)

    return profiles


def _auto_name_cluster(
    cluster_df: pd.DataFrame,
    feature_cols: List[str],
    max_scores: Dict[str, float],
) -> str:
    """基于聚类与全局对比自动命名。

    用得分率做相对命名：
    - 总分率 ≥ 0.88 → "A 顶尖层"
    - 总分率 ≥ 0.82 → "B 优秀层" + 文理标注（如文理偏差 > 0.03）
    - 总分率 ≥ 0.70 → "C 良好层"
    - 总分率 ≥ 0.55 → "D 达标层"
    - 总分率 < 0.55 → "基础薄弱"
    """
    arts_subjects = SUBJECT_GROUPS["文科"]
    science_subjects = SUBJECT_GROUPS["理科"]

    arts_cols = [s for s in arts_subjects if s in cluster_df.columns and s in feature_cols]
    science_cols = [s for s in science_subjects if s in cluster_df.columns and s in feature_cols]

    arts_rate = 0
    for c in arts_cols:
        mx = max_scores.get(c, 100)
        if mx > 0:
            arts_rate += cluster_df[c].mean() / mx
    arts_rate = arts_rate / len(arts_cols) if arts_cols else 0

    science_rate = 0
    for c in science_cols:
        mx = max_scores.get(c, 100)
        if mx > 0:
            science_rate += cluster_df[c].mean() / mx
    science_rate = science_rate / len(science_cols) if science_cols else 0

    total_full = sum(max_scores.values())
    total_rate = cluster_df["total_score"].mean() / total_full if "total_score" in cluster_df.columns and total_full > 0 else 0

    bias = arts_rate - science_rate
    if bias > 0.03:
        tag = "偏文"
    elif bias < -0.03:
        tag = "偏理"
    else:
        tag = ""

    # 按总分率分档
    if total_rate >= 0.90:
        level = "A 顶尖"
    elif total_rate >= 0.80:
        level = "B 优秀"
    elif total_rate >= 0.65:
        level = "C 良好"
    elif total_rate >= 0.50:
        level = "D 达标"
    else:
        return "基础薄弱"

    # 对 B/C/D 档，加最强/最弱学科标签区分
    if level != "A 顶尖":
        feature_cols_in_df = [c for c in feature_cols if c in cluster_df.columns]
        rates = {}
        for c in feature_cols_in_df:
            mx = max_scores.get(c, 100)
            if mx > 0:
                rates[SUBJECT_LABELS.get(c, c)] = cluster_df[c].mean() / mx
        if rates:
            best = max(rates, key=rates.get)
            worst = min(rates, key=rates.get)
            if best != worst:
                tag = f"·{best}强{worst}弱"
            else:
                tag = ""
        name = f"{level}{tag}"
    else:
        name = f"{level}{'·' + tag if tag else ''}"

    return name


def get_cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    profiles: List[Dict[str, Any]],
) -> pd.DataFrame:
    """聚类结果汇总表。

    Returns:
        DataFrame: 每行一个聚类
    """
    rows = []
    for p in profiles:
        row = {
            "聚类": f"聚类 {p['cluster_id']}: {p['name']}",
            "人数": p["size"],
            "占比": f"{p['percent']}%",
            "总分均分": p.get("total_mean", "-"),
        }
        rows.append(row)

    return pd.DataFrame(rows)
