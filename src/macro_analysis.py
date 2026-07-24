"""全校宏观分析模块。

分析维度：
- score_distribution(): 总分分布（直方图 + 描述统计 + 正态性检验）
- subject_comparison(): 各科成绩对比（箱线图 + 统计表）
- excellence_rates(): 优秀率/及格率/低分率
- subject_correlation(): 学科相关性矩阵
- gender_comparison(): 男女生成绩对比 + t 检验
- score_band_analysis(): 分数段金字塔
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
import plotly.graph_objects as go

from .utils import (
    SUBJECT_LABELS,
    ALL_SUBJECTS,
    SUBJECT_GROUPS,
    grade_level,
    grade_label,
    format_percent,
)


# ============================================================
# 1. 总分分布分析
# ============================================================

def score_distribution(
    df: pd.DataFrame,
    max_scores: Dict[str, float],
) -> Dict[str, Any]:
    """总分分布分析。

    Returns:
        {
            "stats": {mean, median, std, skewness, kurtosis, min, max},
            "normality": {shapiro_stat, shapiro_pvalue, is_normal},
            "score_bands": DataFrame (分数段统计),
        }
    """
    total = df["total_score"].dropna()
    if len(total) == 0:
        return {"stats": None, "normality": None, "score_bands": None}

    # 描述统计
    stats = {
        "mean": round(float(total.mean()), 1),
        "median": round(float(total.median()), 1),
        "std": round(float(total.std()), 1),
        "skewness": round(float(total.skew()), 2),
        "kurtosis": round(float(total.kurtosis()), 2),
        "min": round(float(total.min()), 1),
        "max": round(float(total.max()), 1),
        "q25": round(float(total.quantile(0.25)), 1),
        "q75": round(float(total.quantile(0.75)), 1),
    }

    # 正态性检验 (Shapiro-Wilk)
    try:
        if len(total) >= 3 and len(total) <= 5000:
            shapiro_stat, shapiro_p = scipy_stats.shapiro(total)
            normality = {
                "shapiro_stat": round(float(shapiro_stat), 4),
                "shapiro_pvalue": round(float(shapiro_p), 4),
                "is_normal": shapiro_p > 0.05,
            }
        else:
            normality = None
    except Exception:
        normality = None

    # 分数段统计（每 10 分一段）
    total_max = total.max()
    total_min = total.min()
    band_width = 10
    bands = list(range(int(total_min // band_width) * band_width, int(total_max + band_width), band_width))
    band_labels = [f"{b}-{b + band_width - 1}" for b in bands[:-1]]
    band_counts, _ = np.histogram(total, bins=bands)

    score_bands = pd.DataFrame({
        "分数段": band_labels,
        "人数": band_counts,
        "占比": [round(c / len(total), 4) for c in band_counts],
    })

    return {
        "stats": stats,
        "normality": normality,
        "score_bands": score_bands,
    }


# ============================================================
# 2. 各科成绩对比
# ============================================================

def subject_comparison(
    df: pd.DataFrame,
    max_scores: Dict[str, float],
) -> Dict[str, Any]:
    """各科成绩对比分析。

    Returns:
        {
            "subject_stats": DataFrame (均分/中位数/标准差/最高/最低/CV),
            "subject_scores": {subject: [scores]},  # 用于箱线图
        }
    """
    rows = []
    subject_scores = {}

    for subject in ALL_SUBJECTS:
        if subject not in df.columns:
            continue

        valid = df[subject].dropna()
        if len(valid) == 0:
            continue

        max_val = max_scores.get(subject, 100)
        mean_val = float(valid.mean())
        std_val = float(valid.std())
        cv = round(std_val / mean_val, 3) if mean_val > 0 else 0

        rows.append({
            "学科": SUBJECT_LABELS.get(subject, subject),
            "满分": max_val,
            "均分": round(mean_val, 1),
            "中位数": round(float(valid.median()), 1),
            "标准差": round(std_val, 1),
            "最高分": round(float(valid.max()), 1),
            "最低分": round(float(valid.min()), 1),
            "得分率": round(mean_val / max_val, 3),
            "变异系数(CV)": cv,
        })
        subject_scores[subject] = valid.tolist()

    return {
        "subject_stats": pd.DataFrame(rows),
        "subject_scores": subject_scores,
    }


# ============================================================
# 3. 优秀率 / 及格率 / 低分率
# ============================================================

def excellence_rates(
    df: pd.DataFrame,
    max_scores: Dict[str, float],
) -> pd.DataFrame:
    """各科 ABC 等级分布。

    Returns:
        DataFrame: 行=学科, 列=等级人数/比例
    """
    rows = []

    for subject in ALL_SUBJECTS:
        if subject not in df.columns:
            continue

        valid = df[subject].dropna()
        if len(valid) == 0:
            continue

        max_val = max_scores.get(subject, 100)
        rate = valid / max_val

        counts = {"A": 0, "B": 0, "C": 0, "D": 0}
        for r in rate:
            level = grade_level(r)
            counts[level] += 1

        total = len(valid)
        rows.append({
            "学科": SUBJECT_LABELS.get(subject, subject),
            "A优秀": counts["A"],
            "A比例": round(counts["A"] / total, 3),
            "B良好": counts["B"],
            "B比例": round(counts["B"] / total, 3),
            "C及格": counts["C"],
            "C比例": round(counts["C"] / total, 3),
            "D不及格": counts["D"],
            "D比例": round(counts["D"] / total, 3),
        })

    return pd.DataFrame(rows)


# ============================================================
# 4. 学科相关性矩阵
# ============================================================

def subject_correlation(df: pd.DataFrame) -> Dict[str, Any]:
    """学科相关性分析。

    Returns:
        {
            "corr_matrix": DataFrame (Pearson 相关系数矩阵),
            "top_correlations": [(学科1, 学科2, r_value), ...] 前 5 强相关对,
        }
    """
    # 只取成绩列
    score_cols = [s for s in ALL_SUBJECTS if s in df.columns]
    if len(score_cols) < 2:
        return {"corr_matrix": None, "top_correlations": []}

    corr = df[score_cols].corr()

    # 用中文标签重命名
    corr.index = [SUBJECT_LABELS.get(c, c) for c in corr.index]
    corr.columns = [SUBJECT_LABELS.get(c, c) for c in corr.columns]

    # 找出最强相关对（排除自相关 1.0）
    pairs = []
    for i, col1 in enumerate(score_cols):
        for j, col2 in enumerate(score_cols):
            if i < j:
                r = corr.iloc[i, j]
                pairs.append((SUBJECT_LABELS.get(col1, col1), SUBJECT_LABELS.get(col2, col2), round(float(r), 3)))

    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    return {
        "corr_matrix": corr,
        "top_correlations": pairs[:5],
    }


# ============================================================
# 5. 男女生成绩对比
# ============================================================

def gender_comparison(df: pd.DataFrame) -> Dict[str, Any]:
    """男女生成绩对比 + 独立样本 t 检验。

    Returns:
        {
            "gender_stats": DataFrame,
            "ttest_results": [{"subject":, "male_mean":, "female_mean":, "t_stat":, "p_value":, "significant":}, ...],
        }
    """
    if "gender" not in df.columns:
        return {"gender_stats": None, "ttest_results": []}

    male_df = df[df["gender"] == "男"]
    female_df = df[df["gender"] == "女"]

    ttest_results = []
    rows = []

    subjects_to_check = [s for s in ALL_SUBJECTS if s in df.columns]
    if "total_score" in df.columns:
        subjects_to_check = ["total_score"] + subjects_to_check

    for subject in subjects_to_check:
        male_scores = male_df[subject].dropna()
        female_scores = female_df[subject].dropna()

        male_mean = float(male_scores.mean()) if len(male_scores) > 0 else 0
        female_mean = float(female_scores.mean()) if len(female_scores) > 0 else 0

        rows.append({
            "学科": SUBJECT_LABELS.get(subject, subject),
            "男生均分": round(male_mean, 1),
            "女生均分": round(female_mean, 1),
            "差值(男-女)": round(male_mean - female_mean, 1),
        })

        # t 检验 (独立样本)
        if len(male_scores) >= 3 and len(female_scores) >= 3:
            try:
                t_stat, p_value = scipy_stats.ttest_ind(male_scores, female_scores)
                ttest_results.append({
                    "subject": subject,
                    "label": SUBJECT_LABELS.get(subject, subject),
                    "male_mean": round(male_mean, 1),
                    "female_mean": round(female_mean, 1),
                    "t_stat": round(float(t_stat), 3),
                    "p_value": round(float(p_value), 4),
                    "significant": p_value < 0.05,
                    "highly_significant": p_value < 0.01,
                })
            except Exception:
                pass

    return {
        "gender_stats": pd.DataFrame(rows),
        "ttest_results": ttest_results,
    }


# ============================================================
# 6. 分数段分析（金字塔）
# ============================================================

def score_band_analysis(
    df: pd.DataFrame,
    max_scores: Dict[str, float],
) -> pd.DataFrame:
    """分数段金字塔分析。

    按得分率分层：
    - 顶尖层: >= 90%
    - 优秀层: >= 80%
    - 良好层: >= 70%
    - 达标层: >= 60%
    - 基础层: < 60%
    """
    if "total_score" not in df.columns:
        return pd.DataFrame()

    total_max = sum(max_scores.values())
    tmp = df.copy()
    tmp["score_rate"] = tmp["total_score"] / total_max

    bands = {
        "顶尖层 (≥90%)": tmp["score_rate"] >= 0.90,
        "优秀层 (80-89%)": (tmp["score_rate"] >= 0.80) & (tmp["score_rate"] < 0.90),
        "良好层 (70-79%)": (tmp["score_rate"] >= 0.70) & (tmp["score_rate"] < 0.80),
        "达标层 (60-69%)": (tmp["score_rate"] >= 0.60) & (tmp["score_rate"] < 0.70),
        "基础层 (<60%)": tmp["score_rate"] < 0.60,
    }

    rows = []
    for band_name, mask in bands.items():
        band_df = tmp[mask]
        n = len(band_df)
        if n > 0:
            avg_total = float(band_df["total_score"].mean())
        else:
            avg_total = 0

        rows.append({
            "分数层": band_name,
            "人数": n,
            "占比": round(n / len(tmp), 4) if len(tmp) > 0 else 0,
            "总分均分": round(avg_total, 1),
        })

    return pd.DataFrame(rows)
