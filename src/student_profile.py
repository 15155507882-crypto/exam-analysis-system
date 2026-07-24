"""个人画像模块。

功能：
- radar_data(): 雷达图数据准备（百分制得分率）
- imbalance_index(): 偏科指数计算
- strength_weakness(): 优势/薄弱学科识别
- percentile_rank(): 全校百分位定位
- arts_science_bias(): 文科/理科倾向度
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .utils import (
    SUBJECT_LABELS,
    ALL_SUBJECTS,
    SUBJECT_GROUPS,
    normalize_scores,
)


def radar_data(
    student_row: pd.Series,
    df_all: pd.DataFrame,
    max_scores: Dict[str, float],
) -> Dict[str, Any]:
    """准备雷达图数据。

    返回学生在各科的百分制得分率 + 全校均分得分率作为参考。

    Args:
        student_row: 该学生的数据行
        df_all: 全校 DataFrame
        max_scores: 各科满分

    Returns:
        {
            "categories": ["语文", "数学", ...],
            "student_rates": [0.85, 0.92, ...],  # 学生各科得分率
            "avg_rates": [0.75, 0.78, ...],      # 全校均分得分率
            "max_rates": [0.95, 1.0, ...],       # 全校最高得分率
        }
    """
    categories = []
    student_rates = []
    avg_rates = []
    max_rates = []

    for subject in ALL_SUBJECTS:
        if subject not in df_all.columns:
            continue
        max_val = max_scores.get(subject, 100)
        if max_val <= 0:
            continue

        label = SUBJECT_LABELS.get(subject, subject)

        # 学生得分率（缺失的学科跳过，不画在雷达图上）
        student_score = student_row.get(subject)
        if pd.isna(student_score):
            continue  # 缺考的不参与雷达图

        student_rates.append(round(float(student_score) / max_val * 100, 1))

        # 全校均分得分率
        avg_score = df_all[subject].mean()
        avg_rates.append(round(float(avg_score) / max_val * 100, 1))

        # 全校最高得分率
        max_score = df_all[subject].max()
        max_rates.append(round(float(max_score) / max_val * 100, 1))

        categories.append(label)

    return {
        "categories": categories,
        "student_rates": student_rates,
        "avg_rates": avg_rates,
        "max_rates": max_rates,
    }


def radar_data_zscore(
    student_row: pd.Series,
    df_all: pd.DataFrame,
) -> Dict[str, Any]:
    """准备标准分雷达图数据（映射到百分制）。

    各科转化为标准分 z = (score - μ) / σ，再映射到 0-100 尺度：
      - 均分 = 50 分
      - 每高于均分 1 个标准差 +15 分
      - 每低于均分 1 个标准差 -15 分

    直观理解：60 = 比均分高约 0.67σ，70 = 高 1.33σ，以此类推。

    Returns:
        {
            "categories": ["语文", "数学", ...],
            "student_scores": [65.2, 48.3, ...],  # 百分制标准分
            "avg_scores": [50, 50, ...],            # 均分线 = 50
        }
    """
    categories = []
    student_scores = []
    avg_scores = []

    for subject in ALL_SUBJECTS:
        if subject not in df_all.columns:
            continue

        mean = df_all[subject].mean()
        std = df_all[subject].std()
        if std == 0 or pd.isna(std):
            continue

        label = SUBJECT_LABELS.get(subject, subject)
        student_score = student_row.get(subject)
        if pd.isna(student_score):
            continue  # 缺考跳过

        z = (float(student_score) - mean) / std
        # 映射到百分制：均分=50，±1σ=±15
        normalized = round(50 + z * 15, 1)
        # 限制在合理范围
        normalized = max(0, min(100, normalized))

        student_scores.append(normalized)
        avg_scores.append(50.0)
        categories.append(label)

    return {
        "categories": categories,
        "student_scores": student_scores,
        "avg_scores": avg_scores,
    }


def imbalance_index(
    student_row: pd.Series,
    df_all: pd.DataFrame,
) -> float:
    """计算偏科指数。

    逻辑：
    1. 全校每科计算 z-score (标准化到同一尺度)
    2. 学生各科 z-score 的标准差 = 偏科指数

    解读：
    - 0 ~ 0.3:  均衡发展
    - 0.3 ~ 0.6: 轻度偏科
    - 0.6 ~ 1.0: 中度偏科
    - > 1.0:    严重偏科

    Args:
        student_row: 学生数据
        df_all: 全校数据

    Returns:
        float: 偏科指数
    """
    z_scores = []

    for subject in ALL_SUBJECTS:
        if subject not in df_all.columns:
            continue

        student_score = student_row.get(subject)
        if pd.isna(student_score):
            continue

        mean = df_all[subject].mean()
        std = df_all[subject].std()
        if std == 0 or pd.isna(std):
            continue

        z = (float(student_score) - mean) / std
        z_scores.append(z)

    if len(z_scores) < 2:
        return 0.0

    return round(float(np.std(z_scores)), 3)


def imbalance_level(imbalance: float) -> Tuple[str, str]:
    """偏科指数 → 等级 + 图标。

    Returns:
        (等级文字, 图标)
    """
    if imbalance < 0.3:
        return "均衡发展", "🌟"
    elif imbalance < 0.6:
        return "轻度偏科", "📘"
    elif imbalance < 1.0:
        return "中度偏科", "⚠️"
    else:
        return "严重偏科", "🔴"


def percentile_rank(
    student_row: pd.Series,
    df_all: pd.DataFrame,
) -> Dict[str, Any]:
    """计算学生在各科的全校百分位。

    Returns:
        {
            "subject_percentiles": {"语文": 0.85, "数学": 0.92, ...},  # 超越%同学
            "total_percentile": 0.88,
            "total_rank_str": "42/856 (前 4.9%)",
        }
    """
    n_total = len(df_all)
    percentiles = {}

    for subject in ALL_SUBJECTS:
        if subject not in df_all.columns:
            continue

        student_score = student_row.get(subject)
        if pd.isna(student_score):
            percentiles[subject] = None
            continue

        # 百分位 = (不高于该分数的人数) / 总人数（含同分）
        rank = (df_all[subject] <= student_score).sum()
        pct = round(float(rank) / n_total, 4) if n_total > 0 else 0
        percentiles[subject] = pct

    # 总分百分位
    total_pct = None
    total_rank_str = ""
    if "total_score" in df_all.columns and "total_score" in student_row.index:
        total_score = student_row["total_score"]
        if not pd.isna(total_score):
            # 排名：高于该分数的人数+1（同分并列）
            rank = (df_all["total_score"] > total_score).sum() + 1
            pct = round(float(rank) / n_total, 4) if n_total > 0 else 0
            total_pct = pct
            total_rank_str = f"{rank}/{n_total} (前 {pct * 100:.1f}%)"

    return {
        "subject_percentiles": percentiles,
        "total_percentile": total_pct,
        "total_rank_str": total_rank_str,
    }


def strength_weakness(
    percentiles: Dict[str, Optional[float]],
) -> Dict[str, Any]:
    """基于百分位识别优势学科和薄弱学科。

    规则：
    - 排名前 25% → 优势学科
    - 排名后 25% → 薄弱学科
    - 中间 → 一般学科

    Args:
        percentiles: {"语文": 0.85, "数学": 0.15, ...}

    Returns:
        {
            "strengths": [("数学", 0.92), ...],  # 按百分位降序
            "weaknesses": [("语文", 0.28), ...],
            "normal": [("英语", 0.55), ...],
        }
    """
    strengths = []
    weaknesses = []
    normal = []

    for subject, pct in percentiles.items():
        if pct is None:
            continue
        label = SUBJECT_LABELS.get(subject, subject)
        if pct >= 0.75:
            strengths.append((label, pct))
        elif pct <= 0.25:
            weaknesses.append((label, pct))
        else:
            normal.append((label, pct))

    # 排序
    strengths.sort(key=lambda x: x[1], reverse=True)
    weaknesses.sort(key=lambda x: x[1])

    return {
        "strengths": strengths,
        "weaknesses": weaknesses,
        "normal": normal,
    }


def arts_science_bias(
    student_row: pd.Series,
    df_all: pd.DataFrame,
) -> Dict[str, Any]:
    """计算文科/理科倾向度。

    公式: arts_bias = mean(z_history, z_geography, z_morality_law)
                    - mean(z_physics, z_chemistry, z_biology)

    正值 → 偏文, 负值 → 偏理, 接近 0 → 均衡

    Returns:
        {
            "arts_bias": float,
            "direction": "偏理科" | "偏文科" | "文理均衡",
            "arts_z_mean": float,
            "science_z_mean": float,
        }
    """
    arts_subjects = SUBJECT_GROUPS["文科"]
    science_subjects = SUBJECT_GROUPS["理科"]

    def get_z_scores(subject_list):
        z_list = []
        for s in subject_list:
            if s not in df_all.columns:
                continue
            score = student_row.get(s)
            if pd.isna(score):
                continue
            mean = df_all[s].mean()
            std = df_all[s].std()
            if std == 0 or pd.isna(std):
                continue
            z_list.append((float(score) - mean) / std)
        return z_list

    arts_z = get_z_scores(arts_subjects)
    science_z = get_z_scores(science_subjects)

    arts_mean = float(np.mean(arts_z)) if arts_z else 0.0
    science_mean = float(np.mean(science_z)) if science_z else 0.0

    bias = round(arts_mean - science_mean, 3)

    if bias > 0.3:
        direction = "偏文科"
    elif bias < -0.3:
        direction = "偏理科"
    else:
        direction = "文理均衡"

    return {
        "arts_bias": bias,
        "direction": direction,
        "arts_z_mean": round(arts_mean, 3),
        "science_z_mean": round(science_mean, 3),
    }


def get_full_profile(
    student_row: pd.Series,
    df_all: pd.DataFrame,
    max_scores: Dict[str, float],
) -> Dict[str, Any]:
    """获取学生的完整画像数据。

    一站式函数，聚合所有画像维度。

    Returns:
        dict: 包含所有分析结果的完整画像
    """
    # 基本信息
    name = student_row.get("name", "未知")
    if pd.isna(name):
        name = "未知"
    school = student_row.get("graduate_school", "")
    if pd.isna(school):
        school = ""
    gender = student_row.get("gender", "")
    if pd.isna(gender):
        gender = ""
    exam_id = student_row.get("exam_id", "")
    total_score = student_row.get("total_score")

    # 雷达图数据
    radar = radar_data(student_row, df_all, max_scores)

    # 偏科指数
    imb = imbalance_index(student_row, df_all)
    imb_level, imb_icon = imbalance_level(imb)

    # 百分位
    pct_data = percentile_rank(student_row, df_all)

    # 优劣势
    sw = strength_weakness(pct_data["subject_percentiles"])

    # 文理倾向
    bias = arts_science_bias(student_row, df_all)

    # 各科具体分数
    subject_scores = {}
    for subject in ALL_SUBJECTS:
        if subject in student_row.index:
            score = student_row[subject]
            max_val = max_scores.get(subject, 100)
            subject_scores[subject] = {
                "score": None if pd.isna(score) else int(score),
                "max": max_val,
                "label": SUBJECT_LABELS.get(subject, subject),
            }

    return {
        "name": str(name),
        "exam_id": str(exam_id),
        "school": str(school),
        "gender": str(gender),
        "total_score": None if pd.isna(total_score) else int(total_score),
        "total_full": sum(max_scores.values()),
        "radar": radar,
        "imbalance_index": imb,
        "imbalance_level": imb_level,
        "imbalance_icon": imb_icon,
        "percentile": pct_data,
        "strength_weakness": sw,
        "arts_science_bias": bias,
        "subject_scores": subject_scores,
    }
