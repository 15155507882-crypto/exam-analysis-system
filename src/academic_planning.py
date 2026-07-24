"""学业规划建议模块。

功能：
- improvement_potential(): 提分空间排序
- subject_selection_advice(): 高中选科建议 (3+1+2)
- study_strategy(): 个性化学习策略
"""

from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from .utils import (
    SUBJECT_LABELS,
    ALL_SUBJECTS,
    SUBJECT_GROUPS,
)


def improvement_potential(
    student_row: pd.Series,
    df_all: pd.DataFrame,
    max_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    """提分潜力分析。

    逻辑：
    1. 计算每科的提分空间
    2. 低于全校 Q3（前 25% 门槛）→ 目标 = Q3，追赶先进
    3. 高于 Q3 → 目标 = 满分，挑战极限
    4. 按可提分排序 → 优先攻克提分空间最大的学科

    Args:
        student_row: 学生数据
        df_all: 全校数据
        max_scores: 满分配置

    Returns:
        按提分空间排序的列表:
        [{"subject": "语文", "current": 89, "target": 105, "potential": 16, "priority": 1}, ...]
    """
    result = []

    for subject in ALL_SUBJECTS:
        if subject not in df_all.columns:
            continue

        student_score = student_row.get(subject)
        if pd.isna(student_score):
            continue

        max_val = max_scores.get(subject, 100)
        current = float(student_score)

        # 全校前 25% 的门槛（Q3）
        q3 = df_all[subject].quantile(0.75)

        # 低于 Q3：追赶到 Q3；高于 Q3：冲刺满分
        if current < q3:
            target = q3
        else:
            target = max_val

        potential = round(target - current, 1)
        if potential < 0:
            potential = 0  # 已经满分

        result.append({
            "subject": subject,
            "label": SUBJECT_LABELS.get(subject, subject),
            "current": round(current, 1),
            "max": max_val,
            "target": round(target, 1),
            "potential": potential,
            "score_rate": round(current / max_val, 3),
        })

    # 按提分空间降序排列
    result.sort(key=lambda x: x["potential"], reverse=True)

    # 标记优先级
    for i, item in enumerate(result):
        item["priority"] = i + 1

    return result


def subject_selection_advice(
    student_row: pd.Series,
    df_all: pd.DataFrame,
) -> Dict[str, Any]:
    """高中选科建议（新高考 3+1+2 模式）。

    模式: 3(语数外) + 1(物理/历史) + 2(化/生/政/地 任选2)

    推荐逻辑：
    1. "1" 的选择：基于文理倾向度
    2. "2" 的选择：剩余 4 科中排名最高的 2 科

    Returns:
        {
            "recommended_1": "物理" | "历史",
            "reason_1": "...",
            "plan_a": {"name": "冲名校", "subjects": ["物理", "化学", "生物"]},
            "plan_b": {"name": "扬长避短", "subjects": [...]},
            "plan_c": {"name": "稳就业", "subjects": [...]},
            "z_scores": {"化学": 0.8, "生物": 0.5, "政治": -0.3, "地理": 0.2},
        }
    """
    arts_subjects = SUBJECT_GROUPS["文科"]
    science_subjects = SUBJECT_GROUPS["理科"]

    # 计算文理 z-score
    def calc_z_mean(sub_list):
        z_vals = []
        for s in sub_list:
            if s not in df_all.columns:
                continue
            score = student_row.get(s)
            if pd.isna(score):
                continue
            mean = df_all[s].mean()
            std = df_all[s].std()
            if std > 0:
                z_vals.append((float(score) - mean) / std)
        return float(np.mean(z_vals)) if z_vals else 0.0

    arts_z = calc_z_mean(arts_subjects)
    science_z = calc_z_mean(science_subjects)
    bias = arts_z - science_z

    # 判断 "1" 的选择
    if bias > 0.3:
        recommended_1 = "历史"
        reason_1 = "文科综合能力明显强于理科，建议选择历史方向"
    elif bias < -0.3:
        recommended_1 = "物理"
        reason_1 = "理科综合能力明显强于文科，建议选择物理方向"
    elif bias > 0.05:
        recommended_1 = "历史"
        reason_1 = "文理能力接近，但文科略有优势，推荐历史"
    elif bias < -0.05:
        recommended_1 = "物理"
        reason_1 = "文理能力接近，但理科略有优势，推荐物理"
    else:
        recommended_1 = "物理"
        reason_1 = "文理能力完全均衡，推荐物理（专业覆盖面更广）"

    # 计算 4 选 2 科目的 z-score
    candidates = {}
    for subj in ["chemistry", "biology"]:
        if subj in df_all.columns:
            score = student_row.get(subj)
            if not pd.isna(score):
                mean = df_all[subj].mean()
                std = df_all[subj].std()
                if std > 0:
                    candidates[subj] = (float(score) - mean) / std
    for subj in ["morality_law", "geography"]:
        if subj in df_all.columns:
            score = student_row.get(subj)
            if not pd.isna(score):
                mean = df_all[subj].mean()
                std = df_all[subj].std()
                if std > 0:
                    candidates[subj] = (float(score) - mean) / std

    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)

    def pick_top_2(exclude=None):
        result = []
        for subj, _ in sorted_candidates:
            if exclude and subj == exclude:
                continue
            result.append(subj)
            if len(result) == 2:
                break
        return result

    def to_labels(subj_list):
        return [SUBJECT_LABELS.get(s, s) for s in subj_list]

    # 方案 A: 取排名最高的 2 科
    plan_a_subjects = [recommended_1] + pick_top_2()

    # 方案 B: 排除排名第 1 的学科，选第 2+3（真正的备选方案）
    if len(sorted_candidates) >= 3:
        top1_subj = sorted_candidates[0][0]
        plan_b_2 = pick_top_2(exclude=top1_subj)
        plan_b_subjects = [recommended_1] + plan_b_2
        plan_b_name = f"备选（不选{SUBJECT_LABELS.get(top1_subj, top1_subj)}）"
    else:
        plan_b_subjects = plan_a_subjects
        plan_b_name = "备选（同上）"

    # 方案 C: 传统组合
    if recommended_1 == "物理":
        plan_c_subjects = ["物理", "化学", "生物"]
        plan_c_name = "传统理科（物化生）"
    else:
        plan_c_subjects = ["历史", "政治", "地理"]
        plan_c_name = "传统文科（历政地）"

    return {
        "recommended_1": recommended_1,
        "reason_1": reason_1,
        "plan_a": {
            "name": "最优方案（扬长）",
            "subjects": to_labels(plan_a_subjects),
            "raw": to_labels(plan_a_subjects),
        },
        "plan_b": {
            "name": plan_b_name,
            "subjects": to_labels(plan_b_subjects),
            "raw": to_labels(plan_b_subjects),
        },
        "plan_c": {
            "name": plan_c_name,
            "subjects": to_labels(plan_c_subjects),
            "raw": to_labels(plan_c_subjects),
        },
        "z_scores": {SUBJECT_LABELS.get(k, k): round(v, 2) for k, v in candidates.items()},
    }


def study_strategy(
    student_row: pd.Series,
    df_all: pd.DataFrame,
    max_scores: Dict[str, float],
    cluster_name: str = "",
) -> Dict[str, Any]:
    """生成个性化学习策略。

    基于学生画像生成暑期学习建议和时间分配。
    """
    # 计算各科与中位数的差距
    weaknesses = []
    for subject in ALL_SUBJECTS:
        if subject not in df_all.columns:
            continue
        score = student_row.get(subject)
        if pd.isna(score):
            continue
        max_val = max_scores.get(subject, 100)
        median = df_all[subject].median()
        gap = median - float(score)
        # 已经是满分的跳过
        if score >= max_val:
            continue
        weaknesses.append((subject, gap))

    # 优先取 gap > 0（真正低于中位数）的学科
    below_median = [(s, g) for s, g in weaknesses if g > 0]
    above_median = [(s, g) for s, g in weaknesses if g <= 0]
    below_median.sort(key=lambda x: x[1], reverse=True)  # 差距大的在前
    above_median.sort(key=lambda x: x[1], reverse=True)  # 接近中位数的在前

    # 重点攻克：只取真正低于中位数的（gap > 0），最多 3 科
    focus_pairs = below_median[:3]
    focus = [s for s, _ in focus_pairs]

    if not focus:
        # 全部高于中位数，不需要重点攻克
        return {
            "focus_subjects": [],
            "time_allocation": {},
            "tips": ["🎉 各科均高于全校中位数，保持当前学习节奏即可！", "📅 暑期可以预习高中内容，拓宽知识面。"],
        }

    # 学习策略提示
    tips = []

    for subj in focus:
        label = SUBJECT_LABELS.get(subj, subj)
        if subj in ["chinese", "history", "morality_law", "geography"]:
            tips.append(f"📖 **{label}**: 建议每日 1 篇文言文/时政阅读 + 素材积累")
        elif subj in ["math", "physics", "chemistry"]:
            tips.append(f"🧮 **{label}**: 建议每日刷 10 道基础题 + 3 道拓展题")
        elif subj == "english":
            tips.append(f"🌐 **{label}**: 建议每日 30 单词 + 1 篇阅读理解")
        elif subj == "pe":
            tips.append(f"🏃 **{label}**: 建议每日 30 分钟有氧运动")

    # 通用建议
    tips.append("📅 **暑期规划**: 上午主攻薄弱学科，下午巩固优势学科")
    tips.append("🎯 **目标设定**: 每周小测检验进步，动态调整方向")

    # 时间分配（按差距比例，但每科至少 15%）
    time_alloc = {}
    if focus:
        gaps = [max(next((g for s, g in focus_pairs if s == subj), 1), 1) for subj in focus]
        total_gap = sum(gaps)
        # 基础分配：每科至少 20%，剩余按比例
        base_pct = max(20, round(100 / len(focus)))
        remaining = 100 - base_pct * len(focus)
        for i, subj in enumerate(focus):
            gap = gaps[i]
            label = SUBJECT_LABELS.get(subj, subj)
            if total_gap > 0:
                extra = round(remaining * gap / total_gap)
            else:
                extra = 0
            time_alloc[label] = f"{base_pct + extra}%"

    return {
        "focus_subjects": [SUBJECT_LABELS.get(s, s) for s in focus],
        "time_allocation": time_alloc,
        "tips": tips,
    }
