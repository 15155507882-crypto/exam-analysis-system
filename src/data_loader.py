"""数据加载与清洗模块。

功能：
- load_excel(): 读取 .xlsx/.xls 文件
- detect_columns(): 基于列名模糊匹配，自动识别字段映射
- clean_data(): 数据类型转换、去除空格
- validate_data(): 数据校验（成绩范围、重复考号、缺失检测）
- get_summary(): 数据概览统计
"""

import re
from typing import Dict, List, Optional, Tuple, Any
from io import BytesIO

import pandas as pd
import numpy as np

# ============================================================
# 列名映射表（原始中文列名 → 内部字段名）
# ============================================================

# 身份信息字段
IDENTITY_COLUMNS: Dict[str, List[str]] = {
    "name": ["姓名", "学生姓名", "考生姓名"],
    "exam_id": ["考生号", "准考证号", "考试号", "报名号"],
    "id_card": ["身份证号", "身份证号码", "证件号码"],
    "phone": ["移动电话", "电话", "手机号", "联系电话", "手机号码"],
    "recipient": ["收件人"],
    "contact": ["联系人", "联系人姓名"],
    "address": ["家庭地址", "通讯地址", "地址"],
    "address_detail": ["家庭地址详情", "地址详情", "详细地址"],
    "zip_code": ["邮政编码", "邮编"],
}

# 人口学信息字段
DEMOGRAPHIC_COLUMNS: Dict[str, List[str]] = {
    "gender": ["性别"],
    "ethnicity": ["民族"],
    "city": ["地市", "城市", "地区"],
    "hukou_location": ["户口所在地", "户籍所在地"],
    "hukou_type": ["户籍类型", "户口类型"],
    "political_status": ["政治面貌"],
    "candidate_type": ["考生类别", "考生类型"],
    "plan_type": ["计划类别"],
    "registration_point": ["报名点"],
    "school_district": ["中学所在地区", "学校所在地区", "学校地区"],
    "graduate_school": ["毕业中学", "毕业学校", "中学"],
    "birth_date": ["出生日期", "出生年月", "生日"],
}

# 成绩字段
SCORE_COLUMNS: Dict[str, List[str]] = {
    "total_score": ["总分", "总成绩", "合计总分"],
    "total_rank": ["总分名次", "总分排名", "名次", "排名"],
    "chinese": ["语文"],
    "math": ["数学"],
    "english": ["英语", "外语"],
    "physics": ["物理"],
    "chemistry": ["化学"],
    "biology": ["生物"],
    "history": ["历史"],
    "geography": ["地理"],
    "morality_law": ["道法", "道德与法治", "政治", "思想政治", "思想品德"],
    "pe": ["体育", "体育与健康"],
}

# 合并所有映射
ALL_COLUMN_MAPS = {**IDENTITY_COLUMNS, **DEMOGRAPHIC_COLUMNS, **SCORE_COLUMNS}

# 需要转换为 float 的字段
NUMERIC_FIELDS = list(SCORE_COLUMNS.keys()) + ["total_rank"]

# 需要转换为 str 的字段
STRING_FIELDS = [
    "name", "exam_id", "id_card", "phone", "recipient", "contact",
    "address", "address_detail", "zip_code", "ethnicity", "city",
    "hukou_location", "hukou_type", "political_status", "candidate_type",
    "plan_type", "registration_point", "school_district", "graduate_school",
    "gender",
]


def _fuzzy_match(col_name: str, candidates: List[str]) -> bool:
    """模糊匹配：列名包含候选词的任意一个。

    清洗后做包含匹配，忽略空格。
    """
    cleaned = col_name.strip().replace(" ", "").replace("\u3000", "")
    for candidate in candidates:
        c_clean = candidate.strip().replace(" ", "").replace("\u3000", "")
        if c_clean in cleaned or cleaned in c_clean:
            return True
    return False


def load_excel(file) -> pd.DataFrame:
    """读取上传的 Excel 文件。

    Args:
        file: Streamlit UploadedFile 对象 或 文件路径

    Returns:
        pd.DataFrame: 原始数据
    """
    if isinstance(file, str):
        # 文件路径
        if file.endswith(".xlsx"):
            return pd.read_excel(file, engine="openpyxl")
        else:
            return pd.read_excel(file)
    else:
        # BytesIO
        content = file.read()
        if file.name.endswith(".xlsx"):
            return pd.read_excel(BytesIO(content), engine="openpyxl")
        else:
            return pd.read_excel(BytesIO(content))


def detect_columns(df: pd.DataFrame) -> Dict[str, str]:
    """自动识别列名映射。

    遍历所有列，逐一匹配到内部字段名。
    对未匹配的列发出 warning（但不影响流程）。

    Args:
        df: 原始 DataFrame

    Returns:
        Dict[str, str]: {"原始列名": "内部字段名"}
    """
    mapping: Dict[str, str] = {}
    matched_internal: set = set()

    for col in df.columns:
        col_str = str(col)
        found = False
        for internal_name, candidates in ALL_COLUMN_MAPS.items():
            if internal_name in matched_internal:
                continue
            if _fuzzy_match(col_str, candidates):
                mapping[col_str] = internal_name
                matched_internal.add(internal_name)
                found = True
                break
        # 未匹配的列保留原名
        if not found:
            mapping[col_str] = col_str

    return mapping


def get_matched_summary(mapping: Dict[str, str]) -> Dict[str, list]:
    """分析列映射结果，分类列出已匹配/未匹配字段。

    Returns:
        {
            "identity_matched": [...],
            "demographic_matched": [...],
            "score_matched": [...],
            "score_missing": [...],
            "unmapped_columns": [...],
        }
    """
    identity_fields = set(IDENTITY_COLUMNS.keys())
    demo_fields = set(DEMOGRAPHIC_COLUMNS.keys())
    score_fields = set(SCORE_COLUMNS.keys())

    mapped_internal = set(mapping.values())

    result = {
        "identity_matched": sorted(mapped_internal & identity_fields),
        "demographic_matched": sorted(mapped_internal & demo_fields),
        "score_matched": sorted(mapped_internal & score_fields),
        "score_missing": sorted(score_fields - mapped_internal),
        "unmapped_columns": [
            orig for orig, internal in mapping.items() if internal not in ALL_COLUMN_MAPS
        ],
    }
    return result


def clean_data(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    """清洗数据：重命名列、类型转换、去除空格。

    Args:
        df: 原始 DataFrame
        mapping: detect_columns() 的输出

    Returns:
        清洗后的 DataFrame（内部字段名）
    """
    # 1. 重命名列
    rename_map = {orig: internal for orig, internal in mapping.items() if orig != internal}
    df = df.rename(columns=rename_map)

    # 2. 去除字符串字段的前后空格
    for field in STRING_FIELDS:
        if field in df.columns:
            df[field] = df[field].astype(str).str.strip()
            # 将 "nan" / "None" / "" 替换为 pd.NA
            df[field] = df[field].replace(["nan", "None", ""], pd.NA)

    # 3. 数值字段转换
    for field in NUMERIC_FIELDS:
        if field in df.columns:
            df[field] = pd.to_numeric(df[field], errors="coerce")

    # 4. 出生日期转换
    if "birth_date" in df.columns:
        df["birth_date"] = pd.to_datetime(df["birth_date"], errors="coerce")

    return df


def validate_data(df: pd.DataFrame, max_scores: Dict[str, float]) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """数据校验。

    校验规则：
    - 成绩字段：不得为空（体育除外），0 <= 分数 <= 满分
    - 总分与各科加和一致性（允许 ±3 分误差）
    - 考生号不允许重复
    - 身份证号格式校验（18 位或空）

    Args:
        df: 清洗后的 DataFrame
        max_scores: 各科满分配置 {"chinese": 120, ...}

    Returns:
        (df_with_flags, issues):
        - df_with_flags: 增加标记列的 DataFrame
        - issues: 校验问题列表 [{"level": "error"/"warning", "message": "...", "count": N}, ...]
    """
    issues = []

    # --- 成绩范围校验 ---
    for subject, max_val in max_scores.items():
        if subject not in df.columns:
            continue
        col = df[subject]
        # 允许体育为空（免考）
        if subject == "pe":
            valid_mask = col.notna()
            if valid_mask.any():
                out_of_range = valid_mask & ((col < 0) | (col > max_val))
            else:
                out_of_range = pd.Series(False, index=df.index)
        else:
            out_of_range = col.isna() | (col < 0) | (col > max_val)

        if out_of_range.any():
            flag_col = f"_flag_{subject}"
            df[flag_col] = df.get(flag_col, "")
            df.loc[out_of_range, flag_col] = df.loc[out_of_range, flag_col].apply(
                lambda x: (x + "; " if x else "") + "成绩异常"
            )
            n = out_of_range.sum()
            level = "error" if (n > len(df) * 0.1) else "warning"
            issues.append({
                "level": level,
                "field": subject,
                "message": f"{subject} 有 {n} 条成绩异常（缺失或超出 0~{max_val} 范围）",
                "count": n,
            })

    # --- 总分一致性校验 ---
    score_subjects = [s for s in max_scores.keys() if s in df.columns]
    if "total_score" in df.columns and len(score_subjects) >= 3:
        calculated = df[score_subjects].sum(axis=1, skipna=False)
        diff = (df["total_score"] - calculated).abs()
        total_issue = diff > 3  # 允许 ±3 分误差
        if total_issue.any():
            n = total_issue.sum()
            df.loc[total_issue, "_flag_total_mismatch"] = "总分与各科加和不一致"
            issues.append({
                "level": "warning",
                "field": "total_score",
                "message": f"{n} 名学生总分与各科加和不一致（偏差 > 3 分）",
                "count": n,
            })

    # --- 考生号重复校验 ---
    if "exam_id" in df.columns:
        valid_exam = df["exam_id"].notna()
        dup = df.loc[valid_exam, "exam_id"].duplicated(keep=False)
        if dup.any():
            n = dup.sum()
            df.loc[dup[dup].index, "_flag_dup_exam"] = "考生号重复"
            issues.append({
                "level": "error",
                "field": "exam_id",
                "message": f"发现 {n} 条重复考生号",
                "count": n,
            })

    # --- 身份证号格式校验 ---
    if "id_card" in df.columns:
        id_pattern = re.compile(r"^\d{17}[\dXx]$|^$")
        valid_ids = df["id_card"].notna() & (df["id_card"] != "")
        if valid_ids.any():
            bad_ids = valid_ids & ~df.loc[valid_ids, "id_card"].str.match(id_pattern)
            if bad_ids.any():
                n = bad_ids.sum()
                df.loc[bad_ids, "_flag_id_format"] = "身份证号格式异常"
                issues.append({
                    "level": "warning",
                    "field": "id_card",
                    "message": f"{n} 条身份证号格式不符合 18 位规则",
                    "count": n,
                })

    # --- 汇总标记列 ---
    flag_cols = [c for c in df.columns if c.startswith("_flag_")]
    if flag_cols:
        df["_has_issues"] = df[flag_cols].apply(
            lambda row: any(str(v) not in ("", "nan", "None", None) and pd.notna(v) for v in row),
            axis=1,
        )
    else:
        df["_has_issues"] = False

    return df, issues


def get_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """生成数据概览统计。

    Returns:
        {
            "total_students": int,
            "male_count": int,
            "female_count": int,
            "school_count": int,
            "completeness": float (0~1),
            "total_score_stats": {...} or None,
            "subject_stats": {...} or None,
        }
    """
    summary: Dict[str, Any] = {
        "total_students": len(df),
        "male_count": 0,
        "female_count": 0,
        "school_count": 0,
        "completeness": 1.0,
        "total_score_stats": None,
        "subject_stats": None,
    }

    # 性别统计
    if "gender" in df.columns:
        gender_series = df["gender"].str.strip()
        summary["male_count"] = int((gender_series == "男").sum())
        summary["female_count"] = int((gender_series == "女").sum())

    # 学校数
    for school_col in ["graduate_school", "school_district", "registration_point"]:
        if school_col in df.columns:
            n_schools = df[school_col].dropna().nunique()
            if n_schools > 0:
                summary["school_count"] = int(n_schools)
                break

    # 数据完整度（成绩字段非空比例）
    score_fields = [c for c in df.columns if c in SCORE_COLUMNS and c != "total_rank"]
    if score_fields:
        notna_counts = df[score_fields].notna().sum()
        total_cells = len(df) * len(score_fields)
        summary["completeness"] = round(notna_counts.sum() / total_cells, 4) if total_cells > 0 else 0.0

    # 总分统计
    if "total_score" in df.columns:
        ts = df["total_score"].dropna()
        if len(ts) > 0:
            summary["total_score_stats"] = {
                "mean": round(float(ts.mean()), 1),
                "median": round(float(ts.median()), 1),
                "std": round(float(ts.std()), 1),
                "min": round(float(ts.min()), 1),
                "max": round(float(ts.max()), 1),
            }

    # 各科统计
    subject_stats = {}
    for subject in SCORE_COLUMNS:
        if subject == "total_rank":
            continue
        if subject in df.columns:
            s = df[subject].dropna()
            if len(s) > 0:
                subject_stats[subject] = {
                    "count": int(len(s)),
                    "mean": round(float(s.mean()), 1),
                    "median": round(float(s.median()), 1),
                    "std": round(float(s.std()), 1),
                    "min": round(float(s.min()), 1),
                    "max": round(float(s.max()), 1),
                }
    if subject_stats:
        summary["subject_stats"] = subject_stats

    return summary
