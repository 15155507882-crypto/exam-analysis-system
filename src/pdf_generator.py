"""PDF 报告生成模块。

功能：
- render_html(): Jinja2 模板渲染（含雷达图+百分位图）
- generate_single_pdf(): 单份 PDF 生成
- generate_batch_pdf(): 批量 PDF 生成
- create_zip_archive(): ZIP 打包下载
"""

import os
import io
import base64
import zipfile
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

from jinja2 import Environment, FileSystemLoader

from .utils import (
    SUBJECT_LABELS,
    ALL_SUBJECTS,
    grade_level,
    grade_label,
    format_percent,
)

# 模板目录
TEMPLATE_DIR = Path(__file__).parent.parent / "templates"


def fig_to_base64(fig) -> str:
    """将 Plotly Figure 转换为 base64 PNG 字符串，用于嵌入 HTML/PDF。

    Returns:
        "data:image/png;base64,..." 格式的字符串
    """
    try:
        img_bytes = fig.to_image(format="png", width=500, height=400, scale=2)
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return ""


def render_html(
    profile_data: Dict[str, Any],
    improvements: List[Dict[str, Any]],
    selection_advice: Dict[str, Any],
    study_strategy_data: Dict[str, Any],
    school_name: str = "",
    radar_img: str = "",
    percentile_img: str = "",
) -> str:
    """使用 Jinja2 渲染个人报告 HTML。

    Args:
        profile_data: get_full_profile() 的输出
        improvements: improvement_potential() 的输出
        selection_advice: subject_selection_advice() 的输出
        study_strategy_data: study_strategy() 的输出
        school_name: 学校名称

    Returns:
        完整的 HTML 字符串
    """
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)))
    template = env.get_template("student_report.html")

    # 读取 CSS
    css_path = TEMPLATE_DIR / "report_style.css"
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
    else:
        css = ""

    # 学科成绩表数据
    subjects = []
    for subject in ALL_SUBJECTS:
        info = profile_data.get("subject_scores", {}).get(subject, {})
        score = info.get("score")
        max_val = info.get("max", 100)
        if score is not None and max_val > 0:
            rate = score / max_val
            rate_str = format_percent(rate)
            g = grade_label(grade_level(rate))
        else:
            rate_str = "—"
            g = "—"

        subjects.append({
            "label": info.get("label", SUBJECT_LABELS.get(subject, subject)),
            "score": score,
            "max": max_val,
            "rate": rate_str,
            "grade": g,
        })

    # 总分
    total_score = profile_data.get("total_score")
    total_full = profile_data.get("total_full", 1)
    if total_score is not None and total_full > 0:
        total_rate_val = total_score / total_full
        total_rate = format_percent(total_rate_val)
        total_grade = grade_label(grade_level(total_rate_val))
    else:
        total_rate = "—"
        total_grade = "—"

    # 优劣势
    sw = profile_data.get("strength_weakness", {})
    strengths = []
    for label, pct in sw.get("strengths", []):
        orig_key = next((k for k, v in SUBJECT_LABELS.items() if v == label), label)
        info = profile_data.get("subject_scores", {}).get(orig_key, {})
        strengths.append({
            "label": label,
            "score": info.get("score", "—"),
            "max": info.get("max", "—"),
        })

    weaknesses = []
    for label, pct in sw.get("weaknesses", []):
        orig_key = next((k for k, v in SUBJECT_LABELS.items() if v == label), label)
        info = profile_data.get("subject_scores", {}).get(orig_key, {})
        weaknesses.append({
            "label": label,
            "score": info.get("score", "—"),
            "max": info.get("max", "—"),
        })

    # 选科方案
    selection_plans = [
        {"name": selection_advice.get("plan_a", {}).get("name", "方案A"),
         "subjects": " + ".join(selection_advice.get("plan_a", {}).get("subjects", []))},
        {"name": selection_advice.get("plan_b", {}).get("name", "方案B"),
         "subjects": " + ".join(selection_advice.get("plan_b", {}).get("subjects", []))},
        {"name": selection_advice.get("plan_c", {}).get("name", "方案C"),
         "subjects": " + ".join(selection_advice.get("plan_c", {}).get("subjects", []))},
    ]

    context = {
        "css": css,
        "school_name": school_name or profile_data.get("school", "XX中学"),
        "name": profile_data.get("name", ""),
        "exam_id": profile_data.get("exam_id", ""),
        "school": profile_data.get("school", ""),
        "gender": profile_data.get("gender", ""),
        "subjects": subjects,
        "total_score": total_score,
        "total_full": total_full,
        "total_rate": total_rate,
        "total_grade": total_grade,
        "total_rank_str": profile_data.get("percentile", {}).get("total_rank_str", ""),
        "imbalance_index": profile_data.get("imbalance_index", 0),
        "imbalance_level": profile_data.get("imbalance_level", ""),
        "imbalance_icon": profile_data.get("imbalance_icon", ""),
        "arts_bias_direction": profile_data.get("arts_science_bias", {}).get("direction", ""),
        "arts_bias_value": profile_data.get("arts_science_bias", {}).get("arts_bias", 0),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "improvements": improvements[:5],
        "recommended_1": selection_advice.get("recommended_1", ""),
        "reason_1": selection_advice.get("reason_1", ""),
        "selection_plans": selection_plans,
        "study_tips": study_strategy_data.get("tips", []),
        "focus_subjects": study_strategy_data.get("focus_subjects", []),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "radar_img": radar_img,
        "percentile_img": percentile_img,
    }

    return template.render(**context)


def generate_single_pdf(html_content: str, output_path: str) -> bool:
    """使用 WeasyPrint 生成单份 PDF。

    Args:
        html_content: 渲染后的 HTML 字符串
        output_path: PDF 输出路径

    Returns:
        bool: 是否成功
    """
    try:
        from weasyprint import HTML
        HTML(string=html_content).write_pdf(output_path)
        return True
    except Exception as e:
        print(f"PDF 生成失败: {e}")
        return False


def generate_batch_pdf(
    df,
    max_scores: Dict[str, float],
    profile_func: Callable,
    planning_improvement_func: Callable,
    planning_selection_func: Callable,
    planning_strategy_func: Callable,
    output_dir: str,
    student_indices: Optional[List[int]] = None,
    progress_callback: Optional[Callable] = None,
) -> List[str]:
    """批量生成个人 PDF 报告。

    Args:
        df: 清洗后的 DataFrame
        max_scores: 满分配置
        profile_func: get_full_profile 函数
        planning_improvement_func: improvement_potential 函数
        planning_selection_func: subject_selection_advice 函数
        planning_strategy_func: study_strategy 函数
        output_dir: 输出目录
        student_indices: 需要生成的学生索引列表（None=全部）
        progress_callback: 进度回调 (current, total)

    Returns:
        生成的文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    if student_indices is None:
        student_indices = list(range(len(df)))

    total = len(student_indices)

    # 延迟导入可视化，避免循环依赖
    from .visualizations import plot_radar, plot_horizontal_percentile

    for i, idx in enumerate(student_indices):
        student_row = df.iloc[idx]

        # 获取画像数据
        profile_data = profile_func(student_row, df, max_scores)

        # 获取规划数据
        improvements = planning_improvement_func(student_row, df, max_scores)
        selection_advice = planning_selection_func(student_row, df)
        strategy_data = planning_strategy_func(student_row, df, max_scores)

        # 生成雷达图
        radar = profile_data.get("radar", {})
        radar_img = ""
        if radar.get("categories") and radar.get("student_rates"):
            fig_radar = plot_radar(
                categories=radar["categories"],
                values=radar["student_rates"],
                title=f"{profile_data.get('name', '')} - 各科得分率",
                reference_values=radar.get("avg_rates"),
                reference_label="全校均分",
            )
            radar_img = fig_to_base64(fig_radar)

        # 生成百分位图
        pct_data = profile_data.get("percentile", {}).get("subject_percentiles", {})
        percentile_img = ""
        if pct_data:
            cat_labels = [SUBJECT_LABELS.get(s, s) for s in pct_data.keys()]
            cat_values = [v if v is not None else 0 for v in pct_data.values()]
            fig_pct = plot_horizontal_percentile(
                categories=cat_labels,
                values=cat_values,
                title=f"{profile_data.get('name', '')} - 各科全校百分位",
            )
            percentile_img = fig_to_base64(fig_pct)

        # 渲染 HTML
        html = render_html(
            profile_data, improvements, selection_advice, strategy_data,
            radar_img=radar_img, percentile_img=percentile_img,
        )

        # 生成 PDF
        name = profile_data.get("name", f"student_{idx}")
        safe_name = "".join(c for c in str(name) if c.isalnum() or c in "_- ").strip()
        if not safe_name:
            safe_name = f"student_{idx}"

        output_path = os.path.join(output_dir, f"{safe_name}.pdf")

        if generate_single_pdf(html, output_path):
            generated.append(output_path)

        if progress_callback:
            progress_callback(i + 1, total)

    return generated


def create_zip_archive(file_paths: List[str], zip_path: str) -> str:
    """将 PDF 文件列表打包为 ZIP。

    Args:
        file_paths: PDF 文件路径列表
        zip_path: ZIP 输出路径

    Returns:
        ZIP 文件路径
    """
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in file_paths:
            zf.write(fp, os.path.basename(fp))

    return zip_path


def create_zip_in_memory(file_paths: List[str]) -> io.BytesIO:
    """在内存中创建 ZIP 存档。

    Returns:
        BytesIO 对象
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in file_paths:
            zf.write(fp, os.path.basename(fp))
    buffer.seek(0)
    return buffer
