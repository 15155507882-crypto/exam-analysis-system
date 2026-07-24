"""中考成绩分析系统 — Streamlit 主入口。

页面导航：
- 📤 数据导入：上传 Excel、满分配置、数据预览、校验报告
- 📊 全校宏观分析：总分分布、各科对比、相关性、男女对比
- 👤 个人画像：学生搜索、雷达图、偏科诊断、优劣势
- 🧩 学生聚类：K-Means 聚类、PCA 可视化、聚类画像
- 🎯 学业规划：提分潜力、选科建议、学习策略
- 📄 报告导出：HTML 报告、PDF 批量生成

用法：
    streamlit run app.py
"""

import sys
import os
import tempfile
import hashlib
from pathlib import Path

# 确保 src 在 Python 路径中
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.utils import (
    SUBJECT_LABELS,
    ALL_SUBJECTS,
    SUBJECT_GROUPS,
    grade_level,
    grade_label,
    format_percent,
    get_plotly_font_family,
)
from src.data_loader import (
    load_excel,
    detect_columns,
    get_matched_summary,
    clean_data,
    validate_data,
    get_summary,
    SCORE_COLUMNS,
)
from src import macro_analysis as macro
from src import student_profile as profile
from src import clustering as clust
from src import academic_planning as planning
from src import pdf_generator as pdf_gen
from src import task_manager as tm
from src.visualizations import (
    plot_distribution,
    plot_boxplot,
    plot_heatmap,
    plot_stacked_bar,
    plot_grouped_bar,
    plot_horizontal_bar,
    plot_radar,
    plot_horizontal_percentile,
    plot_scatter,
    plot_dual_line,
    get_plotly_font_family,
)

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="中考成绩分析系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 满分配置默认值
# ============================================================
DEFAULT_MAX_SCORES = {
    "chinese": 120,
    "math": 120,
    "english": 120,
    "physics": 100,
    "chemistry": 100,
    "biology": 100,
    "history": 100,
    "geography": 100,
    "morality_law": 100,
    "pe": 60,
}


def init_session_state():
    """初始化 session_state 中的默认值。"""
    # 首次启动时清除所有 Streamlit 缓存（防止旧数据残留）
    if "tasks_initialized" not in st.session_state:
        st.cache_data.clear()
        st.cache_resource.clear()

    defaults = {
        "df_raw": None,
        "df_clean": None,
        "column_mapping": None,
        "max_scores": DEFAULT_MAX_SCORES.copy(),
        "data_loaded": False,
        "validation_issues": [],
        "data_summary": None,
        # 任务持久化
        "current_task_id": None,
        "current_task_name": "",
        "tasks_list": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # 首次启动：从磁盘加载任务列表，尝试恢复最近任务
    if "tasks_initialized" not in st.session_state:
        st.session_state["tasks_list"] = tm.list_tasks()
        if st.session_state["tasks_list"] and not st.session_state["data_loaded"]:
            # 自动加载最近的任务
            last_task = st.session_state["tasks_list"][0]
            _load_task_to_session(last_task["id"])
        st.session_state["tasks_initialized"] = True


def _load_task_to_session(task_id: str):
    """从磁盘加载任务到 session_state。"""
    df, config = tm.load_task(task_id)
    if df is not None and config is not None:
        st.session_state["current_task_id"] = task_id
        st.session_state["current_task_name"] = tm.get_task_info(task_id).get("name", "")
        st.session_state["df_clean"] = df
        st.session_state["max_scores"] = config.get("max_scores", DEFAULT_MAX_SCORES.copy())
        st.session_state["data_summary"] = config.get("summary")
        st.session_state["validation_issues"] = config.get("validation_issues", [])
        st.session_state["column_mapping"] = config.get("column_mapping", {})
        st.session_state["data_loaded"] = True
        return True
    return False


def _refresh_tasks():
    """刷新任务列表到 session_state。"""
    st.session_state["tasks_list"] = tm.list_tasks()


# ============================================================
# 侧边栏导航
# ============================================================
def render_sidebar():
    """渲染侧边栏：任务管理 + 导航。"""
    with st.sidebar:
        st.title("📊 中考成绩分析系统")

        st.markdown("---")

        # ========================
        # 任务管理
        # ========================
        st.subheader("📋 任务管理")

        tasks = st.session_state.get("tasks_list", [])
        current_id = st.session_state.get("current_task_id")

        # 任务选择器
        if tasks:
            task_options = {t["id"]: f"{t['name']} ({t['student_count']}人)" for t in tasks}
            task_labels = list(task_options.values())
            task_ids = list(task_options.keys())

            # 找到当前任务在列表中的位置
            try:
                current_idx = task_ids.index(current_id) if current_id in task_ids else 0
            except ValueError:
                current_idx = 0

            def on_task_switch():
                new_id = task_ids[st.session_state.get("_task_selector_idx", 0)]
                if new_id != st.session_state.get("current_task_id"):
                    _load_task_to_session(new_id)
                    _refresh_tasks()

            st.selectbox(
                "切换任务",
                options=range(len(task_ids)),
                format_func=lambda i: task_labels[i],
                index=current_idx,
                key="_task_selector_idx",
                on_change=on_task_switch,
            )
        else:
            st.info("暂无任务，请新建")

        # 新建 / 删除按钮
        col_add, col_del = st.columns(2)
        with col_add:
            if st.button("➕ 新建任务", use_container_width=True, key="btn_new_task"):
                st.session_state["show_new_task_dialog"] = True

        with col_del:
            if tasks and current_id:
                if st.button("🗑 删除", use_container_width=True, key="btn_del_task"):
                    tm.delete_task(current_id)
                    _refresh_tasks()
                    # 清空当前数据
                    st.session_state["data_loaded"] = False
                    st.session_state["current_task_id"] = None
                    st.session_state["df_clean"] = None
                    # 尝试加载其他任务
                    remaining = tm.list_tasks()
                    if remaining:
                        _load_task_to_session(remaining[0]["id"])
                    st.rerun()

        # 新建任务弹窗
        if st.session_state.get("show_new_task_dialog", False):
            st.markdown("---")
            new_name = st.text_input("任务名称", placeholder="如: 2026年中考", key="new_task_name")
            col_ok, col_cancel = st.columns(2)
            with col_ok:
                if st.button("✅ 创建", use_container_width=True, key="btn_create_task"):
                    if new_name.strip():
                        task_id = tm.create_task(new_name.strip())
                        _refresh_tasks()
                        st.session_state["current_task_id"] = task_id
                        st.session_state["current_task_name"] = new_name.strip()
                        st.session_state["data_loaded"] = False
                        st.session_state["df_clean"] = None
                        st.session_state["max_scores"] = DEFAULT_MAX_SCORES.copy()
                        st.session_state["show_new_task_dialog"] = False
                        st.rerun()
                    else:
                        st.error("请输入任务名称")
            with col_cancel:
                if st.button("取消", use_container_width=True, key="btn_cancel_task"):
                    st.session_state["show_new_task_dialog"] = False
                    st.rerun()

        st.markdown("---")

        # ========================
        # 页面导航
        # ========================
        page = st.radio(
            "导航菜单",
            options=[
                "📤 数据导入",
                "📊 全校宏观分析",
                "👤 个人画像",
                "🧩 学生聚类",
                "🎯 学业规划",
                "📄 报告导出",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("---")

        # 数据状态指示
        if st.session_state.get("data_loaded"):
            n = len(st.session_state["df_clean"])
            task_name = st.session_state.get("current_task_name", "")
            st.success(f"✅ {task_name}: {n} 人")
        else:
            st.info("⏳ 请新建任务并导入数据")

        st.markdown("---")
        st.caption("v1.2 | 纯本地处理 · 数据不上传云端")

        st.markdown("---")
        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state["admin_authenticated"] = False
            st.rerun()

    return page


# ============================================================
# 页面：数据导入
# ============================================================
def render_data_import():
    """数据导入页面。"""
    st.header("📤 数据导入")

    # 检查是否有当前任务
    task_id = st.session_state.get("current_task_id")
    if not task_id:
        st.warning("⚠️ 请先在左侧边栏「新建任务」，再导入数据。")
        return

    task_name = st.session_state.get("current_task_name", "")
    st.caption(f"📋 当前任务: **{task_name}**")

    # 如果已有数据，显示概览
    if st.session_state.get("data_loaded"):
        st.info("✅ 该任务已有数据。上传新数据将覆盖。")

    # --- 第一步：上传文件 ---
    st.subheader("1. 上传中考成绩 Excel")
    uploaded_file = st.file_uploader(
        "支持 .xlsx / .xls 格式",
        type=["xlsx", "xls"],
        help="拖拽文件或点击选择。系统会自动识别字段名、清洗数据。",
    )

    if uploaded_file is not None:
        # 加载原始数据
        with st.spinner("正在读取 Excel 文件..."):
            try:
                df_raw = load_excel(uploaded_file)
            except Exception as e:
                st.error(f"❌ 读取 Excel 失败：{e}")
                return

        st.info(f"📋 原始数据: {len(df_raw)} 行 × {len(df_raw.columns)} 列")

        # --- 第二步：满分配置 ---
        st.subheader("2. 各科满分配置")
        st.caption("根据实际考试设定各科满分值，系统将据此计算得分率、等级等指标。")

        max_scores = {}
        cols = st.columns(4)
        for i, subject in enumerate(ALL_SUBJECTS):
            label = SUBJECT_LABELS.get(subject, subject)
            with cols[i % 4]:
                val = st.number_input(
                    f"{label}满分",
                    min_value=1,
                    max_value=300,
                    value=st.session_state["max_scores"].get(subject, 100),
                    step=10,
                    key=f"max_{subject}",
                )
                max_scores[subject] = val

        total_max = sum(max_scores.values())
        st.caption(f"📐 总分满分 = {total_max} 分")

        # --- 第三步：识别列映射 ---
        st.subheader("3. 字段识别")
        with st.spinner("正在识别字段映射..."):
            mapping = detect_columns(df_raw)
            match_summary = get_matched_summary(mapping)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**已识别成绩字段**")
            score_labels = [SUBJECT_LABELS.get(s, s) for s in match_summary["score_matched"]]
            if score_labels:
                for sl in score_labels:
                    st.success(f"✅ {sl}")
            if match_summary["score_missing"]:
                missing_labels = [SUBJECT_LABELS.get(s, s) for s in match_summary["score_missing"]]
                for ml in missing_labels:
                    st.error(f"❌ 未找到: {ml}")

        with col2:
            st.write("**已识别其他字段**")
            for field in match_summary["identity_matched"]:
                st.success(f"✅ {field}")
            for field in match_summary["demographic_matched"]:
                st.success(f"✅ {field}")

        if match_summary["unmapped_columns"]:
            st.warning(f"⚠️ 未匹配的列（将保留原列名）: {', '.join(match_summary['unmapped_columns'])}")

        # --- 第四步：清洗与校验 + 自动保存 ---
        if st.button("🚀 开始清洗与保存", type="primary", use_container_width=True):
            with st.spinner("正在清洗数据..."):
                df_clean = clean_data(df_raw, mapping)
                df_clean, issues = validate_data(df_clean, max_scores)
                summary = get_summary(df_clean)

            # 存入 session_state
            st.session_state["df_raw"] = df_raw
            st.session_state["df_clean"] = df_clean
            st.session_state["column_mapping"] = mapping
            st.session_state["max_scores"] = max_scores
            st.session_state["data_loaded"] = True
            st.session_state["validation_issues"] = issues
            st.session_state["data_summary"] = summary

            # 💾 持久化到磁盘
            tm.save_task(
                task_id=task_id,
                df=df_clean,
                max_scores=max_scores,
                summary=summary,
                column_mapping=mapping,
                validation_issues=issues,
            )
            _refresh_tasks()

            st.success(f"✅ 数据已清洗并保存到「{task_name}」！下次打开无需重新上传。")
            st.rerun()

    # 如果数据已加载，显示数据概览和预览
    if st.session_state.get("data_loaded"):
        show_data_overview()


def show_data_overview():
    """显示数据概览和预览表格。"""
    summary = st.session_state.get("data_summary", {})
    df_clean = st.session_state["df_clean"]
    issues = st.session_state.get("validation_issues", [])

    st.markdown("---")
    st.subheader("📋 数据概览")

    # 统计指标
    cols = st.columns(6)
    with cols[0]:
        st.metric("学生总数", f"{summary.get('total_students', 0)} 人")
    with cols[1]:
        st.metric("男生", f"{summary.get('male_count', 0)} 人")
    with cols[2]:
        st.metric("女生", f"{summary.get('female_count', 0)} 人")
    with cols[3]:
        st.metric("涉及学校", f"{summary.get('school_count', 0)} 所")
    with cols[4]:
        st.metric("数据完整度", format_percent(summary.get("completeness", 1.0)))
    with cols[5]:
        issue_count = len(issues)
        st.metric("校验问题", f"{issue_count} 项", delta=None if issue_count == 0 else f"⚠️ {issue_count}")

    # 总分统计
    ts = summary.get("total_score_stats")
    if ts:
        cols2 = st.columns(5)
        with cols2[0]:
            st.metric("均分", f"{ts['mean']}")
        with cols2[1]:
            st.metric("中位数", f"{ts['median']}")
        with cols2[2]:
            st.metric("标准差", f"{ts['std']}")
        with cols2[3]:
            st.metric("最高分", f"{ts['max']}")
        with cols2[4]:
            st.metric("最低分", f"{ts['min']}")

    # 校验问题
    if issues:
        st.markdown("---")
        st.subheader("⚠️ 数据校验报告")
        for issue in issues:
            icon = "🔴" if issue["level"] == "error" else "🟡"
            st.warning(f"{icon} {issue['message']}")

    # 各科统计表
    subject_stats = summary.get("subject_stats")
    if subject_stats:
        st.markdown("---")
        st.subheader("📊 各科统计")
        stats_rows = []
        for subject, s in subject_stats.items():
            stats_rows.append({
                "学科": SUBJECT_LABELS.get(subject, subject),
                "人数": s.get("count", 0),
                "均分": s.get("mean", 0),
                "中位数": s.get("median", 0),
                "标准差": s.get("std", 0),
                "最高分": s.get("max", 0),
                "最低分": s.get("min", 0),
            })
        stats_df = pd.DataFrame(stats_rows)
        st.dataframe(stats_df.set_index("学科"), use_container_width=True)

    # 数据预览表格
    st.markdown("---")
    st.subheader("📋 数据预览")
    show_all = st.checkbox("显示全部数据", value=False, key="show_all_preview")

    # 选择展示列：优先成绩列 + 姓名 + 学校
    display_cols = []
    for col in ["name", "exam_id", "gender", "graduate_school", "total_score"]:
        if col in df_clean.columns:
            display_cols.append(col)
    # 加上所有成绩列
    for subject in ALL_SUBJECTS:
        if subject in df_clean.columns and subject not in display_cols:
            display_cols.append(subject)

    # 加标记列
    flag_col = "_has_issues"
    if flag_col in df_clean.columns:
        display_cols.append(flag_col)

    # 确保只选存在的列
    display_cols = [c for c in display_cols if c in df_clean.columns]

    # 用中文标签
    rename_map = {k: SUBJECT_LABELS.get(k, k) for k in display_cols}
    rename_map["_has_issues"] = "⚠️异常"

    if show_all:
        preview_df = df_clean[display_cols].rename(columns=rename_map)
        st.caption(f"共 {len(df_clean)} 条记录")
    else:
        preview_df = df_clean[display_cols].head(20).rename(columns=rename_map)
        st.caption(f"共 {len(df_clean)} 条记录，显示前 20 条")
    st.dataframe(preview_df, use_container_width=True, hide_index=True)

    if len(df_clean) > 20:
        st.caption(f"共 {len(df_clean)} 条记录，仅显示前 20 条")


# ============================================================
# 页面：全校宏观分析
# ============================================================
def render_macro_analysis():
    """全校宏观分析页面。"""
    st.header("📊 全校宏观分析")

    df_clean = st.session_state["df_clean"]
    max_scores = st.session_state["max_scores"]

    # 筛选器
    with st.expander("🔍 筛选器", expanded=False):
        col1, col2, col3 = st.columns(3)
        gender_filter = None
        school_filter = None

        with col1:
            if "gender" in df_clean.columns:
                genders = ["全部"] + sorted(df_clean["gender"].dropna().unique().tolist())
                gender_filter = st.selectbox("性别", genders, key="macro_gender")
        with col2:
            school_col = None
            for sc in ["graduate_school", "registration_point"]:
                if sc in df_clean.columns:
                    school_col = sc
                    break
            if school_col:
                schools = ["全部"] + sorted(df_clean[school_col].dropna().unique().tolist())
                school_filter = st.selectbox("学校", schools, key="macro_school")

    # 应用筛选
    filtered_df = df_clean.copy()
    if gender_filter and gender_filter != "全部":
        filtered_df = filtered_df[filtered_df["gender"] == gender_filter]
    if school_filter and school_filter != "全部" and school_col:
        filtered_df = filtered_df[filtered_df[school_col] == school_filter]

    if len(filtered_df) == 0:
        st.warning("筛选后无数据，请调整筛选条件。")
        return

    st.caption(f"当前筛选: {len(filtered_df)} 名学生")

    # Tab 切换
    tabs = st.tabs(["📈 总分分布", "📊 各科对比", "🔗 学科相关性", "⚡ 男女对比", "🏅 等级分布", "🔺 分数段"])

    # --- Tab 1: 总分分布 ---
    with tabs[0]:
        dist_result = macro.score_distribution(filtered_df, max_scores)

        if dist_result["stats"]:
            st.subheader("描述统计")
            stats = dist_result["stats"]
            cols = st.columns(4)
            with cols[0]:
                st.metric("均分", f"{stats['mean']}")
                st.metric("标准差", f"{stats['std']}")
            with cols[1]:
                st.metric("中位数", f"{stats['median']}")
                st.metric("偏度", f"{stats['skewness']}")
            with cols[2]:
                st.metric("最高分", f"{stats['max']}")
                st.metric("峰度", f"{stats['kurtosis']}")
            with cols[3]:
                st.metric("最低分", f"{stats['min']}")
                if dist_result["normality"]:
                    p = dist_result["normality"]["shapiro_pvalue"]
                    is_norm = dist_result["normality"]["is_normal"]
                    st.metric("正态性", f"{'✅ 正态' if is_norm else '❌ 非正态'} (p={p:.3f})")

            # 分布直方图
            st.subheader("总分分布")
            fig = plot_distribution(
                filtered_df["total_score"].dropna(),
                title=f"总分分布直方图 (n={len(filtered_df)})",
                vertical_lines={
                    "均值": stats["mean"],
                    "中位数": stats["median"],
                },
            )
            st.plotly_chart(fig, use_container_width=True)

    # --- Tab 2: 各科对比 ---
    with tabs[1]:
        comp = macro.subject_comparison(filtered_df, max_scores)
        if comp["subject_stats"] is not None and len(comp["subject_stats"]) > 0:
            st.subheader("各科统计表")
            st.dataframe(
                comp["subject_stats"].set_index("学科"),
                use_container_width=True,
            )

            st.subheader("各科成绩箱线图")
            subjects = [s for s in ALL_SUBJECTS if s in filtered_df.columns]
            fig = plot_boxplot(filtered_df, subjects, title="各科成绩分布对比")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("无成绩数据可分析。")

    # --- Tab 3: 学科相关性 ---
    with tabs[2]:
        corr_result = macro.subject_correlation(filtered_df)
        if corr_result["corr_matrix"] is not None:
            st.subheader("相关性热力图")
            fig = plot_heatmap(corr_result["corr_matrix"], title="学科 Pearson 相关系数矩阵")
            st.plotly_chart(fig, use_container_width=True)

            if corr_result["top_correlations"]:
                st.subheader("最强相关学科对")
                for subj1, subj2, r_val in corr_result["top_correlations"]:
                    if abs(r_val) > 0.3:
                        direction = "正相关" if r_val > 0 else "负相关"
                        icon = "🔗" if abs(r_val) > 0.6 else "➡️"
                        st.write(f"{icon} **{subj1} ↔ {subj2}**: r={r_val:.3f} ({direction})")
        else:
            st.info("学科数据不足，无法计算相关性。")

    # --- Tab 4: 男女对比 ---
    with tabs[3]:
        gender_result = macro.gender_comparison(filtered_df)
        if gender_result["gender_stats"] is not None and len(gender_result["gender_stats"]) > 0:
            st.subheader("男女生各科均分对比")

            # 柱状图
            stats_df = gender_result["gender_stats"]
            categories = stats_df["学科"].tolist()
            fig = plot_grouped_bar(
                categories=categories,
                values_dict={
                    "男生": stats_df["男生均分"].tolist(),
                    "女生": stats_df["女生均分"].tolist(),
                },
                title="男女生各科均分对比",
                ylabel="均分",
            )
            st.plotly_chart(fig, use_container_width=True)

            # t 检验结果
            if gender_result["ttest_results"]:
                st.subheader("独立样本 t 检验结果")
                for r in gender_result["ttest_results"]:
                    sig_mark = "🔴" if r["highly_significant"] else ("🟡" if r["significant"] else "⚪")
                    diff = r["male_mean"] - r["female_mean"]
                    direction = "男高" if diff > 0 else ("女高" if diff < 0 else "持平")
                    st.write(
                        f"{sig_mark} **{r['label']}**: "
                        f"男 {r['male_mean']} vs 女 {r['female_mean']} "
                        f"(差 {diff:+.1f}, t={r['t_stat']:.2f}, p={r['p_value']:.4f}) → {direction}"
                    )
        else:
            st.info("缺少性别字段，无法进行男女对比分析。")

    # --- Tab 5: 等级分布 ---
    with tabs[4]:
        rates_df = macro.excellence_rates(filtered_df, max_scores)
        if len(rates_df) > 0:
            st.subheader("各科 ABCD 等级分布")

            # 堆叠柱状图 - 人数
            st.write("**人数分布**")
            count_df = rates_df[["学科", "A优秀", "B良好", "C及格", "D不及格"]].copy()
            count_df = count_df.set_index("学科")
            fig = plot_stacked_bar(
                rates_df,
                x_col="学科",
                y_cols=["A优秀", "B良好", "C及格", "D不及格"],
                title="各科等级人数分布",
                ylabel="人数",
            )
            st.plotly_chart(fig, use_container_width=True)

            # 堆叠柱状图 - 比例
            st.write("**比例分布**")
            pct_df = rates_df[["学科", "A比例", "B比例", "C比例", "D比例"]].copy()
            pct_df["A比例"] = pct_df["A比例"] * 100
            pct_df["B比例"] = pct_df["B比例"] * 100
            pct_df["C比例"] = pct_df["C比例"] * 100
            pct_df["D比例"] = pct_df["D比例"] * 100
            fig2 = go.Figure()
            colors = ["#2ECC40", "#0074D9", "#FF851B", "#FF4136"]
            labels_map = {"A比例": "A优秀%", "B比例": "B良好%", "C比例": "C及格%", "D比例": "D不及格%"}
            for i, col in enumerate(["A比例", "B比例", "C比例", "D比例"]):
                fig2.add_trace(go.Bar(
                    x=pct_df["学科"], y=pct_df[col],
                    name=labels_map[col],
                    marker_color=colors[i],
                ))
            fig2.update_layout(
                title="各科等级比例分布",
                yaxis_title="比例 (%)",
                barmode="stack",
                font=dict(family=get_plotly_font_family()),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("无成绩数据可分析。")

    # --- Tab 6: 分数段金字塔 ---
    with tabs[5]:
        band_df = macro.score_band_analysis(filtered_df, max_scores)
        if len(band_df) > 0:
            st.subheader("分数段金字塔")
            fig = plot_horizontal_bar(
                labels=band_df["分数层"].tolist(),
                values=band_df["人数"].tolist(),
                title="各分数层人数分布",
                xlabel="人数",
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("分数段统计表")
            display_df = band_df.copy()
            display_df["占比"] = display_df["占比"].apply(lambda x: f"{x * 100:.1f}%")
            st.dataframe(display_df.set_index("分数层"), use_container_width=True)
        else:
            st.info("无总分数据可分析。")


# ============================================================
# 页面：个人画像
# ============================================================
def render_student_profile():
    """学生个人画像页面。"""
    st.header("👤 学生个人画像")

    df_clean = st.session_state["df_clean"]
    max_scores = st.session_state["max_scores"]

    # 初始化 session_state 中的选中索引
    if "profile_selected_idx" not in st.session_state:
        st.session_state["profile_selected_idx"] = df_clean.index[0]

    # 学生选择
    st.subheader("选择学生")

    col_search, col_select = st.columns([1, 1])

    with col_search:
        search_text = st.text_input(
            "🔍 搜索姓名或考生号",
            placeholder="输入姓名或考生号...",
            key="profile_search_input",
        )

    # 构建选项列表（保持原始 df 索引）
    student_options = []
    for i, row in df_clean.iterrows():
        name = row.get("name", "")
        if pd.isna(name):
            name = ""
        exam_id = row.get("exam_id", "")
        if pd.isna(exam_id):
            exam_id = ""
        label = f"{name} ({exam_id})" if name else str(exam_id)
        student_options.append((i, label))

    # 搜索过滤
    if search_text:
        filtered_options = [
            (i, label) for i, label in student_options
            if search_text.lower() in label.lower()
        ]
    else:
        filtered_options = student_options

    if not filtered_options:
        st.warning("未找到匹配的学生。")
        return

    # 确保当前选中的索引在过滤后的列表中
    current_idx = st.session_state["profile_selected_idx"]
    valid_indices = [i for i, _ in filtered_options]
    if current_idx not in valid_indices:
        current_idx = valid_indices[0]
        st.session_state["profile_selected_idx"] = current_idx

    # 找到当前索引在过滤列表中的位置
    try:
        default_pos = valid_indices.index(current_idx)
    except ValueError:
        default_pos = 0
        st.session_state["profile_selected_idx"] = valid_indices[0]

    with col_select:
        selected_pos = st.selectbox(
            "选择学生",
            options=range(len(filtered_options)),
            format_func=lambda pos: filtered_options[pos][1] if 0 <= pos < len(filtered_options) else "",
            index=default_pos,
            key="_profile_select_pos",
        )

    # 直接从返回值更新 session_state（不用 on_change，因为回调触发时 widget 值尚未更新）
    if 0 <= selected_pos < len(valid_indices):
        st.session_state["profile_selected_idx"] = valid_indices[selected_pos]

    selected_idx = st.session_state["profile_selected_idx"]

    # 整个画像内容区用容器包裹，key 包含 selected_idx，切换学生时整体重建
    profile_content = st.container(key=f"profile_all_{selected_idx}")

    with profile_content:

        # 获取学生完整画像
        student_row = df_clean.loc[selected_idx]
        p = profile.get_full_profile(student_row, df_clean, max_scores)

        # ============================
        # 基本信息卡片
        # ============================
        st.markdown("---")
        info_box = st.container(key=f"info_{selected_idx}")
        with info_box:
            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.caption("姓名")
                st.subheader(p["name"])
            with col2:
                st.caption("性别")
                st.subheader(p["gender"] if p["gender"] else "未知")
            with col3:
                st.caption("总分")
                ts = p["total_score"]
                st.subheader(f"{ts} / {p['total_full']}" if ts else "未知")
            with col4:
                st.caption("全校排名")
                st.subheader(p["percentile"]["total_rank_str"] or "未知")
            with col5:
                st.caption("文理倾向")
                st.subheader(p["arts_science_bias"]["direction"])

        # ============================
        # 左右布局：雷达图 + 优劣势
        # ============================
        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("📊 学科雷达图")

            # 切换雷达图模式
            radar_mode = st.radio(
                "视图模式",
                options=["百分制得分率", "标准分 (百分制)"],
                horizontal=True,
                key=f"radar_mode_{selected_idx}",
            )

            if radar_mode == "标准分 (百分制)":
                radar_z = profile.radar_data_zscore(student_row, df_clean)
                fig_radar = plot_radar(
                    categories=radar_z["categories"],
                    values=radar_z["student_scores"],
                    title=f"{p['name']} - 各科标准分（均分=50, ±1σ=±15）",
                    reference_values=radar_z["avg_scores"],
                    reference_label="全校均分 (50)",
                    fill_color="rgba(234, 67, 53, 0.25)",
                    line_color="rgba(234, 67, 53, 1)",
                )
            else:
                radar = p["radar"]
                fig_radar = plot_radar(
                    categories=radar["categories"],
                    values=radar["student_rates"],
                    title=f"{p['name']} - 各科得分率 (%)",
                    reference_values=radar["avg_rates"],
                    reference_label="全校均分",
                )
            st.plotly_chart(fig_radar, use_container_width=True, key=f"radar_{selected_idx}_{radar_mode}")

            # 偏科指数
            st.metric(
                f"📐 偏科指数: {p['imbalance_index']:.3f}",
                f"{p['imbalance_icon']} {p['imbalance_level']}",
            )

        with col_right:
            st.subheader("🎯 优劣势学科")

            sw = p["strength_weakness"]

            # 优势学科
            st.write("**🌟 优势学科**")
            if sw["strengths"]:
                for subj_label, pct in sw["strengths"]:
                    st.success(f"**{subj_label}** — 超越全校 {pct * 100:.0f}% 同学")
            else:
                st.caption("无明显优势学科")

            # 薄弱学科
            st.write("**⚠️ 薄弱学科**")
            if sw["weaknesses"]:
                for subj_label, pct in sw["weaknesses"]:
                    st.error(f"**{subj_label}** — 仅超越 {pct * 100:.0f}% 同学")
            else:
                st.caption("无明显薄弱学科")

            # 文理倾向解释
            bias = p["arts_science_bias"]
            st.subheader("📈 文理倾向")
            st.write(f"**{bias['direction']}** (偏差值: {bias['arts_bias']:.2f})")
            st.caption(f"文科 z-score 均值: {bias['arts_z_mean']:.2f} | 理科 z-score 均值: {bias['science_z_mean']:.2f}")

        # ============================
        # 学科百分位条形图
        # ============================
        st.markdown("---")
        st.subheader("📊 各科全校百分位")

        pct_data = p["percentile"]["subject_percentiles"]
        if pct_data:
            cat_labels = [SUBJECT_LABELS.get(s, s) for s in pct_data.keys()]
            cat_values = [v if v is not None else 0 for v in pct_data.values()]

            fig_pct = plot_horizontal_percentile(
                categories=cat_labels,
                values=cat_values,
                title=f"{p['name']} - 各科超越全校比例",
            )
            st.plotly_chart(fig_pct, use_container_width=True, key=f"pct_{selected_idx}")

        # ============================
        # 各科具体得分
        # ============================
        st.markdown("---")
        st.subheader("📋 各科详细得分")

        score_data = p["subject_scores"]
        if score_data:
            subject_box = st.container(key=f"subject_scores_{selected_idx}")
            with subject_box:
                cols = st.columns(5)
                for i, (subject, info) in enumerate(score_data.items()):
                    with cols[i % 5]:
                        score = info["score"]
                        max_val = info["max"]
                        display_score = f"{score}/{max_val}" if score is not None else "缺失"
                        st.caption(info["label"])
                        st.subheader(display_score)


# ============================================================
# 页面：学生聚类
# ============================================================
def render_clustering():
    """学生聚类分析页面。"""
    st.header("🧩 学生聚类分析")

    df_clean = st.session_state["df_clean"]
    max_scores = st.session_state["max_scores"]

    # 准备特征
    X_scaled, feature_cols, scaler = clust.prepare_features(df_clean, max_scores)

    # K 值选择器
    st.subheader("聚类参数")
    col1, col2 = st.columns([1, 3])
    with col1:
        n_clusters = st.slider(
            "聚类数 K",
            min_value=3,
            max_value=8,
            value=4,
            step=1,
            key="cluster_k",
        )

    # 执行聚类
    labels, kmeans_model = clust.run_kmeans(X_scaled, n_clusters=n_clusters)
    profiles = clust.cluster_profiles(df_clean, labels, feature_cols, max_scores, scaler)

    # K 值选择分析
    with st.expander("📈 K 值选择分析（肘部法则 + 轮廓系数）", expanded=False):
        k_info = clust.find_optimal_k(X_scaled)
        fig_k = plot_dual_line(
            x=k_info["k_values"],
            y1=k_info["wcss"],
            y1_name="WCSS (簇内平方和)",
            y2=k_info["silhouette"],
            y2_name="轮廓系数",
            title="最优 K 值选择",
        )
        st.plotly_chart(fig_k, use_container_width=True)
        st.info(f"📌 推荐 K = **{k_info['optimal_k']}** (轮廓系数最大)")

    # 聚类概览
    st.subheader("聚类概览")
    cluster_box = st.container(key=f"cluster_overview_{n_clusters}")
    with cluster_box:
        summary_df = clust.get_cluster_summary(df_clean, labels, profiles)
        cols = st.columns(len(profiles))
        for i, p in enumerate(profiles):
            with cols[i]:
                st.caption(f"聚类 {p['cluster_id']}")
                st.subheader(p["name"])
                st.write(f"{p['size']} 人 ({p['percent']}%)")

    # PCA 可视化
    st.subheader("PCA 聚类可视化")
    X_pca, pca_model = clust.pca_transform(X_scaled)

    # 悬停文本
    hover_texts = []
    for idx in range(len(df_clean)):
        name = df_clean.iloc[idx].get("name", "")
        total = df_clean.iloc[idx].get("total_score", "")
        if pd.isna(name):
            name = ""
        hover_texts.append(f"{name} | 总分: {total}")

    cluster_names = {p["cluster_id"]: p["name"] for p in profiles}
    fig_pca = plot_scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        labels=labels,
        hover_texts=hover_texts,
        title=f"PCA 2D 聚类可视化 (K={n_clusters})",
        xlabel=f"PC1 ({pca_model.explained_variance_ratio_[0]*100:.1f}%)",
        ylabel=f"PC2 ({pca_model.explained_variance_ratio_[1]*100:.1f}%)",
        cluster_names=cluster_names,
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    st.caption("💡 PCA 把 10 科成绩压缩到 2 维平面。每个点 = 一个学生，颜色 = 聚类归属。点越近 = 成绩模式越像。")

    # 各聚类画像详情
    st.subheader("各聚类画像详情")
    for p in profiles:
        with st.expander(f"聚类 {p['cluster_id']}: {p['name']} ({p['size']} 人, {p['percent']}%)"):
            # 各科均分表格
            if p["subject_means"]:
                means_df = pd.DataFrame(
                    list(p["subject_means"].items()),
                    columns=["学科", "均分"],
                ).set_index("学科")
                st.dataframe(means_df, use_container_width=True)

            # 雷达图：该聚类各科均分 vs 全校均分
            if p["subject_means"]:
                cat = list(p["subject_means"].keys())
                val = list(p["subject_means"].values())
                # 全校均分
                avg_vals = []
                for c in cat:
                    orig_key = next((k for k, v in SUBJECT_LABELS.items() if v == c), c)
                    if orig_key in df_clean.columns:
                        avg_vals.append(round(float(df_clean[orig_key].mean()), 1))
                    else:
                        avg_vals.append(0)
                fig_cluster_radar = plot_radar(
                    categories=cat,
                    values=val,
                    title=f"{p['name']} - 各科均分",
                    reference_values=avg_vals,
                    reference_label="全校均分",
                )
                st.plotly_chart(fig_cluster_radar, use_container_width=True, key=f"cluster_radar_{n_clusters}_{p['cluster_id']}")


# ============================================================
# 页面：学业规划
# ============================================================
def render_academic_planning():
    """学业规划建议页面。"""
    st.header("🎯 学业规划建议")

    df_clean = st.session_state["df_clean"]
    max_scores = st.session_state["max_scores"]

    # 初始化 session_state
    if "plan_selected_idx" not in st.session_state:
        st.session_state["plan_selected_idx"] = df_clean.index[0]

    # 学生选择
    st.subheader("选择学生")

    search_text = st.text_input(
        "🔍 搜索姓名或考生号",
        placeholder="输入姓名或考生号...",
        key="plan_search_input",
    )

    student_options = []
    for i, row in df_clean.iterrows():
        name = row.get("name", "")
        if pd.isna(name):
            name = ""
        exam_id = row.get("exam_id", "")
        if pd.isna(exam_id):
            exam_id = ""
        label = f"{name} ({exam_id})" if name else str(exam_id)
        student_options.append((i, label))

    if search_text:
        filtered_options = [(i, l) for i, l in student_options if search_text.lower() in l.lower()]
    else:
        filtered_options = student_options

    if not filtered_options:
        st.warning("未找到匹配的学生。")
        return

    # 确保选中索引在过滤列表中
    current_idx = st.session_state["plan_selected_idx"]
    valid_indices = [i for i, _ in filtered_options]
    if current_idx not in valid_indices:
        current_idx = valid_indices[0]
        st.session_state["plan_selected_idx"] = current_idx

    try:
        default_pos = valid_indices.index(current_idx)
    except ValueError:
        default_pos = 0
        st.session_state["plan_selected_idx"] = valid_indices[0]

    selected_pos = st.selectbox(
        "选择学生",
        options=range(len(filtered_options)),
        format_func=lambda pos: filtered_options[pos][1] if 0 <= pos < len(filtered_options) else "",
        index=default_pos,
        key="_plan_select_pos",
    )

    # 直接从返回值更新（不用 on_change 回调，回调时 widget 值尚未更新）
    if 0 <= selected_pos < len(valid_indices):
        st.session_state["plan_selected_idx"] = valid_indices[selected_pos]

    selected_idx = st.session_state["plan_selected_idx"]

    student_row = df_clean.loc[selected_idx]
    student_name = student_row.get("name", "未知")
    if pd.isna(student_name):
        student_name = "未知"

    # ============================
    # 1. 提分潜力分析
    # ============================
    st.markdown("---")
    st.subheader(f"📈 {student_name} - 提分潜力分析")
    st.caption("💡 **Q3** = 全校前 25% 的门槛分数（即排名前 25% 的学生的最低分）。低于 Q3 时目标是追上 Q3，高于 Q3 时目标是冲刺满分。")

    imp = planning.improvement_potential(student_row, df_clean, max_scores)

    if imp:
        # 表格
        imp_df = pd.DataFrame(imp)
        imp_df = imp_df[["priority", "label", "current", "target", "potential", "score_rate"]]
        imp_df.columns = ["优先级", "学科", "当前分", "目标分", "可提分", "得分率"]
        imp_df["得分率"] = imp_df["得分率"].apply(lambda x: f"{x * 100:.0f}%")
        imp_df = imp_df.set_index("优先级")
        st.dataframe(imp_df, use_container_width=True)

        # 横向柱状图
        plot_imp = imp_df.copy().reset_index()
        fig_imp = plot_horizontal_bar(
            labels=plot_imp["学科"].tolist(),
            values=plot_imp["可提分"].tolist(),
            title="各科提分空间（目标：追 Q3 / 冲满分）",
            xlabel="可提分",
        )
        st.plotly_chart(fig_imp, use_container_width=True, key=f"plan_imp_{selected_idx}")
    else:
        st.info("无成绩数据可分析。")

    # ============================
    # 2. 高中选科建议
    # ============================
    st.markdown("---")
    st.subheader(f"🎓 {student_name} - 高中选科建议 (3+1+2)")

    selection = planning.subject_selection_advice(student_row, df_clean)

    # "1" 的选择
    plan_container = st.container(key=f"plan_container_{selected_idx}")
    with plan_container:
        st.write(f"### 「1」的选择: **{selection['recommended_1']}**")
        st.info(selection["reason_1"])

    # 三套方案
    st.write("### 三套选科方案")
    plan_cols = st.columns(3)

    plans = [selection["plan_a"], selection["plan_b"], selection["plan_c"]]
    for i, plan in enumerate(plans):
        with plan_cols[i]:
            st.markdown(f"**{plan['name']}**")
            for subj in plan["subjects"]:
                st.write(f"- {subj}")
            raw_str = " + ".join(plan.get("raw", plan["subjects"]))
            st.caption(f"组合: {raw_str}")

    # 各科 z-score 展示
    if selection["z_scores"]:
        st.write("### 选考科目 z-score 对比")
        z_df = pd.DataFrame(
            list(selection["z_scores"].items()),
            columns=["学科", "z-score"],
        ).set_index("学科")
        st.dataframe(z_df, use_container_width=True)

    # ============================
    # 3. 学习策略
    # ============================
    st.markdown("---")
    st.subheader(f"📋 {student_name} - 个性化学习策略")

    # 获取聚类名称
    cluster_name = ""
    try:
        X_scaled, feature_cols, scaler = clust.prepare_features(df_clean, max_scores)
        labels, _ = clust.run_kmeans(X_scaled, n_clusters=4)
        profiles = clust.cluster_profiles(df_clean, labels, feature_cols, max_scores, scaler)
        student_label = labels[selected_idx]
        for p in profiles:
            if p["cluster_id"] == student_label:
                cluster_name = p["name"]
                break
    except Exception:
        pass

    strategy = planning.study_strategy(student_row, df_clean, max_scores, cluster_name)

    if cluster_name:
        st.caption(f"学生类型: **{cluster_name}**")

    st.write("### 🎯 暑期重点攻克")
    for subj in strategy["focus_subjects"]:
        st.write(f"- **{subj}**")

    st.write("### ⏰ 建议时间分配")
    time_box = st.container(key=f"time_alloc_{selected_idx}")
    with time_box:
        time_cols = st.columns(len(strategy["time_allocation"]))
        for i, (subj, pct) in enumerate(strategy["time_allocation"].items()):
            with time_cols[i]:
                st.caption(subj)
                st.subheader(pct)

    st.write("### 💡 学习建议")
    for tip in strategy["tips"]:
        st.write(tip)


# ============================================================
# 页面：报告导出
# ============================================================
def render_report_export():
    """报告导出页面。"""
    st.header("📄 报告导出")

    df_clean = st.session_state["df_clean"]
    max_scores = st.session_state["max_scores"]

    tab_html, tab_pdf = st.tabs(["📊 HTML 交互报告", "📑 个人 PDF 报告"])

    # ============================
    # Tab 1: HTML 报告
    # ============================
    with tab_html:
        st.subheader("导出交互式 HTML 报告")

        st.write("将当前所有分析页面整合为一个独立的 HTML 文件，可在浏览器中离线查看。")

        options_html = st.multiselect(
            "选择导出内容",
            options=["全校宏观分析", "学生聚类分析", "各科统计表格"],
            default=["全校宏观分析", "学生聚类分析"],
            key="html_export_options",
        )

        st.info("💡 HTML 报告不包含个人画像（内容较多），个人画像请使用 PDF 报告。")

        if st.button("📥 生成 HTML 报告", type="primary", key="gen_html"):
            with st.spinner("正在生成 HTML 报告..."):
                # 收集各个分析页面的图表和表格
                html_parts = []

                # 基础样式
                html_parts.append(f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>中考成绩分析报告</title>
<style>
body {{ font-family: {get_plotly_font_family()}, sans-serif; max-width: 1100px; margin: 0 auto; padding: 20px; }}
h1 {{ color: #1a73e8; border-bottom: 3px solid #1a73e8; padding-bottom: 10px; }}
h2 {{ color: #333; margin-top: 30px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th, td {{ padding: 8px 12px; text-align: center; border-bottom: 1px solid #eee; }}
th {{ background: #1a73e8; color: white; }}
</style>
</head>
<body>
<h1>📊 中考成绩分析报告</h1>
<p>生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')} | 学生总数: {len(df_clean)}</p>
""")

                # 总分分布
                if "全校宏观分析" in options_html:
                    dist = macro.score_distribution(df_clean, max_scores)
                    if dist["stats"]:
                        html_parts.append("<h2>总分分布</h2>")
                        fig = plot_distribution(
                            df_clean["total_score"].dropna(),
                            title="总分分布",
                            vertical_lines={"均值": dist["stats"]["mean"], "中位数": dist["stats"]["median"]},
                        )
                        html_parts.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))

                # 各科对比
                if "全校宏观分析" in options_html:
                    comp = macro.subject_comparison(df_clean, max_scores)
                    if comp["subject_stats"] is not None:
                        html_parts.append("<h2>各科成绩对比</h2>")
                        fig = plot_boxplot(df_clean, [s for s in ALL_SUBJECTS if s in df_clean.columns])
                        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

                # 相关性
                if "全校宏观分析" in options_html:
                    corr_result = macro.subject_correlation(df_clean)
                    if corr_result["corr_matrix"] is not None:
                        html_parts.append("<h2>学科相关性</h2>")
                        fig = plot_heatmap(corr_result["corr_matrix"])
                        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))

                # 聚类
                if "学生聚类分析" in options_html:
                    try:
                        X_scaled, feats, scaler = clust.prepare_features(df_clean, max_scores)
                        labels, _ = clust.run_kmeans(X_scaled, n_clusters=4)
                        X_pca, _ = clust.pca_transform(X_scaled)
                        profiles = clust.cluster_profiles(df_clean, labels, feats, max_scores, scaler)

                        html_parts.append("<h2>学生聚类分析</h2>")
                        hover = [f"{df_clean.iloc[i].get('name','')} | {df_clean.iloc[i].get('total_score','')}" for i in range(len(df_clean))]
                        cluster_names = {p["cluster_id"]: p["name"] for p in profiles}
                        fig = plot_scatter(X_pca[:, 0], X_pca[:, 1], labels, hover, cluster_names=cluster_names)
                        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
                    except Exception:
                        pass

                html_parts.append("</body></html>")
                full_html = "\n".join(html_parts)

            st.download_button(
                label="📥 下载 HTML 报告",
                data=full_html,
                file_name=f"中考成绩分析报告_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.html",
                mime="text/html",
                use_container_width=True,
            )
            st.success("✅ HTML 报告已生成，点击上方按钮下载！")

    # ============================
    # Tab 2: PDF 报告
    # ============================
    with tab_pdf:
        st.subheader("批量生成个人 PDF 报告")

        # 选择范围
        st.write("**选择导出范围**")
        export_mode = st.radio(
            "导出范围",
            options=["全部学生", "按学校筛选", "按聚类筛选", "前 N 名"],
            horizontal=True,
            key="pdf_export_mode",
        )

        indices_to_export = list(range(len(df_clean)))

        if export_mode == "按学校筛选":
            school_col = None
            for sc in ["graduate_school", "registration_point"]:
                if sc in df_clean.columns:
                    school_col = sc
                    break
            if school_col:
                schools = sorted(df_clean[school_col].dropna().unique().tolist())
                selected_school = st.selectbox("选择学校", schools, key="pdf_school")
                school_mask = df_clean[school_col] == selected_school
                indices_to_export = [i for i in range(len(df_clean)) if school_mask.iloc[i]]
                st.caption(f"已选 {len(indices_to_export)} 名学生")
            else:
                st.warning("数据中无学校字段。")

        elif export_mode == "按聚类筛选":
            try:
                X_scaled, feats, scaler = clust.prepare_features(df_clean, max_scores)
                labels, _ = clust.run_kmeans(X_scaled, n_clusters=4)
                profiles = clust.cluster_profiles(df_clean, labels, feats, max_scores, scaler)
                cluster_options = [f"聚类 {p['cluster_id']}: {p['name']} ({p['size']}人)" for p in profiles]
                selected_cluster = st.selectbox("选择聚类", cluster_options, key="pdf_cluster")
                selected_id = int(selected_cluster.split(":")[0].replace("聚类 ", ""))
                indices_to_export = [i for i in range(len(df_clean)) if labels[i] == selected_id]
                st.caption(f"已选 {len(indices_to_export)} 名学生")
            except Exception as e:
                st.error(f"聚类分析失败: {e}")
                indices_to_export = []

        elif export_mode == "前 N 名":
            top_n = st.number_input("导出前 N 名", min_value=1, max_value=len(df_clean), value=10, key="pdf_topn")
            if "total_score" in df_clean.columns:
                sorted_indices = df_clean["total_score"].argsort()[::-1].tolist()
                indices_to_export = sorted_indices[:top_n]
            else:
                indices_to_export = list(range(min(top_n, len(df_clean))))

        st.metric("将导出", f"{len(indices_to_export)} 份 PDF 报告")

        # PDF 选项
        include_selection = st.checkbox("包含高中选科建议", value=True, key="pdf_include_selection")
        include_strategy = st.checkbox("包含暑期学习策略", value=True, key="pdf_include_strategy")

        if st.button("📥 批量生成 PDF (ZIP 下载)", type="primary", key="gen_pdf", disabled=len(indices_to_export) == 0):
            progress_bar = st.progress(0)
            status_text = st.empty()

            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_dir = os.path.join(tmpdir, "pdfs")
                os.makedirs(pdf_dir, exist_ok=True)

                def progress_cb(current, total):
                    progress_bar.progress(current / total)
                    status_text.text(f"正在生成: {current}/{total}")

                try:
                    generated = pdf_gen.generate_batch_pdf(
                        df_clean,
                        max_scores,
                        profile.get_full_profile,
                        planning.improvement_potential,
                        planning.subject_selection_advice,
                        planning.study_strategy,
                        pdf_dir,
                        student_indices=indices_to_export,
                        progress_callback=progress_cb,
                    )

                    if generated:
                        # 打包 ZIP
                        zip_buffer = pdf_gen.create_zip_in_memory(generated)
                        status_text.text(f"✅ 已生成 {len(generated)} 份 PDF")

                        st.download_button(
                            label=f"📥 下载 ZIP ({len(generated)} 份 PDF)",
                            data=zip_buffer,
                            file_name=f"中考个人报告_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.zip",
                            mime="application/zip",
                            use_container_width=True,
                        )
                    else:
                        st.error("❌ PDF 生成失败，请检查 WeasyPrint 是否正确安装。")
                        st.info("macOS 可能需要: brew install pango")
                except Exception as e:
                    st.error(f"❌ 生成过程出错: {e}")


# ============================================================
# 占位页面（后续里程碑实现）
# ============================================================
def render_placeholder(page_name: str, description: str):
    """渲染占位页面，提示功能将在后续里程碑实现。"""
    st.header(page_name)
    st.info(f"🚧 {description}")
    st.caption("该功能将在后续开发阶段实现。")


# ============================================================
# 管理员登录
# ============================================================

# 密码的 SHA-256 哈希（盐值固定，防止明文存储）
# 默认密码: admin123 — 部署后可修改此哈希值更换密码
ADMIN_PASSWORD_HASH = "08957258f1a9b2b9cb95d858b4d00cfe2d681e03b5686707b4de35d1b00f77c7"
LOGIN_SALT = "zcode_exam_2024"


def hash_password(password: str) -> str:
    """对密码加盐哈希。"""
    return hashlib.sha256(f"{LOGIN_SALT}:{password}".encode()).hexdigest()


def check_login():
    """检查是否已登录。"""
    return st.session_state.get("admin_authenticated", False)


def render_login():
    """渲染登录页面。"""
    st.markdown("<br><br>", unsafe_allow_html=True)

    col_center = st.columns([1, 2, 1])
    with col_center[1]:
        st.markdown("### 🔐 管理员登录")
        st.caption("中考成绩分析系统")

        password = st.text_input(
            "请输入管理员密码",
            type="password",
            placeholder="请输入密码",
            key="login_password_input",
        )

        if st.button("登录", type="primary", use_container_width=True):
            if hash_password(password) == ADMIN_PASSWORD_HASH:
                st.session_state["admin_authenticated"] = True
                st.rerun()
            else:
                st.error("❌ 密码错误")

        st.caption("—")


# ============================================================
# 主入口
# ============================================================
def main():
    init_session_state()

    # 未登录 → 只显示登录页
    if not check_login():
        render_login()
        return

    page = render_sidebar()

    if page == "📤 数据导入":
        render_data_import()
    elif page == "📊 全校宏观分析":
        if st.session_state.get("data_loaded"):
            render_macro_analysis()
        else:
            st.warning("⚠️ 请先在「数据导入」页面上传并清洗数据。")
    elif page == "👤 个人画像":
        if st.session_state.get("data_loaded"):
            render_student_profile()
        else:
            st.warning("⚠️ 请先在「数据导入」页面上传并清洗数据。")
    elif page == "🧩 学生聚类":
        if st.session_state.get("data_loaded"):
            render_clustering()
        else:
            st.warning("⚠️ 请先在「数据导入」页面上传并清洗数据。")
    elif page == "🎯 学业规划":
        if st.session_state.get("data_loaded"):
            render_academic_planning()
        else:
            st.warning("⚠️ 请先在「数据导入」页面上传并清洗数据。")
    elif page == "📄 报告导出":
        if st.session_state.get("data_loaded"):
            render_report_export()
        else:
            st.warning("⚠️ 请先在「数据导入」页面上传并清洗数据。")


if __name__ == "__main__":
    main()
