"""可视化引擎 — 基于 Plotly 的交互式图表生成。

提供统一的中文字体配置和各类图表的封装函数。
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .utils import (
    get_plotly_font_family,
    SUBJECT_LABELS,
    ALL_SUBJECTS,
    grade_label,
)

# ============================================================
# 全局 Plotly 中文字体配置
# ============================================================

_plotly_font = get_plotly_font_family()

PLOTLY_FONT = dict(family=_plotly_font, size=12)
PLOTLY_TITLE_FONT = dict(family=_plotly_font, size=16)
PLOTLY_LAYOUT_DEFAULTS = dict(
    font=PLOTLY_FONT,
    title_font=PLOTLY_TITLE_FONT,
    template="plotly_white",
    hoverlabel=dict(font=PLOTLY_FONT),
)

# 颜色主题
COLOR_PALETTE = px.colors.qualitative.Set2


def _apply_defaults(fig: go.Figure, title: str = "") -> go.Figure:
    """应用默认布局配置。"""
    fig.update_layout(
        title=dict(text=title, font=PLOTLY_TITLE_FONT) if title else None,
        font=PLOTLY_FONT,
        template="plotly_white",
        hoverlabel=dict(font=PLOTLY_FONT),
    )
    return fig


# ============================================================
# V1: 分布直方图 + KDE
# ============================================================

def plot_distribution(
    data: pd.Series,
    title: str = "分数分布",
    xlabel: str = "分数",
    ylabel: str = "人数",
    nbins: int = 30,
    show_kde: bool = True,
    show_rug: bool = False,
    vertical_lines: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """绘制分数分布直方图 + KDE 曲线。

    Args:
        data: 分数序列
        title: 图表标题
        xlabel, ylabel: 轴标签
        nbins: 直方图分箱数
        show_kde: 是否显示核密度估计曲线
        show_rug: 是否显示底部 rug plot
        vertical_lines: 标注竖线 {"均值": 85.0, "中位数": 82.0}

    Returns:
        Plotly Figure
    """
    valid = data.dropna()
    fig = go.Figure()

    # 直方图
    fig.add_trace(go.Histogram(
        x=valid,
        nbinsx=nbins,
        name="人数",
        marker_color=COLOR_PALETTE[0],
        opacity=0.7,
    ))

    if show_kde:
        # 核密度曲线（用双 y 轴）
        kde_x = np.linspace(valid.min(), valid.max(), 200)
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(valid)
            kde_y = kde(kde_x) * len(valid) * (valid.max() - valid.min()) / nbins
            fig.add_trace(go.Scatter(
                x=kde_x,
                y=kde_y,
                mode="lines",
                name="密度曲线",
                line=dict(color=COLOR_PALETTE[1], width=2.5),
                yaxis="y",
            ))
        except Exception:
            pass  # KDE 失败时跳过

    if show_rug:
        fig.add_trace(go.Scatter(
            x=valid,
            y=[-0.5] * len(valid),
            mode="markers",
            marker=dict(symbol="line-ns", size=8, color="rgba(0,0,0,0.3)"),
            name="数据点",
            showlegend=False,
        ))

    # 标注竖线
    if vertical_lines:
        colors = ["red", "orange", "green", "purple"]
        for i, (label, value) in enumerate(vertical_lines.items()):
            fig.add_vline(
                x=value,
                line_dash="dash",
                line_color=colors[i % len(colors)],
                annotation_text=label,
                annotation_position="top",
            )

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        bargap=0.05,
        showlegend=True,
    )
    return _apply_defaults(fig, title)


# ============================================================
# V2: 箱线图（多学科并列）
# ============================================================

def plot_boxplot(
    df: pd.DataFrame,
    subjects: List[str],
    title: str = "各科成绩箱线图",
    max_scores: Optional[Dict[str, float]] = None,
) -> go.Figure:
    """绘制多学科并列箱线图。

    Args:
        df: 包含各科成绩的 DataFrame
        subjects: 学科字段列表
        title: 标题
        max_scores: 满分映射，用于显示得分率参考线

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    for i, subject in enumerate(subjects):
        if subject not in df.columns:
            continue
        valid = df[subject].dropna()
        if len(valid) == 0:
            continue

        label = SUBJECT_LABELS.get(subject, subject)
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        fig.add_trace(go.Box(
            y=valid,
            name=label,
            marker_color=color,
            boxmean="sd",  # 显示均值和标准差
            hoverinfo="y+name",
            hoverlabel=dict(font=PLOTLY_FONT),
        ))

    fig.update_layout(
        title=title,
        yaxis_title="分数",
        showlegend=False,
        boxmode="group",
    )
    return _apply_defaults(fig, title)


# ============================================================
# V3: 相关性热力图
# ============================================================

def plot_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "学科相关性矩阵",
    annotate: bool = True,
    cmap: str = "RdBu_r",
) -> go.Figure:
    """绘制相关性热力图。

    Args:
        corr_matrix: 相关性矩阵 (DataFrame)
        title: 标题
        annotate: 是否在格内显示数值
        cmap: 颜色映射

    Returns:
        Plotly Figure
    """
    # 转换为中文标签
    labels = [SUBJECT_LABELS.get(c, c) for c in corr_matrix.columns]
    z = corr_matrix.values

    if annotate:
        # 用 heatmap trace 显示数值
        annotations = []
        for i, row_label in enumerate(labels):
            for j, col_label in enumerate(labels):
                annotations.append(dict(
                    x=j, y=i,
                    text=f"{z[i][j]:.2f}",
                    showarrow=False,
                    font=dict(color="white" if abs(z[i][j]) > 0.5 else "black", size=10),
                ))

        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale=cmap,
            zmid=0,
            zmin=-1,
            zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in z],
            hoverinfo="text",
            hoverlabel=dict(font=PLOTLY_FONT),
        ))
        fig.update_layout(annotations=annotations)
    else:
        fig = go.Figure(data=go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            colorscale=cmap,
            zmid=0,
            zmin=-1,
            zmax=1,
            hoverlabel=dict(font=PLOTLY_FONT),
        ))

    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=45),
        width=650,
        height=600,
    )
    return _apply_defaults(fig, title)


# ============================================================
# V4: 堆叠柱状图
# ============================================================

def plot_stacked_bar(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str = "等级分布",
    xlabel: str = "",
    ylabel: str = "人数 / 比例",
    horizontal: bool = False,
    normalize: bool = False,
) -> go.Figure:
    """绘制堆叠柱状图（用于等级分布等）。

    Args:
        df: 数据 DataFrame
        x_col: x 轴列
        y_cols: 堆叠的 y 列（按顺序从左到右）
        title: 标题
        xlabel, ylabel: 轴标签
        horizontal: True 为横向柱状图
        normalize: 是否归一化为百分比

    Returns:
        Plotly Figure
    """
    fig = go.Figure()

    for i, col in enumerate(y_cols):
        label = SUBJECT_LABELS.get(col, col)
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]
        values = df[col].values
        if normalize and values.sum() > 0:
            values = values / values.sum() * 100

        if horizontal:
            fig.add_trace(go.Bar(
                y=df[x_col],
                x=values,
                name=label,
                orientation="h",
                marker_color=color,
                hoverlabel=dict(font=PLOTLY_FONT),
            ))
        else:
            fig.add_trace(go.Bar(
                x=df[x_col],
                y=values,
                name=label,
                marker_color=color,
                hoverlabel=dict(font=PLOTLY_FONT),
            ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel if not horizontal else ylabel,
        yaxis_title=ylabel if not horizontal else xlabel,
        barmode="stack",
        showlegend=True,
    )
    return _apply_defaults(fig, title)


# ============================================================
# V5: 分组柱状图（男女对比等）
# ============================================================

def plot_grouped_bar(
    categories: List[str],
    values_dict: Dict[str, List[float]],
    title: str = "分组对比",
    xlabel: str = "",
    ylabel: str = "分数",
) -> go.Figure:
    """绘制分组柱状图。

    Args:
        categories: x 轴分类标签
        values_dict: {"男生": [85, 78, ...], "女生": [82, 83, ...]}
        title: 标题
    """
    fig = go.Figure()

    for i, (name, values) in enumerate(values_dict.items()):
        fig.add_trace(go.Bar(
            x=categories,
            y=values,
            name=name,
            marker_color=COLOR_PALETTE[i % len(COLOR_PALETTE)],
            hoverlabel=dict(font=PLOTLY_FONT),
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
            textfont=dict(size=10),
        ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        barmode="group",
        showlegend=True,
    )
    return _apply_defaults(fig, title)


# ============================================================
# V6: 横向柱状图（分数段金字塔等）
# ============================================================

def plot_horizontal_bar(
    labels: List[str],
    values: List[float],
    title: str = "",
    xlabel: str = "人数",
    color: str = None,
) -> go.Figure:
    """绘制横向柱状图。

    Args:
        labels: y 轴标签
        values: 柱值
        title: 标题
        xlabel: x 轴标签
        color: 柱子颜色
    """
    color = color or COLOR_PALETTE[0]

    fig = go.Figure(data=go.Bar(
        y=labels,
        x=values,
        orientation="h",
        marker_color=color,
        text=[str(v) for v in values],
        textposition="outside",
        hoverlabel=dict(font=PLOTLY_FONT),
    ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis=dict(autorange="reversed"),  # 从上到下
    )
    return _apply_defaults(fig, title)


# ============================================================
# V7: 雷达图
# ============================================================

def plot_radar(
    categories: List[str],
    values: List[float],
    title: str = "学科雷达图",
    fill_color: str = "rgba(66, 133, 244, 0.3)",
    line_color: str = "rgba(66, 133, 244, 1)",
    reference_values: Optional[List[float]] = None,
    reference_label: str = "全校均分",
) -> go.Figure:
    """绘制雷达图。

    Args:
        categories: 各维度名称
        values: 各维度值
        title: 标题
        fill_color, line_color: 填充和线条颜色
        reference_values: 参考线值（全校均分）
        reference_label: 参考线标签
    """
    if not values or not categories:
        fig = go.Figure()
        fig.update_layout(title=title, annotations=[dict(text="无数据", showarrow=False)])
        return _apply_defaults(fig, title)

    fig = go.Figure()

    # 主数据
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],  # 闭合
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color, width=2),
        name="该生",
        hoverlabel=dict(font=PLOTLY_FONT),
    ))

    # 参考线
    if reference_values:
        fig.add_trace(go.Scatterpolar(
            r=reference_values + [reference_values[0]],
            theta=categories + [categories[0]],
            fill="none",
            line=dict(color="gray", width=1.5, dash="dash"),
            name=reference_label,
            hoverlabel=dict(font=PLOTLY_FONT),
        ))

    # 固定范围：得分率用 0-100，z-score 用对称范围
    all_vals = values + (reference_values or [])
    v_min = min(all_vals)
    v_max = max(all_vals)

    if v_min >= 0 and v_max <= 100:
        # 得分率模式：0-100
        r_min, r_max = 0, 100
        dtick = 20
    else:
        # z-score 模式：对称范围，以 0 为中心
        abs_max = max(abs(v_min), abs(v_max), 1.0)
        r_max = abs_max * 1.3
        r_min = -r_max
        dtick = round(r_max / 4, 1) if r_max >= 2 else 0.5

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[r_min, r_max],
                tickmode="linear",
                tick0=0 if v_min < 0 else 0,
                dtick=dtick,
                tickfont=PLOTLY_FONT,
                gridcolor="rgba(0,0,0,0.1)",
            ),
            angularaxis=dict(
                tickfont=PLOTLY_FONT,
                gridcolor="rgba(0,0,0,0.15)",
            ),
            bgcolor="rgba(0,0,0,0.02)",
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, font=PLOTLY_FONT),
    )
    return _apply_defaults(fig, title)


# ============================================================
# V8: 水平条形图（个人学科百分位等）
# ============================================================

def plot_horizontal_percentile(
    categories: List[str],
    values: List[float],
    title: str = "学科百分位",
    color_threshold: float = 0.5,
) -> go.Figure:
    """绘制带颜色标识的水平百分位条形图。

    Args:
        categories: 学科名称
        values: 百分位 (0~1, 越大越好)
        title: 标题
        color_threshold: 低于此值标记为红色（薄弱）
    """
    colors = []
    for v in values:
        if v >= 0.75:
            colors.append(COLOR_PALETTE[0])  # 绿色 - 优势
        elif v >= 0.50:
            colors.append(COLOR_PALETTE[4])  # 黄色 - 一般
        else:
            colors.append("tomato")  # 红色 - 薄弱

    fig = go.Figure(data=go.Bar(
        y=categories,
        x=[v * 100 for v in values],
        orientation="h",
        marker_color=colors,
        text=[f"{v * 100:.0f}%" for v in values],
        textposition="outside",
        hoverlabel=dict(font=PLOTLY_FONT),
    ))

    fig.update_layout(
        title=title,
        xaxis_title="超越全校 %",
        yaxis=dict(autorange="reversed"),
        xaxis=dict(range=[0, 105]),
    )
    return _apply_defaults(fig, title)


# ============================================================
# V9: PCA 散点图
# ============================================================

def plot_scatter(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    hover_texts: Optional[List[str]] = None,
    title: str = "PCA 聚类可视化",
    xlabel: str = "PC1",
    ylabel: str = "PC2",
    cluster_names: Optional[Dict[int, str]] = None,
) -> go.Figure:
    """绘制 PCA 散点图（按聚类着色）。

    Args:
        x, y: 坐标数据
        labels: 聚类标签数组
        hover_texts: 鼠标悬停文本
        title: 标题
        xlabel, ylabel: 轴标签
        cluster_names: 聚类名称映射 {0: "全能型", 1: "理科优势型", ...}
    """
    unique_labels = sorted(set(labels))

    fig = go.Figure()

    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = cluster_names.get(label, f"聚类 {label}") if cluster_names else f"聚类 {label}"
        color = COLOR_PALETTE[i % len(COLOR_PALETTE)]

        hover = None
        if hover_texts:
            hover = [hover_texts[j] for j in range(len(hover_texts)) if mask[j]]

        fig.add_trace(go.Scatter(
            x=x[mask],
            y=y[mask],
            mode="markers",
            name=name,
            marker=dict(
                color=color,
                size=8,
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            text=hover,
            hoverinfo="text+x+y" if hover else "x+y",
            hoverlabel=dict(font=PLOTLY_FONT),
        ))

    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        showlegend=True,
    )
    return _apply_defaults(fig, title)


# ============================================================
# V10: 双线图（肘部法则 + 轮廓系数）
# ============================================================

def plot_dual_line(
    x: List[int],
    y1: List[float],
    y1_name: str = "WCSS",
    y2: List[float] = None,
    y2_name: str = "轮廓系数",
    title: str = "最优 K 值选择",
) -> go.Figure:
    """绘制双 y 轴折线图（用于肘部法则 + 轮廓系数）。

    Args:
        x: x 轴数据 (K 值)
        y1: 左 y 轴数据
        y1_name: 左 y 轴标签
        y2: 右 y 轴数据
        y2_name: 右 y 轴标签
        title: 标题
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=x, y=y1, mode="lines+markers",
            name=y1_name, line=dict(color=COLOR_PALETTE[0], width=2.5),
            marker=dict(size=8),
            hoverlabel=dict(font=PLOTLY_FONT),
        ),
        secondary_y=False,
    )

    if y2 is not None:
        fig.add_trace(
            go.Scatter(
                x=x, y=y2, mode="lines+markers",
                name=y2_name, line=dict(color=COLOR_PALETTE[1], width=2.5, dash="dot"),
                marker=dict(size=8),
                hoverlabel=dict(font=PLOTLY_FONT),
            ),
            secondary_y=True,
        )

    fig.update_layout(title=title)
    fig.update_xaxes(title_text="K (聚类数)")
    fig.update_yaxes(title_text=y1_name, secondary_y=False)
    if y2 is not None:
        fig.update_yaxes(title_text=y2_name, secondary_y=True)

    return _apply_defaults(fig, title)
