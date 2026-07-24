"""工具函数：中文字体检测、等级评定、百分比格式化、分数归一化"""

import platform
import matplotlib
import matplotlib.font_manager as fm
from typing import Dict, Optional, Tuple

# ============================================================
# 中文字体配置
# ============================================================

# 跨平台中文字体候选列表（优先级从高到低）
_CHINESE_FONT_CANDIDATES: Dict[str, list] = {
    "Darwin": ["PingFang SC", "Heiti SC", "STHeiti", "Arial Unicode MS"],
    "Windows": ["Microsoft YaHei", "SimHei", "KaiTi", "FangSong"],
    "Linux": [
        "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
        "Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans",
    ],
}

# 缓存检测结果
_cached_chinese_font: Optional[str] = None


def detect_chinese_font() -> str:
    """跨平台检测可用中文字体，返回字体名称。

    检测顺序：
    1. 按当前平台候选列表依次尝试
    2. 遍历系统所有已安装字体，找支持中文的
    3. 回退到 sans-serif

    Returns:
        str: 可用的中文字体名称
    """
    global _cached_chinese_font
    if _cached_chinese_font is not None:
        return _cached_chinese_font

    system = platform.system()
    candidates = _CHINESE_FONT_CANDIDATES.get(system, [])

    # 获取系统中所有字体名称（小写集合用于快速查找）
    available_fonts = {f.name for f in fm.fontManager.ttflist}

    # 按候选列表匹配
    for font_name in candidates:
        if font_name in available_fonts:
            _cached_chinese_font = font_name
            return font_name

    # 候选都未匹配，遍历所有字体找含中文名称的
    for f in fm.fontManager.ttflist:
        name = f.name
        if any(ord(c) > 0x4E00 for c in name):
            _cached_chinese_font = name
            return name

    # 最终回退
    _cached_chinese_font = "sans-serif"
    return _cached_chinese_font


def get_matplotlib_font_properties():
    """获取 matplotlib 中文字体属性对象。"""
    font_name = detect_chinese_font()
    return fm.FontProperties(fname=font_name) if font_name != "sans-serif" else None


def set_chinese_font():
    """配置 matplotlib 全局中文字体（用于静态图表）。"""
    font_name = detect_chinese_font()
    if font_name != "sans-serif":
        matplotlib.rcParams["font.family"] = font_name
        matplotlib.rcParams["axes.unicode_minus"] = False


def get_plotly_font_family() -> str:
    """返回 Plotly 图表可用的中文字体族名。"""
    font_name = detect_chinese_font()
    # Plotly 需要 font family，回退值
    return font_name if font_name != "sans-serif" else "Arial"


# ============================================================
# 等级评定
# ============================================================

def grade_level(score_rate: float) -> str:
    """根据得分率评定等级。

    Args:
        score_rate: 得分率 (0.0 ~ 1.0+)

    Returns:
        等级标签: A / B / C / D
    """
    if score_rate >= 0.85:
        return "A"
    elif score_rate >= 0.70:
        return "B"
    elif score_rate >= 0.60:
        return "C"
    else:
        return "D"


GRADE_LABELS = {
    "A": "优秀",
    "B": "良好",
    "C": "及格",
    "D": "不及格",
}


def grade_label(level: str) -> str:
    """等级代码 → 中文标签。"""
    return GRADE_LABELS.get(level, "未知")


# ============================================================
# 格式化函数
# ============================================================

def format_percent(value: float, decimals: int = 1) -> str:
    """格式化为百分比字符串。

    Args:
        value: 0.0~1.0 的比例值
        decimals: 小数位数

    Returns:
        "85.0%" 格式字符串
    """
    return f"{value * 100:.{decimals}f}%"


def format_rank(rank: int, total: int) -> str:
    """格式化为 "排名/总人数 (前 X%)"。"""
    if total == 0:
        return f"{rank}/0"
    pct = rank / total * 100
    return f"{rank}/{total} (前 {pct:.1f}%)"


# ============================================================
# 分数归一化
# ============================================================

def normalize_scores(scores, max_scores: Dict[str, float]) -> Dict[str, float]:
    """将原始分数归一化为百分制得分率。

    Args:
        scores: {"chinese": 105, "math": 112, ...}
        max_scores: {"chinese": 120, "math": 120, ...}

    Returns:
        {"chinese": 0.875, "math": 0.933, ...}
    """
    result = {}
    for subject, score in scores.items():
        max_val = max_scores.get(subject)
        if max_val and max_val > 0 and score is not None:
            result[subject] = score / max_val
        else:
            result[subject] = None
    return result


# ============================================================
# 学科常量（从数据中动态获取，这里仅作为后备）
# ============================================================

SUBJECT_GROUPS = {
    "主科": ["chinese", "math", "english"],
    "理科": ["physics", "chemistry", "biology"],
    "文科": ["history", "geography", "morality_law"],
    "体育": ["pe"],
}

ALL_SUBJECTS = [
    "chinese", "math", "english",
    "physics", "chemistry", "biology",
    "history", "geography", "morality_law",
    "pe",
]

SUBJECT_LABELS = {
    "chinese": "语文",
    "math": "数学",
    "english": "英语",
    "physics": "物理",
    "chemistry": "化学",
    "biology": "生物",
    "history": "历史",
    "geography": "地理",
    "morality_law": "道法",
    "pe": "体育",
    "total_score": "总分",
}


def get_subject_label(key: str) -> str:
    """获取学科中文标签。"""
    return SUBJECT_LABELS.get(key, key)
