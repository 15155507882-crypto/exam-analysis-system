"""任务管理器 — 数据持久化。

存储结构：
    data/tasks/
    ├── tasks.json              # 任务索引
    └── {task_id}/
        ├── data.parquet         # 清洗后的 DataFrame
        └── config.json          # 配置和摘要

所有数据纯本地存储，不上传云端。
"""

import json
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd

# 任务根目录
TASKS_DIR = Path(__file__).parent.parent / "data" / "tasks"


def _ensure_dir():
    """确保任务目录存在。"""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)


def _index_path() -> Path:
    """任务索引文件路径。"""
    return TASKS_DIR / "tasks.json"


def _load_index() -> List[Dict[str, Any]]:
    """加载任务索引。"""
    _ensure_dir()
    path = _index_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def _save_index(tasks: List[Dict[str, Any]]):
    """保存任务索引。"""
    _ensure_dir()
    with open(_index_path(), "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)


def _task_dir(task_id: str) -> Path:
    """任务子目录路径。"""
    return TASKS_DIR / task_id


# ============================================================
# 公开 API
# ============================================================

def list_tasks() -> List[Dict[str, Any]]:
    """列出所有任务，按创建时间倒序。

    Returns:
        [{id, name, created, student_count, total_max}, ...]
    """
    return sorted(_load_index(), key=lambda t: t.get("created", ""), reverse=True)


def create_task(name: str) -> str:
    """创建新任务。

    Args:
        name: 任务名称（如 "2026年中考"）

    Returns:
        task_id: 新任务 ID
    """
    task_id = uuid.uuid4().hex[:12]

    tasks = _load_index()
    tasks.append({
        "id": task_id,
        "name": name.strip(),
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "student_count": 0,
        "total_max": 0,
    })
    _save_index(tasks)

    # 创建任务子目录
    _task_dir(task_id).mkdir(parents=True, exist_ok=True)

    return task_id


def save_task(
    task_id: str,
    df: pd.DataFrame,
    max_scores: Dict[str, float],
    summary: Optional[Dict[str, Any]] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    validation_issues: Optional[List[Dict]] = None,
):
    """保存任务数据到磁盘。

    Args:
        task_id: 任务 ID
        df: 清洗后的 DataFrame
        max_scores: 满分配置
        summary: 数据摘要
        column_mapping: 列映射
        validation_issues: 校验问题
    """
    task_dir = _task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)

    # 保存 DataFrame（Parquet 格式，快速且保留类型）
    df.to_parquet(task_dir / "data.parquet", index=False)

    # 保存配置
    config = {
        "max_scores": max_scores,
        "summary": summary,
        "student_count": len(df),
        "total_max": sum(max_scores.values()),
    }
    if column_mapping:
        config["column_mapping"] = column_mapping
    if validation_issues:
        config["validation_issues"] = validation_issues

    with open(task_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # 更新索引
    tasks = _load_index()
    for t in tasks:
        if t["id"] == task_id:
            t["student_count"] = len(df)
            t["total_max"] = sum(max_scores.values())
            t["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            break
    _save_index(tasks)


def load_task(task_id: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
    """从磁盘加载任务数据。

    Returns:
        (df, config) — df 可能为 None（数据文件不存在）
    """
    task_dir = _task_dir(task_id)
    data_path = task_dir / "data.parquet"
    config_path = task_dir / "config.json"

    df = None
    config = None

    if data_path.exists():
        try:
            df = pd.read_parquet(data_path)
        except Exception:
            df = None

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except (json.JSONDecodeError, IOError):
            config = None

    return df, config


def delete_task(task_id: str):
    """删除任务及其数据。"""
    task_dir = _task_dir(task_id)
    if task_dir.exists():
        shutil.rmtree(task_dir)

    tasks = _load_index()
    tasks = [t for t in tasks if t["id"] != task_id]
    _save_index(tasks)


def rename_task(task_id: str, new_name: str):
    """重命名任务。"""
    tasks = _load_index()
    for t in tasks:
        if t["id"] == task_id:
            t["name"] = new_name.strip()
            break
    _save_index(tasks)


def get_task_info(task_id: str) -> Optional[Dict[str, Any]]:
    """获取单个任务信息。"""
    for t in _load_index():
        if t["id"] == task_id:
            return t
    return None
