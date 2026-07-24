# 中考成绩分析系统 v1.0

> 教育数据分析工具，面向学校教务/年级组，支持全校中考成绩批量导入 → 多维度分析 → 交互式报告 + 个人 PDF 画像批量生成

---

## 快速开始

### 环境要求

- Python 3.9+
- macOS / Windows / Linux
- (macOS 可选) `brew install pango gobject-introspection` — 用于 PDF 生成

### 启动

```bash
# 方式一：使用启动脚本（推荐）
./start.sh

# 方式二：手动启动
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

浏览器打开 `http://localhost:8501`

### 局域网共享

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
# 同事访问: http://你的IP:8501
```

---

## 功能

| 模块 | 功能 |
|------|------|
| 📤 **数据导入** | 上传 Excel (.xlsx/.xls)，自动识别字段映射，清洗校验数据，满分配置 |
| 📊 **全校宏观分析** | 总分分布（直方图+KDE）、各科箱线图对比、学科相关性热力图、男女生 t 检验、ABCD 等级分布、分数段金字塔 |
| 👤 **个人画像** | 学生搜索/选择、11 维雷达图、偏科指数诊断、优势/薄弱学科识别、文理倾向分析、各科百分位 |
| 🧩 **学生聚类** | K-Means 聚类（K=3~8 可调）、肘部法则+轮廓系数、PCA 2D 可视化、聚类画像雷达图 |
| 🎯 **学业规划** | 提分潜力排序、新高考 3+1+2 选科建议（3 套方案）、个性化学习策略 + 时间分配 |
| 📄 **报告导出** | 交互式 HTML 报告下载、批量 PDF 个人报告生成（ZIP 打包） |

---

## 数据安全

- **纯本地处理**：所有数据仅在本地计算，不上传任何云端
- **无需登录**：单机或局域网部署，直接使用
- **无数据库**：数据仅存在于内存和上传的 Excel 文件中

---

## 项目结构

```
├── app.py                     # Streamlit 主入口
├── start.sh                   # 启动脚本
├── requirements.txt           # Python 依赖
├── src/
│   ├── data_loader.py         # 数据加载、列映射、清洗、校验
│   ├── macro_analysis.py      # 全校宏观分析
│   ├── student_profile.py     # 个人画像（偏科指数、雷达图、优劣势）
│   ├── clustering.py          # K-Means 聚类 + PCA
│   ├── academic_planning.py   # 学业规划（提分、选科、策略）
│   ├── visualizations.py      # Plotly 图表引擎
│   ├── pdf_generator.py       # PDF 报告生成（Jinja2 + WeasyPrint）
│   └── utils.py               # 工具函数（中文字体、等级评定）
├── templates/
│   ├── student_report.html    # PDF 报告 Jinja2 模板
│   └── report_style.css       # PDF 报告样式
├── data/                      # 示例数据目录
└── output/                    # 导出目录
```

---

## 各科满分配置

系统默认满分（可在数据导入页面修改）：

| 学科 | 满分 |
|------|------|
| 语文 | 120 |
| 数学 | 120 |
| 英语 | 120 |
| 物理 | 100 |
| 化学 | 100 |
| 生物 | 100 |
| 历史 | 100 |
| 地理 | 100 |
| 道法 | 100 |
| 体育 | 60 |

---

## Excel 数据格式

系统自动识别以下字段（支持模糊匹配）：

- **身份信息**：姓名、考生号、身份证号、联系电话等
- **人口学信息**：性别、民族、毕业中学、户籍类型等
- **成绩数据**：总分、总分名次、语文、数学、英语、物理、化学、生物、历史、地理、道法、体育

---

## 技术栈

| 层次 | 技术 |
|------|------|
| Web 框架 | Streamlit |
| 数据处理 | pandas, numpy |
| 可视化 | Plotly |
| 机器学习 | scikit-learn (K-Means, PCA, StandardScaler) |
| 统计检验 | scipy (t-test, Shapiro-Wilk) |
| PDF 生成 | WeasyPrint + Jinja2 |
| Excel 读取 | openpyxl |
