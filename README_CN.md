# 英国人口老龄化趋势预测平台

[English](README.md) | [中文](README_CN.md)

基于 **Python + React** 的全栈数据分析与预测平台，使用英国国家统计局（ONS）官方数据，对英国各地区 65 岁及以上人口比例进行长期预测与可视化交互分析。

| 层 | 技术栈 |
|---|---|
| **后端** | Python, FastAPI, Prophet, ARIMA, scikit-learn |
| **前端** | React 18, Vite, Recharts, Tailwind CSS |
| **数据** | ONS 官方 Excel / XLS |

---

## 项目结构

```text
forecasting-uk-ageing-trends/
├── backend/                         # FastAPI 后端
│   ├── app/
│   │   ├── main.py                  # 应用入口，CORS 与 OpenAPI 配置
│   │   ├── routers/
│   │   │   ├── data.py              # /api/ageing-ratio, /regions, /overview
│   │   │   ├── forecast.py          # /api/forecast/prophet, /forecast/arima, /metrics
│   │   │   └── cluster.py           # /api/cluster
│   │   └── services/
│   │       ├── data_service.py      # 读取历史老龄化比例数据
│   │       ├── forecast_service.py  # 读取 Prophet / ARIMA 预测结果
│   │       └── cluster_service.py   # KMeans 聚类分析
│   └── requirements.txt
├── frontend/                        # React 前端
│   ├── src/
│   │   ├── App.jsx                  # React Router 配置
│   │   ├── api/client.js            # Axios API 封装
│   │   ├── components/Navbar.jsx    # 顶部导航栏
│   │   └── pages/
│   │       ├── Dashboard.jsx        # 历史趋势总览与统计卡片
│   │       ├── Forecast.jsx         # Prophet / ARIMA 交互式预测视图
│   │       └── Cluster.jsx          # 地区聚类分析
│   ├── vite.config.js               # Vite 代理配置
│   └── package.json
├── src/                             # Python 分析模块
│   ├── preprocess*.py               # 各地区数据清洗脚本
│   ├── merge_projection_data.py     # 历史数据与投影数据融合
│   ├── model_prophet.py             # Prophet 预测模型
│   ├── model_arima.py               # ARIMA 模型与多模型对比
│   ├── cluster_analysis.py          # KMeans 聚类
│   └── plot_*.py                    # 离线可视化脚本
├── data/
│   ├── raw/                         # ONS 原始 Excel / XLS 文件
│   └── processed/                   # 清洗后的 CSV 文件
├── output/                          # 预测结果与生成图表
│   └── multi_compare/               # 多模型对比结果
├── main.py                          # 离线批处理入口
└── requirements.txt                 # Python 依赖
```

---

## 项目概述

| 项目 | 说明 |
|---|---|
| **研究对象** | 英格兰、威尔士、苏格兰 65 岁及以上人口比例变化 |
| **预测跨度** | 2020-2150 |
| **预测方法** | Prophet, ARIMA, KMeans 聚类 |
| **数据来源** | 英国国家统计局（ONS）官方数据 |

---

## 核心功能

### 1. 数据预处理 (`src/preprocess*.py`)
- 从 ONS Excel / XLS 原始文件中提取人口数据。
- 按年龄、年份、地区进行清洗与整形。
- 支持英格兰、威尔士、苏格兰和英国整体数据处理。

### 2. 数据融合 (`src/merge_projection_data.py`)
- 合并历史观测数据与官方人口投影数据。
- 计算各地区 65+ 人口占比，也就是老龄化比例。

### 3. 时间序列预测
- **Prophet**：趋势建模、变化点检测与预测曲线平滑。
- **ARIMA**：自动参数搜索，并基于 AIC 准则选优。

### 4. 聚类分析 (`src/cluster_analysis.py`)
- 使用 KMeans 识别老龄化趋势相似的地区。
- 对时序特征进行标准化，保证聚类比较更公平。

### 5. 交互式 Web 平台
- **Dashboard**：历史趋势折线图与地区统计卡片。
- **Forecast**：支持地区和模型切换，并展示评估指标。
- **Cluster**：支持调整聚类数量，展示地区分组与趋势曲线。

---

## 快速开始

### 前置要求

- Python 3.8+，推荐使用 conda 环境。
- Node.js 18+。

### 1. 生成分析数据

```bash
pip install -r requirements.txt
python main.py
```

### 2. 启动 FastAPI 后端

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

API 文档：http://localhost:8000/docs

### 3. 启动 React 前端

```bash
cd frontend
npm install
npm run dev
```

浏览器访问：http://localhost:5173。

### 页面一览

| 路径 | 页面 | 说明 |
|---|---|---|
| `/dashboard` | Dashboard | 历史老龄化趋势总览与统计卡片 |
| `/forecast` | Forecast | Prophet / ARIMA 交互式预测图与指标表 |
| `/cluster` | Cluster | KMeans 聚类分析与地区分组展示 |

---

## API 端点

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/ageing-ratio` | 各地区历史老龄化比例数据 |
| GET | `/api/regions` | 可用地区列表 |
| GET | `/api/overview` | 各地区统计摘要 |
| GET | `/api/forecast/prophet?region=England` | Prophet 历史 + 预测序列 |
| GET | `/api/forecast/arima?region=England` | ARIMA 历史 + 预测序列 |
| GET | `/api/metrics` | Prophet vs ARIMA 评估指标：MAE、RMSE、MAPE |
| GET | `/api/cluster?n_clusters=3` | KMeans 聚类结果与趋势数据 |

---

## 技术栈

### 后端

| 技术 | 用途 |
|---|---|
| **Python 3.8+** | 核心开发语言 |
| **FastAPI** | REST API 框架 |
| **uvicorn** | ASGI 服务器 |
| **Pandas / NumPy** | 数据处理与数值计算 |
| **Prophet** | 时间序列预测 |
| **pmdarima / Statsmodels** | ARIMA 建模与统计诊断 |
| **scikit-learn** | KMeans 聚类与数据预处理 |

### 前端

| 技术 | 用途 |
|---|---|
| **React 18** | 组件化 UI |
| **Vite** | 前端构建工具与开发服务器 |
| **Recharts** | React 交互式图表 |
| **Tailwind CSS** | 实用优先的样式系统 |
| **React Router v6** | 客户端路由 |
| **Axios** | HTTP 请求客户端 |

---

## 离线批处理配置

可在 `main.py` 中调整以下参数：

```python
CONFIG = {
    "regions": ["England", "Wales", "Scotland"],
    "end_year": 2070,
    "test_year_start": 2030,
    "horizon": 30,
    "n_clusters": 3,
    "random_state": 42
}
```

---

## 模型说明

### Prophet
- 分段趋势建模与变化点检测。
- 对长期预测曲线进行平滑，提高可读性。

### ARIMA
- 通过 `pmdarima` 自动搜索 `(p, d, q)` 参数。
- 使用 AIC 准则选择适合平稳或差分平稳序列的模型。

### KMeans
- 对各地区老龄化时序轨迹进行标准化后聚类。
- 按相似的长期老龄化模式对地区进行分组。

---

## 项目特点

- **全栈架构**：FastAPI 后端 + React 前端。
- **官方数据流程**：ONS 原始数据、清洗结果和预测输出完整保留。
- **多模型对比**：使用 MAE、RMSE、MAPE 对比 Prophet 与 ARIMA。
- **交互式可视化**：支持地区、模型和聚类数量切换。
- **端到端 ETL**：从原始数据到清洗数据、预测结果和图表输出。
- **可复现分析**：固定随机种子，并保留处理后的输出文件。

---

## 待优化方向

### 1. 工程化与可复现性
- 增加 `Makefile` 或任务脚本，统一数据生成、后端启动、前端启动和测试命令。
- 增加 `.env.example`，将路径、端口、API 地址等配置从代码中抽离。
- 增加 Docker 或 `docker-compose`，实现一条命令启动完整应用。
- 明确区分源数据、处理后数据和生成输出，降低复现实验成本。

### 2. 后端 API 质量
- 缓存 CSV 读取结果，避免每次请求重复加载同一份数据。
- 校验 `region` 参数，对不支持的地区返回清晰的 400 / 404 错误。
- 使用 Pydantic response models 描述接口返回结构。
- 增强数据文件缺失、列名变化、空数据等异常处理。
- 增加 `/health` 接口，返回后端服务和数据文件可用状态。

### 3. 预测与分析可信度
- 在模型支持时加入预测置信区间。
- 增加 naive 或 linear trend 等 baseline 模型，便于证明 Prophet / ARIMA 的提升。
- 在指标表中突出每个地区表现更好的模型。
- 补充长期预测假设说明，尤其是预测区间超过官方投影范围时。

### 4. 前端体验
- 增加关键洞察卡片，例如老龄化最快地区、最新 65+ 占比最高地区、历史变化最大地区。
- 支持在同一张图中对比 Prophet 和 ARIMA。
- 增强空状态、加载状态、错误状态和重试交互。
- 优化移动端图表和指标表展示。
- 增加简短解读文案，让图表直接传达结论。

### 5. 测试与质量门禁
- 增加 `pytest`，覆盖后端 service 和 API endpoint。
- 增加前端 smoke test 或主要页面组件测试。
- 引入 `ruff`、ESLint、Prettier 等格式化和 lint 工具。
- 增加 GitHub Actions 或其他 CI 流程，自动运行测试和前端构建。

### 6. 作品集展示与部署
- 在 README 中加入页面截图或简短演示 GIF。
- 增加前端和后端部署方案。
- 补充架构图和数据流图。
- 扩展项目难点、技术决策和取舍说明，方便面试讲解。

---

## 许可证

此项目仅供学术和研究使用。
