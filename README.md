# 英国人口老龄化趋势预测平台

基于 **Python + React** 的全栈数据分析与预测平台，使用英国国家统计局（ONS）官方数据，对英国各地区 65 岁及以上人口比例进行长期预测与可视化交互分析。

| 层 | 技术栈 |
|----|--------|
| **后端** | Python · FastAPI · Prophet · ARIMA · scikit-learn |
| **前端** | React 18 · Vite · Recharts · Tailwind CSS |
| **数据** | ONS 官方 Excel / XLS |

---

## 🗂️ 项目结构

```
forecasting-uk-ageing-trends/
├── backend/                         # FastAPI 后端
│   ├── app/
│   │   ├── main.py                  # 应用入口，CORS 配置
│   │   ├── routers/
│   │   │   ├── data.py              # /api/ageing-ratio · /regions · /overview
│   │   │   ├── forecast.py          # /api/forecast/prophet · /forecast/arima · /metrics
│   │   │   └── cluster.py           # /api/cluster
│   │   └── services/
│   │       ├── data_service.py      # 读取历史老龄化比例数据
│   │       ├── forecast_service.py  # 读取 Prophet / ARIMA 预测结果
│   │       └── cluster_service.py   # KMeans 聚类分析
│   └── requirements.txt
├── frontend/                        # React 前端
│   ├── src/
│   │   ├── App.jsx                  # 路由配置（React Router v6）
│   │   ├── api/client.js            # Axios API 封装
│   │   ├── components/Navbar.jsx    # 顶部导航栏
│   │   └── pages/
│   │       ├── Dashboard.jsx        # 历史趋势总览 + 统计卡片
│   │       ├── Forecast.jsx         # 交互式预测图（Prophet / ARIMA）
│   │       └── Cluster.jsx          # 地区聚类分析
│   ├── vite.config.js               # Vite + 反向代理配置
│   └── package.json
├── src/                             # Python 分析模块（被后端 services 调用）
│   ├── preprocess*.py               # 各地区数据清洗
│   ├── merge_projection_data.py     # 历史 + 投影数据融合
│   ├── model_prophet.py             # Prophet 预测模型
│   ├── model_arima.py               # ARIMA 模型 & 多模型对比
│   ├── cluster_analysis.py          # KMeans 聚类
│   └── plot_*.py                    # 离线可视化脚本
├── data/
│   ├── raw/                         # ONS 原始 Excel / XLS 文件
│   └── processed/                   # 清洗后的 CSV 文件
├── output/                          # 预测结果 & 图表输出
│   └── multi_compare/               # 多模型对比结果
├── main.py                          # 离线批处理入口（保留）
└── requirements.txt                 # Python 依赖
```

---

## 📋 项目概述

| 项目 | 说明 |
|------|------|
| **研究对象** | 英格兰、威尔士、苏格兰 65 岁及以上人口比例变化 |
| **预测跨度** | 2020 – 2150 年 |
| **预测方法** | Prophet · ARIMA · KMeans 聚类 |
| **数据来源** | 英国国家统计局（ONS）官方数据 |

---

## 🎯 核心功能

### 1. 数据预处理 (`src/preprocess*.py`)
- 从 ONS Excel / XLS 原始文件中提取人口数据
- 按年龄、年份、地区进行清洗与整形（宽格式 → 长格式）
- 支持英格兰、威尔士、苏格兰、英国整体分别处理

### 2. 数据融合 (`src/merge_projection_data.py`)
- 合并历史观测数据（1991–2018）与官方投影数据（2018–2070）
- 计算各地区 65+ 人口占比（老龄化比例）

### 3. 时间序列预测
- **Prophet** — 逻辑增长约束 + 自适应变化点检测 + 滚动平均平滑
- **ARIMA** — 自动参数搜索（pmdarima）+ AIC 准则选优 + 多地区对比

### 4. 聚类分析 (`src/cluster_analysis.py`)
- KMeans 对各地区老龄化趋势进行无监督聚类（默认 3 类）
- StandardScaler 标准化确保公平对比

### 5. 交互式 Web 平台
- **Dashboard** — 历史趋势折线图 + 各地区统计卡片
- **Forecast** — 点击切换地区 / 模型，实时渲染预测曲线与评估指标
- **Cluster** — 可调聚类数，聚类结果分组卡片 + 趋势图

---

## 🚀 快速开始

### 前置要求

- Python 3.8+（推荐 conda 环境）
- Node.js 18+

### 1. 生成分析数据（离线批处理，仅首次需要）

```bash
pip install -r requirements.txt
python main.py
```

### 2. 启动后端（FastAPI）

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

> 交互式 API 文档：http://localhost:8000/docs

### 3. 启动前端（React + Vite）

```bash
cd frontend
npm install
npm run dev
```

> 浏览器访问：http://localhost:5173

### 页面一览

| 路径 | 页面 | 说明 |
|------|------|------|
| `/dashboard` | Dashboard | 历史老龄化趋势总览与统计卡片 |
| `/forecast` | Forecast | 交互式预测图（切换地区 / 模型）+ 评估指标表 |
| `/cluster` | Cluster | KMeans 聚类分析与地区分组可视化 |

---

## 📈 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/ageing-ratio` | 各地区历史老龄化比例数据 |
| GET | `/api/regions` | 可用地区列表 |
| GET | `/api/overview` | 各地区统计摘要（起止年份、变化量） |
| GET | `/api/forecast/prophet?region=England` | Prophet 历史 + 预测序列 |
| GET | `/api/forecast/arima?region=England` | ARIMA 历史 + 预测序列 |
| GET | `/api/metrics` | Prophet vs ARIMA 评估指标（MAE / RMSE / MAPE） |
| GET | `/api/cluster?n_clusters=3` | KMeans 聚类结果与趋势数据 |

---

## 🛠️ 技术栈

### 后端

| 技术 | 用途 |
|------|------|
| **Python 3.8+** | 核心开发语言 |
| **FastAPI** | 高性能 REST API 框架 |
| **uvicorn** | ASGI 服务器 |
| **Pandas / NumPy** | 数据处理与数值计算 |
| **Prophet** | 时间序列趋势预测 |
| **pmdarima / Statsmodels** | ARIMA 自动参数选择与统计诊断 |
| **scikit-learn** | KMeans 聚类 + 数据标准化 |

### 前端

| 技术 | 用途 |
|------|------|
| **React 18** | 组件化 UI 框架 |
| **Vite** | 前端构建工具，内置开发代理 |
| **Recharts** | 基于 React 的交互式图表库 |
| **Tailwind CSS** | 实用优先的 CSS 框架 |
| **React Router v6** | 客户端路由 |
| **Axios** | HTTP 请求客户端 |

---

## 📦 主要依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| fastapi | 0.110+ | REST API 框架 |
| uvicorn | 0.29+ | ASGI 服务器 |
| pandas | 2.2.2 | 数据处理 |
| numpy | 1.24.4 | 数值计算 |
| prophet | 1.1.7 | 时间序列预测 |
| pmdarima | 2.0.4 | ARIMA 自动参数选择 |
| statsmodels | 0.14.1 | 统计模型 |
| scikit-learn | 1.3.2 | 聚类与预处理 |
| react | 18.3+ | 前端 UI |
| recharts | 2.12+ | React 图表库 |
| tailwindcss | 3.4+ | CSS 框架 |

---

## 🔧 离线批处理配置

在 `main.py` 中可调整以下参数：

```python
CONFIG = {
    "regions": ["England", "Wales", "Scotland"],
    "end_year": 2070,        # 预测终点年份
    "test_year_start": 2030, # 测试集起始年份
    "horizon": 30,           # 预测时间跨度（年）
    "n_clusters": 3,         # 聚类数量
    "random_state": 42       # 随机种子
}
```

---

## 📝 模型说明

### Prophet
- 分段线性 / 逻辑增长趋势 + 季节性分解
- 自适应变化点检测捕捉趋势转折
- 滚动平均平滑预测曲线

### ARIMA
- pmdarima 自动搜索最优 (p, d, q) 参数
- AIC 准则选优，适合平稳或一阶差分平稳序列

### KMeans 聚类
- StandardScaler 归一化各地区时序特征
- 无监督聚类识别老龄化特征相似地区
- 支持动态调整聚类数（2–4）

---

## 💡 项目特点

✅ **全栈架构** — FastAPI 后端 + React 前端，前后端分离  
✅ **多源数据整合** — 融合 ONS 历史数据与官方投影数据  
✅ **多模型对比** — Prophet vs ARIMA，量化评估 MAE / RMSE / MAPE  
✅ **交互式可视化** — 实时切换地区、模型、聚类数  
✅ **完整 ETL 流程** — 从原始数据到预测结果的端到端工作流  
✅ **可重现性** — 固定随机种子，结果可复现  

---

## 📄 许可证

此项目仅供学术和研究使用。
