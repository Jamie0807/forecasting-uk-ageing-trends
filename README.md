# 英国人口老龄化趋势预测平台

基于 **Python + React** 的全栈数据分析与预测平台，使用英国国家统计局（ONS）数据，对英国各地区 65 岁及以上人口比例进行长期预测与可视化分析。

| 层 | 技术栈 |
|----|--------|
| **后端** | Python · FastAPI · Prophet · ARIMA · scikit-learn |
| **前端** | React 18 · Vite · Recharts · Tailwind CSS |
| **数据** | ONS 官方 Excel/XLS 数据 |

---

## 🗂️ 项目结构

```
forecasting-uk-ageing-trends/
├── backend/                         # FastAPI 后端
│   ├── app/
│   │   ├── main.py                  # FastAPI 应用入口，CORS 配置
│   │   ├── routers/
│   │   │   ├── data.py              # GET /api/ageing-ratio, /regions, /overview
│   │   │   ├── forecast.py          # GET /api/forecast/prophet, /forecast/arima, /metrics
│   │   │   └── cluster.py           # GET /api/cluster
│   │   └── services/
│   │       ├── data_service.py      # 读取历史老龄化比例数据
│   │       ├── forecast_service.py  # 读取 Prophet/ARIMA 预测结果
│   │       └── cluster_service.py   # KMeans 聚类分析
│   └── requirements.txt
├── frontend/                        # React 前端
│   ├── src/
│   │   ├── App.jsx                  # 路由配置
│   │   ├── api/client.js            # Axios API 封装
│   │   ├── components/Navbar.jsx
│   │   └── pages/
│   │       ├── Dashboard.jsx        # 历史趋势总览
│   │       ├── Forecast.jsx         # 交互式预测图（Prophet / ARIMA）
│   │       └── Cluster.jsx          # 地区聚类分析
│   ├── vite.config.js               # Vite + 代理配置
│   └── package.json
├── src/                             # 原始分析模块（被后端 services 调用）
├── data/                            # 数据目录
├── output/                          # 预测结果输出
└── main.py                          # 离线批处理入口（保留）
```

## 📋 项目概述

- **研究对象**：英国各地区（英格兰、威尔士、苏格兰）65 岁及以上人口的比例变化
- **预测时间跨度**：2020-2150 年
- **预测方法**：Prophet 时间序列预测、ARIMA 模型、KMeans 聚类分析
- **数据来源**：英国国家统计局（ONS）官方数据


## 🎯 核心功能

### 1. **数据预处理** (`src/preprocess*.py`)
- 从 Excel/XLS 格式的 ONS 原始数据中提取人口数据
- 按年龄、年份、地区等维度进行数据清洗和整形
- 支持英国各地区的分别处理：英格兰、威尔士、苏格兰、英国整体
- 输出长格式 CSV 便于后续分析

### 2. **数据融合** (`src/merge_projection_data.py`)
- 将历史观测数据和预测数据进行合并
- 计算各地区的老龄化比例（65+人口占比）
- 生成统一的数据集用于建模

### 3. **时间序列预测** 
- **Prophet 模型** (`src/model_prophet.py`)
  - 支持逻辑增长约束
  - 自适应变化点检测
  - 滚动平均平滑处理
  
- **ARIMA 模型** (`src/model_arima.py`)
  - 自动 ARIMA 参数选择
  - 与 Prophet 结果对比评估
  - 支持多地区对比分析

### 4. **聚类分析** (`src/cluster_analysis.py`)
- 对各地区老龄化趋势进行 KMeans 聚类（默认 3 类）
- 标准化预处理确保公平对比
- 可视化展示各地区聚类结果

### 5. **可视化** (`src/plot_*.py`)
- 65+ 人口比例趋势图
- 历史数据 + 预测数据对比
- 多模型预测结果对比（Prophet vs ARIMA）
- 地区聚类可视化
- 时间序列分解可视化

## 🗂️ 项目结构

```
forecasting-uk-ageing-trends/
├── main.py                          # 项目主入口，定义执行流程
├── requirements.txt                 # Python 依赖包列表
├── README.md                        # 项目说明文档
│
├── data/
│   ├── raw/                         # 原始数据（ONS Excel/XLS 文件）
│   │   ├── mid_year_population_estimates_uk.xlsx
│   │   ├── SNPP18dt2.xlsx
│   │   ├── enppvsumpop20.xls        # 英格兰投影数据
│   │   ├── scppvsumpop20.xls        # 苏格兰投影数据
│   │   ├── wappvsumpop20.xls        # 威尔士投影数据
│   │   └── ukppvsumpop20.xls        # 英国整体投影数据
│   └── processed/                   # 处理后的数据
│       ├── cleaned_population_long.csv          # 清洗后的历史人口数据
│       ├── projected_population_long.csv        # 投影人口数据
│       ├── england_clean.csv / scotland_clean.csv / wales_clean.csv
│       ├── uk_population_projection_all.csv     # 合并数据
│       ├── ageing_ratio_per_region.csv          # 各地区老龄化比例
│       └── ageing_cluster_input.csv             # 聚类分析输入数据
│
├── output/                          # 输出结果
│   ├── *.png                        # 各类可视化图表
│   ├── *.csv                        # 预测结果数据
│   └── multi_compare/               # 多模型对比结果
│
└── src/                             # 源代码
    ├── __init__.py
    ├── preprocess.py                # 英国整体数据清洗
    ├── preprocess_england.py        # 英格兰数据处理
    ├── preprocess_scotland.py       # 苏格兰数据处理
    ├── preprocess_wales.py          # 威尔士数据处理
    ├── preprocess_uk.py             # 英国数据处理
    ├── preprocess_projections.py    # 投影数据处理
    ├── merge_projection_data.py     # 数据融合
    ├── model_prophet.py             # Prophet 时间序列模型
    ├── model_arima.py               # ARIMA 模型 & 多模型对比
    ├── arima_model.py               # ARIMA 基础函数
    ├── plot_ageing.py               # 老龄化趋势可视化
    ├── plot_forecast_england.py     # 英格兰预测结果可视化
    ├── plot_comparison.py           # 模型对比可视化
    ├── forecast_export.py           # 预测结果导出
    ├── generate_england_timeseries.py  # 生成英格兰时间序列
    ├── generate_cluster_input.py    # 生成聚类输入数据
    ├── cluster_analysis.py          # 聚类分析
    └── multi_region_compare.py      # 多地区 Prophet vs ARIMA 对比
```

## � 数据处理流程详解

本项目采用系统化的 ETL（Extract-Transform-Load）流程进行数据处理。以下详细说明整个数据处理的各个环节：

### 1. 数据采集与提取（Extract）

#### 源数据获取
项目使用来自英国国家统计局（ONS）的官方数据，包括两类数据源：

| 数据类型 | 文件名 | 描述 |
|--------|--------|------|
| **历史人口数据** | `mid_year_population_estimates_uk.xlsx` | 英国各地区 1991-2018 年的人口观测数据，按年龄和性别分组 |
| **人口投影数据** | `SNPP18dt2.xlsx` | 英国 2018 年发布的官方人口预测数据（2018-2043） |
| **地区投影数据** | `*vsumpop20.xls` 系列 | 各地区详细投影数据（英格兰、苏格兰、威尔士、英国整体） |

#### 数据提取方式
```python
# 使用 openpyxl 和 pandas 读取 Excel/XLS 文件
import pandas as pd

# 示例：读取历史人口数据
df_history = pd.read_excel('data/raw/mid_year_population_estimates_uk.xlsx', sheet_name='Population')

# 示例：读取投影数据
df_projection = pd.read_excel('data/raw/SNPP18dt2.xlsx', sheet_name='Projections')
```

### 2. 数据清洗与转换（Transform）

#### 步骤 2.1：缺失值处理
```
原始数据 → 识别缺失值 → 分析缺失原因 → 填充或删除
```
- 检测 `NaN` 值和空字符串
- 对于年份连续的人口数据，使用前向填充或线性插值
- 对于投影数据的缺失值，根据趋势进行合理填充
- 记录清洗日志便于后续审计

**示例代码**：
```python
# 检测缺失值
missing_data = df.isnull().sum()
print(f"缺失值统计：\n{missing_data}")

# 填充缺失值
df_filled = df.fillna(method='ffill')  # 前向填充
df_interpolated = df.interpolate(method='linear')  # 线性插值
```

#### 步骤 2.2：数据类型转换
- 将年份字段转换为 `int64` 类型
- 将人口数据转换为 `int64` 或 `float64`
- 统一地区名称的数据类型和格式

**示例代码**：
```python
df['Year'] = df['Year'].astype('int64')
df['Population'] = df['Population'].astype('int64')
df['Region'] = df['Region'].astype('str').str.strip()
```

#### 步骤 2.3：数据去重
- 检测完全重复的行
- 检测基于(年份、地区、年龄段)的重复
- 保留第一次出现的记录或选择最可靠的数据源

**示例代码**：
```python
# 删除完全重复行
df_deduplicated = df.drop_duplicates()

# 基于特定列的去重
df_deduplicated = df.drop_duplicates(subset=['Year', 'Region', 'AgeGroup'])
```

#### 步骤 2.4：异常值检测与处理
- **统计异常**：使用 3-sigma 规则检测人口数据异常
- **逻辑异常**：检测人口数据中的不合理值（如负数、超大增长）
- **一致性异常**：检测同一年份不同地区数据的不一致

**示例代码**：
```python
# 3-sigma 异常检测
mean = df['Population'].mean()
std = df['Population'].std()
df['is_outlier'] = (df['Population'] - mean).abs() > 3 * std

# 检测负数或零值
invalid_records = df[df['Population'] <= 0]

# 处理异常值
df_clean = df[~df['is_outlier']]
```

#### 步骤 2.5：格式标准化
将原始数据转换为**长格式（Long Format）**，便于时间序列分析和建模：

**原始格式（宽格式）**：
```
| Year | 0-4岁 | 5-9岁 | 10-14岁 | ... |
|------|-------|-------|---------|-----|
| 2000 |       |       |         |     |
| 2001 |       |       |         |     |
```

**目标格式（长格式）**：
```
| Year | Region  | AgeGroup | Population |
|------|---------|----------|------------|
| 2000 | England | 0-4      | 3,234,567  |
| 2000 | England | 5-9      | 3,345,678  |
| 2000 | Scotland| 0-4      | 234,567    |
```

**示例代码**：
```python
import pandas as pd

# 从宽格式转换为长格式
df_long = df.melt(
    id_vars=['Year', 'Region'],
    value_vars=['0-4', '5-9', '10-14', ...],
    var_name='AgeGroup',
    value_name='Population'
)
```

### 3. 地区数据融合（Transform - 多源融合）

#### 步骤 3.1：历史数据与投影数据合并
```
历史数据（1991-2018）+ 投影数据（2018-2070）
                    ↓
              数据对齐与去重
                    ↓
          完整时间序列（1991-2070）
```

**合并逻辑**：
```python
# 合并历史数据和投影数据
df_merged = pd.concat([df_history, df_projection], ignore_index=True)

# 按年份排序
df_merged = df_merged.sort_values('Year').reset_index(drop=True)

# 检测并处理重叠部分
overlap_period = df_merged[(df_merged['Year'] >= 2018) & (df_merged['Year'] <= 2020)]
```

#### 步骤 3.2：地区数据统一
- 英格兰、苏格兰、威尔士分别处理后统一
- 创建"英国总体"数据（聚合所有地区）
- 统一地区名称表示法

**示例代码**：
```python
# 按地区过滤
df_england = df[df['Region'] == 'England']
df_scotland = df[df['Region'] == 'Scotland']
df_wales = df[df['Region'] == 'Wales']

# 创建英国整体数据
df_uk = df.groupby('Year').agg({'Population': 'sum'}).reset_index()
df_uk['Region'] = 'UK'
```

### 4. 特征计算与工程（Transform - 特征创建）

#### 步骤 4.1：计算老龄化比例
```
65+ 人口 = 所有 ≥65 岁年龄段人口之和
老龄化比例 = 65+ 人口 / 总人口 * 100%
```

**示例代码**：
```python
# 定义老龄化年龄段
age_groups_65plus = ['65-69', '70-74', '75-79', '80-84', '85+']

# 计算 65+ 人口
df['population_65plus'] = df[df['AgeGroup'].isin(age_groups_65plus)]['Population'].sum()

# 计算老龄化比例
df['ageing_ratio'] = (df['population_65plus'] / df['total_population']) * 100
```

#### 步骤 4.2：时间特征提取
为模型提供时间序列特征：
```python
# 创建时间特征
df['year_numeric'] = df['Year'] - df['Year'].min()  # 从 0 开始的年份编码
df['year_scaled'] = df['year_numeric'] / df['year_numeric'].max()  # 归一化年份
```

#### 步骤 4.3：趋势特征计算
```python
# 计算变化率
df['population_change'] = df['Population'].diff()  # 绝对变化
df['population_pct_change'] = df['Population'].pct_change() * 100  # 百分比变化

# 计算滚动平均（平滑噪声）
df['population_ma3'] = df['Population'].rolling(window=3).mean()
df['population_ma5'] = df['Population'].rolling(window=5).mean()
```

### 5. 数据验证与质量评估（Quality Assurance）

#### 步骤 5.1：数据完整性检查
```python
# 检查时间跨度完整性
years_range = range(df['Year'].min(), df['Year'].max() + 1)
missing_years = set(years_range) - set(df['Year'].unique())

# 检查地区覆盖
regions_coverage = df.groupby('Region')['Year'].count()
print(f"各地区数据覆盖情况：\n{regions_coverage}")

# 检查年龄段覆盖
age_groups_coverage = df.groupby('AgeGroup').size()
```

#### 步骤 5.2：数据一致性验证
```python
# 验证部分和整体的一致性
df_parts = df[df['Region'].isin(['England', 'Scotland', 'Wales'])].groupby('Year')['Population'].sum()
df_total = df[df['Region'] == 'UK'].groupby('Year')['Population'].sum()

# 比较差异
inconsistencies = (df_parts - df_total).abs() > 1000
```

#### 步骤 5.3：统计验证
```python
# 基本统计检查
print(f"人口数据统计：")
print(df['Population'].describe())

# 检查是否有异常的增长或下降
yearly_change = df.groupby('Year')['Population'].sum().pct_change()
extreme_changes = yearly_change[yearly_change.abs() > 0.05]  # >5% 的变化
```

### 6. 数据输出与存储（Load）

#### 步骤 6.1：处理结果导出
清洗后的数据导出为标准化 CSV 格式：

```python
# 导出清洗后的数据
df_clean.to_csv('data/processed/cleaned_population_long.csv', index=False, encoding='utf-8')

# 导出老龄化比例数据
df_ageing.to_csv('data/processed/ageing_ratio_per_region.csv', index=False, encoding='utf-8')

# 导出融合后的完整数据集
df_merged_final.to_csv('data/processed/uk_population_projection_all.csv', index=False, encoding='utf-8')
```

#### 步骤 6.2：聚类分析数据准备
为聚类算法准备数据：

```python
# 提取特征
features = ['Year', 'AgeGroup', 'Population', 'Ageing_Ratio']
df_cluster_input = df[features].copy()

# 标准化（使不同量纲的特征可比）
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_cluster_input_scaled = scaler.fit_transform(df_cluster_input)

# 导出聚类输入数据
pd.DataFrame(df_cluster_input_scaled).to_csv('data/processed/ageing_cluster_input.csv', index=False)
```

### 7. 数据处理流程

**数据处理流程概览**

```
步骤 1: 数据采集 (Extract)
├─ 从 Excel/XLS 读取 ONS 原始数据
├─ 来源: data/raw/
└─ 输出: 原始 DataFrame

步骤 2: 数据清洗 (Transform - Cleaning)
├─ 缺失值处理 (检测、填充或删除)
├─ 数据类型转换 (年份、人口数等)
├─ 去重处理 (基于多列组合去重)
└─ 异常值检测 (3-sigma、逻辑异常)

步骤 3: 数据融合 (Transform - Fusion)
├─ 格式标准化 (宽格式 → 长格式)
├─ 历史+投影数据合并 (1991-2070)
├─ 地区数据统一 (英格兰/威尔士/苏格兰)
└─ 年份对齐 (确保完整的时间序列)

步骤 4: 特征工程 (Feature Engineering)
├─ 计算老龄化比例 (65+ 人口占比)
├─ 时间特征提取 (年份编码、归一化)
├─ 趋势特征计算 (变化率、百分比变化)
└─ 滚动平均平滑 (3年、5年)

步骤 5: 数据验证 (Validate)
├─ 完整性检查 (缺失率、覆盖率)
├─ 一致性验证 (部分和与整体对齐)
└─ 统计验证 (数据分布、异常检测)

步骤 6: 数据导出 (Load)
├─ 输出目录: data/processed/
├─ 清洗数据: cleaned_population_long.csv
├─ 老龄化比例: ageing_ratio_per_region.csv
├─ 融合数据: uk_population_projection_all.csv
└─ 聚类输入: ageing_cluster_input.csv

步骤 7: 下游应用
├─ 时间序列预测 (Prophet/ARIMA 模型)
├─ 聚类分析 (KMeans 地区分类)
└─ 结果可视化 (图表生成)
```

### 8. 数据处理质量指标

| 指标 | 目标 | 验证方式 |
|-------|------|---------------------|
| **缺失率** | < 1% | `df.isnull().sum() / len(df)` |
| **重复率** | = 0% | `len(df) == len(df.drop_duplicates())` |
| **离群点率** | < 2% | 使用 IQR 或 3-sigma 检测 |
| **数据一致性** | ≥ 98% | 部分和与整体的差异 |
| **时间覆盖率** | = 100% | 所有年份都有数据 |

### 9. 代码实现

完整的数据处理流程在以下文件中实现：

| 文件 | 功能 |
|---------------------------|---------------------|
| `src/preprocess.py` | 英国整体数据清洗 |
| `src/preprocess_england.py` | 英格兰数据处理 |
| `src/preprocess_scotland.py` | 苏格兰数据处理 |
| `src/preprocess_wales.py` | 威尔士数据处理 |
| `src/preprocess_projections.py` | 投影数据处理 |
| `src/merge_projection_data.py` | 数据融合与特征计算 |
| `main.py` | 完整流程编排 |

---

## 🚀 快速开始

### 前置要求
- Python 3.8+ (推荐 conda 环境)
- Node.js 18+

### 1. 生成分析数据（离线批处理）

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

API 文档自动生成：`http://localhost:8000/docs`

### 3. 启动前端（React + Vite）

```bash
cd frontend
npm install
npm run dev
```

打开浏览器访问：`http://localhost:5173`

### 主要页面

| 页面 | 路径 | 说明 |
|------|------|------|
| **Dashboard** | `/dashboard` | 历史老龄化趋势总览与统计卡片 |
| **Forecast** | `/forecast` | 交互式预测图（切换地区 / 模型） |
| **Cluster** | `/cluster` | KMeans 聚类分析与地区分组 |

## 🛠️ 技术栈

### 后端
- **Python 3.8+** - 核心开发语言
- **FastAPI** - 高性能 REST API 框架
- **uvicorn** - ASGI 服务器

### 数据分析与建模

### 数据处理与分析
| 技术 | 功能 |
|------|------|
| **Pandas** | 数据框架、数据清洗、时间序列处理 |
| **NumPy** | 多维数组运算、数值计算 |
| **SciPy** | 科学计算、统计分析 |

### 时间序列预测
| 技术 | 功能 |
|------|------|
| **Prophet** (Facebook) | 时间序列分解、趋势预测、变化点检测 |
| **ARIMA** (pmdarima) | 自回归综合移动平均模型、自动参数优化 |
| **Statsmodels** | 统计模型、假设检验、时间序列诊断 |

### 机器学习与聚类
| 技术 | 功能 |
|------|------|
| **Scikit-learn** | KMeans 聚类、数据标准化、模型评估 |

### 前端
| 技术 | 功能 |
|------|------|
| **React 18** | 组件化 UI 框架 |
| **Vite** | 前端构建工具，内置开发代理 |
| **Recharts** | 基于 React 的数据可视化图表库 |
| **Tailwind CSS** | 实用优先的 CSS 框架 |
| **React Router v6** | 前端路由 |
| **Axios** | HTTP 请求客户端 |

## 📊 主要依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| fastapi | 0.110+ | REST API 框架 |
| uvicorn | 0.29+ | ASGI 服务器 |
| pandas | 2.2.2 | 数据处理和操作 |
| numpy | 1.24.4 | 数值计算 |
| prophet | 1.1.7 | Facebook Prophet 时间序列预测 |
| statsmodels | 0.14.1 | 统计模型和测试 |
| pmdarima | 2.0.4 | ARIMA 自动参数选择 |
| scikit-learn | 1.3.2 | 机器学习（聚类、预处理） |
| react | 18.3+ | 前端 UI 框架 |
| recharts | 2.12+ | React 图表库 |

## 📈 API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/ageing-ratio` | 各地区历史老龄化比例数据 |
| GET | `/api/regions` | 可用地区列表 |
| GET | `/api/overview` | 各地区统计摘要 |
| GET | `/api/forecast/prophet?region=England` | Prophet 预测序列 |
| GET | `/api/forecast/arima?region=England` | ARIMA 预测序列 |
| GET | `/api/metrics` | 模型评估指标对比 |
| GET | `/api/cluster?n_clusters=3` | KMeans 聚类结果 |

- `england_timeseries.csv` - 英格兰时间序列数据
- `multi_compare/prophet_arima_metrics.csv` - 模型评估指标

## 🔧 配置参数

在 `main.py` 中可以调整以下参数：

```python
CONFIG = {
    "regions": ["England", "Wales", "Scotland"],  # 分析地区
    "end_year": 2070,                              # 预测终点年份
    "test_year_start": 2030,                       # 测试集起始年份
    "horizon": 30,                                 # 预测时间跨度（年）
    "n_clusters": 3,                               # 聚类数量
    "random_state": 42                             # 随机种子
}
```

## 📝 模型说明

### Prophet 模型
- 使用分段线性趋势 + 季节性分解
- 可选逻辑增长约束（防止不切实际的高增长）
- 自适应变化点检测捕捉趋势变化
- 支持滚动平均平滑预测结果

### ARIMA 模型
- 使用 pmdarima 库进行自动参数搜索 (p,d,q)
- 通过 AIC 准则选择最优参数
- 适合于平稳或一阶差分平稳的时间序列

### 聚类分析
- 对各地区的老龄化趋势进行特征提取
- 使用 KMeans 进行无监督聚类
- 标准化处理确保公平对比

## 💡 主要特点

✅ **多源数据整合** - 融合 ONS 历史数据和官方投影数据  
✅ **多模型对比** - Prophet、ARIMA 等多种预测方法  
✅ **区域分析** - 支持英国各地区的独立和对比分析  
✅ **聚类识别** - 自动识别区域老龄化特征相似性  
✅ **完整流程** - 从数据预处理到可视化的端到端工作流  
✅ **可重现性** - 固定随机种子确保结果可重现  

## 📧 联系方式

如有问题或建议，欢迎提出 Issue 或 Pull Request。

## 📄 许可证

此项目仅供学术和研究使用。
