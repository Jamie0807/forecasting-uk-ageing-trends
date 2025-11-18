# 英国人口老龄化趋势预测

这是一个基于英国国家统计局（ONS）数据的人口老龄化趋势预测和区域差异分析项目。项目采用多种时间序列预测模型（Prophet 和 ARIMA）对英国各地区 65 岁及以上人口比例进行长期预测，并通过聚类分析识别区域老龄化特征。

## 📋 项目概述

- **研究对象**：英国各地区（英格兰、威尔士、苏格兰）65 岁及以上人口的比例变化
- **预测时间跨度**：2020-2070 年
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

## 🚀 快速开始

### 前置要求
- Python 3.8+
- pip 或 conda

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行项目

完整流程（从数据预处理到最终预测和可视化）：
```bash
python main.py
```

main.py 包含以下主要步骤：
1. **数据预处理** - 清洗 ONS 原始数据
2. **计算老龄化比例** - 按地区计算 65+ 人口占比
3. **Prophet 预测** - 各地区长期预测
4. **ARIMA 预测** - 对比分析
5. **多模型评估** - 计算 MAE、RMSE 等指标
6. **聚类分析** - 地区老龄化特征分组
7. **结果可视化** - 生成各类图表和预测数据

## �️ 技术栈

### 编程语言与平台
- **Python 3.8+** - 核心开发语言
- **Jupyter Notebook** - 交互式数据分析和探索

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
| **标准化 (StandardScaler)** | 数据预处理、特征归一化 |

### 数据可视化
| 技术 | 功能 |
|------|------|
| **Matplotlib** | 基础绘图、图表生成 |
| **Pillow** | 图像处理 |

### 性能优化
| 技术 | 功能 |
|------|------|
| **Cython** | C 加速、性能优化 |
| **Joblib** | 并行计算、缓存管理 |
| **Threadpoolctl** | 线程池控制 |

### 模型框架
- **CmdStanPy** - Stan 概率编程框架（Prophet 内部使用）

### 数据输入格式
- **Excel** (.xlsx, .xls) - ONS 原始数据
- **CSV** (.csv) - 处理后的数据格式

## �📊 主要依赖

| 包名 | 版本 | 用途 |
|------|------|------|
| pandas | 2.2.2 | 数据处理和操作 |
| numpy | 1.24.4 | 数值计算 |
| matplotlib | 3.7.5 | 数据可视化 |
| prophet | 1.1.7 | Facebook Prophet 时间序列预测 |
| statsmodels | 0.14.1 | 统计模型和测试 |
| pmdarima | 2.0.4 | ARIMA 自动参数选择 |
| scikit-learn | 1.3.2 | 机器学习（聚类、预处理） |
| scipy | 1.10.1 | 科学计算 |
| cmdstanpy | 1.2.5 | Stan 概率编程框架 |
| Cython | 3.1.2 | C 加速编译 |
| joblib | 1.5.1 | 并行计算 |
| holidays | 0.78 | 假期处理（Prophet 使用） |
| pillow | 11.3.0 | 图像处理 |

## 📈 输出示例

项目会生成以下结果：

### 可视化图表
- `ageing_trend_65plus_fixed.png` - 英国 65+ 人口比例历史趋势
- `prophet_65plus_england.png` - 英格兰 Prophet 预测结果
- `prophet_65plus_all_regions.png` - 全部地区 Prophet 预测对比
- `compare_arima_prophet_england.png` - Prophet vs ARIMA 对比
- `ageing_clusters.png` - 地区聚类可视化
- `forecast_comparison_regions.png` - 多地区预测对比

### 数据导出
- `england_forecast.csv` - 英格兰预测数据
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
