# 英国人口老龄化趋势预测平台 - 简历版本

## 📄 项目标题
**英国人口老龄化趋势预测与交互式分析平台（Python + React 全栈）**

---

## 📋 项目描述

### 背景与目标
基于英国国家统计局（ONS）官方数据，构建了一套**全栈数据分析与预测平台**：使用 Python 完成数据 ETL 与模型训练，通过 **FastAPI** 将 Prophet、ARIMA、KMeans 模型封装为 REST API，由 **React + Vite** 前端消费接口并以 Recharts 交互式图表呈现，预测 2020–2070 年英国三地区的老龄化趋势。

---

## 🎯 核心职责与成果

### 1. **数据工程与预处理** 
- 从多个Excel/XLS格式的ONS原始数据源中提取、清洗和转换人口统计数据
- 设计数据处理流程，支持分地区的数据预处理（英格兰、威尔士、苏格兰）
- 实现历史数据与投影数据的融合，生成50年的完整数据集

### 2. **时间序列预测建模** 
- 应用**Prophet**和**ARIMA**两种时间序列预测算法，对各地区老龄化趋势进行长期预测
- 实现模型对比分析框架，评估不同模型的预测准确性和适用性
- 支持多地区并行预测与结果汇总

### 3. **聚类分析与区域洞察** 
- 利用KMeans聚类算法对各地区老龄化特征进行分类，识别同类型地区
- 通过标准化预处理确保分析的公平性和可靠性
- 提供数据驱动的地区分组依据

### 4. **模型服务化与全栈部署**
- 使用 **FastAPI** 将 Prophet、ARIMA、KMeans 封装为 REST API，设计 7 个语义化端点（`/api/forecast/prophet`、`/api/cluster` 等）
- 建立 **`routers/` + `services/`** 双层架构：路由层负责参数校验，服务层封装模型调用，关注点分离
- 构建 **React + Vite** 前端，Axios 动态调用 API，Recharts 实时渲染交互式预测图表，支持地区 / 模型自由切换
- 配置 Vite 反向代理与 FastAPI CORS 中间件，解决前后端跨域问题
- 本地托管 Swagger UI，解决生产环境 CDN 被屏蔽问题，完成 API 文档可访问性部署

---

## 🛠️ 技术栈

**编程语言：** Python、JavaScript（React）

**后端：**
- API 框架：FastAPI、uvicorn
- 数据处理：Pandas, NumPy
- 统计建模：Scikit-learn, Statsmodels, Prophet, PMDarima

**前端：**
- 框架：React 18、Vite
- 图表：Recharts
- 样式：Tailwind CSS
- HTTP 客户端：Axios

---

## 📊 项目规模与成果

| 指标 | 数值 |
|------|------|
| 代码模块数 | 15+ 个（Python ETL + FastAPI 后端 + React 前端） |
| 处理地区数 | 3 个（英格兰、威尔士、苏格兰） |
| 时间跨度 | 50 年（2020-2070） |
| 预测模型 | 2 种（Prophet + ARIMA），通过 REST API 对外服务 |
| REST API 端点 | 7 个（数据 / 预测 / 聚类） |
| 前端交互页面 | 3 个（Dashboard / Forecast / Cluster） |
| 可视化方案 | Matplotlib 离线图 + Recharts 交互式图表 |
| 架构 | 前后端分离（FastAPI + React） |

---

## 💡 面试讲述思路

### 开场白
> "这是我的一个个人项目，利用公开的人口统计数据，预测英国的老龄化趋势。这个项目结合了数据处理、统计建模和可视化等数据分析的全流程。"

### 重点强调（按优先级）
1. **业务理解**：理解人口老龄化的实际意义，知道为什么要预测和如何预测
2. **数据处理**：展示你能处理复杂、不规范的真实数据（Excel/XLS）
3. **建模能力**：展示你了解多种预测算法，能选择合适的方法
4. **结果输出**：强调可视化的重要性，数据最终要被业务人员理解

---

## 🤔 常见面试问题与回答要点

### Q1: "你是怎么把 Python 分析模型部署成 Web 服务的？"
**回答要点：**
- 原本 Prophet、ARIMA 是独立的 Python 脚本，通过 **FastAPI** 将它们封装成 REST API 端点
- 采用 **`routers/` + `services/`** 分层：路由层处理 HTTP 参数，服务层调用模型并返回 JSON
- React 前端用 **Axios** 调用接口，配合 **Vite 反向代理** 解决跨域，数据由 **Recharts** 渲染成交互式图表
- 模型推理读取预生成的预测 CSV，响应 < 100ms，体验流畅
- 踩坑经验：Swagger UI 默认走 CDN，国内访问失败，改为本地托管 `swagger-ui-bundle` 包解决

### Q2: "数据从哪里来？怎么处理的？"
**回答要点：**
- 数据来自英国国家统计局（ONS）官方网站
- 原始数据来自多个Excel文件，格式不一致
- 进行了数据清洗、去重、格式转换、标准化等处理
- 最终生成了结构化的长格式CSV数据

### Q3: "预测准确度如何评估？"
**回答要点：**
- 用Prophet和ARIMA两个模型进行对比
- 评估在不同时间段的表现
- 生成了对比可视化图表

### Q4: "如果要做这个分析，有什么限制或假设？"
**回答要点：**
- 数据仅到2018年，预测基于历史趋势
- 假设未来没有重大社会变化（如大规模移民）
- 预测精度随时间增加而降低

### Q5: "这个项目对业务有什么价值？"
**回答要点：**
- 为政府部门提供养老社会服务规划的数据支撑
- 帮助医疗和福利机构进行资源规划
- 为社会政策制定提供科学依据

---

## 📁 项目文件组织

```
forecasting-uk-ageing-trends/
├── main.py                    # 项目主入口，定义执行流程
├── requirements.txt           # Python 依赖包列表
├── README.md                  # 项目详细说明
│
├── data/
│   ├── raw/                   # 原始数据（ONS Excel/XLS 文件）
│   └── processed/             # 处理后的数据
│
├── output/                    # 输出结果（图表和预测数据）
│
├── backend/                   # FastAPI 后端（REST API 服务）
│   └── app/
│       ├── routers/           # 路由层（HTTP 参数校验）
│       └── services/          # 服务层（模型调用逻辑）
├── frontend/                  # React 前端
│   └── src/
│       ├── api/client.js      # Axios API 统一封装
│       └── pages/             # Dashboard / Forecast / Cluster
└── src/                       # Python 分析模块（被 services 层调用）
    ├── 数据预处理模块
    ├── 数据融合模块
    ├── 时间序列预测模块（Prophet / ARIMA）
    ├── 聚类分析模块（KMeans）
    └── 离线可视化模块
```

---

## 🚀 关键代码片段示例

### 数据处理
```python
# 从Excel读取数据，进行清洗和转换
import pandas as pd
df = pd.read_excel('raw_data.xlsx', sheet_name='Population')
df_clean = df.dropna().sort_values(['Year', 'Region'])
```

### 时间序列预测
```python
from prophet import Prophet
model = Prophet(growth='logistic')
model.fit(df_prophet_format)
forecast = model.predict(future)
```

### 聚类分析
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_normalized)
```

---

## 📈 学习收获

通过这个项目，我获得了：

✅ **数据处理能力**：处理复杂、多源、异构的真实数据  
✅ **统计建模能力**：理解并应用多种时间序列预测模型  
✅ **对比分析能力**：评估不同模型的优劣，选择合适的方案  
✅ **自动化流程设计**：将复杂分析流程组件化和自动化  
✅ **沟通展示能力**：通过可视化让数据变得易理解易传播  

---

## 📝 补充说明

- **项目完整性**：从数据获取到最终输出，完整的端到端流程
- **代码质量**：模块化设计，便于维护和扩展
- **文档完善**：完整的README和注释文档
- **可重现性**：所有结果可完全重现，提供了requirements.txt依赖列表
