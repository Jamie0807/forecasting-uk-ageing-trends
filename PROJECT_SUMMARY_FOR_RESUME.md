# 项目简历文档 ·【前端 / 全栈开发岗】适用

## 📄 简历项目条目（直接复制使用）

**英国人口老龄化趋势可视化平台** | React 18 · FastAPI · Recharts · Tailwind CSS · Vite

> 独立开发的全栈数据可视化 Web 应用。前端使用 React 18 + Vite 构建，设计 Dashboard、Forecast、Cluster 三大交互页面；后端基于 FastAPI 提供 7 个 REST API 端点，将 Python 数据分析模型（Prophet / ARIMA / KMeans）服务化；Axios 统一封装请求层，Recharts 实现动态折线图与聚类趋势图，Vite 反向代理解决跨域，全链路前后端分离。

---

## 🎯 核心职责（写给 HR / 技术面试官看）

### 1. React 前端架构与页面开发 ★★★
- 使用 **React 18 + Vite** 搭建 SPA 项目脚手架，配置 Tailwind CSS 与 PostCSS 完成样式工程化
- 基于 **React Router v6** 实现客户端路由，设计 `/dashboard`、`/forecast`、`/cluster` 三条路由
- 拆分可复用组件：`Navbar`（导航）、`MetricsTable`（指标表格）、`StatCard`（统计卡片），props 设计清晰
- 使用 **React hooks**（`useState`、`useEffect`）管理异步数据请求、loading 状态、错误处理
- 使用 **lucide-react** 图标库，配合 Tailwind 完成深色顶栏 + 卡片式响应布局

### 2. 交互式数据可视化 ★★★
- 使用 **Recharts**（LineChart、ReferenceLine、Tooltip、Legend）渲染动态折线图
- 实现**地区切换**（England / Scotland / Wales）与**模型切换**（Prophet / ARIMA）联动，点击按钮即时刷新图表
- Forecast 页：历史线（蓝色实线）与预测线（琥珀色虚线）双轨渲染，ReferenceLine 标注预测起始年份
- Cluster 页：按聚类结果动态着色折线，不同 `n_clusters`（2/3/4）切换时图表实时更新
- Dashboard 页：将后端扁平 JSON 数据 pivot 转换为 Recharts 多系列格式，封装数据转换工具函数

### 3. 前后端接口对接与请求封装 ★★★
- 封装 `src/api/client.js`：Axios 实例统一设置 `baseURL`、`timeout`，集中管理 7 个 API 函数（`fetchProphetForecast(region)`、`fetchCluster(nClusters)` 等），调用方无需关心 URL 拼接
- 配置 **Vite `server.proxy`**，将 `/api` 请求转发至 `http://localhost:8000`，开发环境零跨域问题
- 处理接口异常：`try/catch` 捕获网络错误，组件内设置 `error` state 展示友好提示

### 4. FastAPI 后端设计与 REST API 开发 ★★
- 使用 **FastAPI** 设计 RESTful 接口，`routers/`（路由层，负责参数类型校验与 HTTP 响应）+ `services/`（服务层，负责数据读取与模型调用）双层分离
- 设计 7 个语义化端点：`GET /api/regions`、`GET /api/forecast/prophet?region=England`、`GET /api/cluster?n_clusters=3` 等
- 配置 **CORS 中间件**（`allow_origins`、`allow_methods`），支持前端跨端口安全访问
- 解决 Swagger UI 依赖外部 CDN 在国内被屏蔽的问题：挂载本地 `swagger-ui-bundle` 静态目录，覆盖默认 `/docs` 路由，自定义 `swagger_js_url` / `swagger_css_url`
- 使用 **uvicorn** 启动，`--reload` 热重载支持开发效率

### 5. 工程化配置 ★★
- `vite.config.js`：配置开发代理、端口（5173），`@vitejs/plugin-react` 插件
- `tailwind.config.js`：自定义 `brand-900` 主题色扩展
- `postcss.config.js`：集成 Tailwind + Autoprefixer
- `frontend/package.json`：设置 `"type": "module"` 消除 ESM 警告，管理全部前端依赖版本

---

## 🛠️ 技术栈（简历关键词）

| 分类 | 技术 |
|------|------|
| **前端框架** | React 18、Vite 5 |
| **路由** | React Router v6 |
| **数据可视化** | Recharts 2 |
| **样式** | Tailwind CSS 3、PostCSS |
| **HTTP 客户端** | Axios |
| **图标** | lucide-react |
| **后端框架** | FastAPI、uvicorn |
| **语言** | JavaScript（ES Module）、Python |
| **架构模式** | 前后端分离、REST API、SPA |

---

## 📊 量化成果（面试时报数据）

| 指标 | 数值 |
|------|------|
| 前端交互页面 | 3 个（Dashboard / Forecast / Cluster） |
| 可复用组件 | 5+ 个（Navbar / StatCard / MetricsTable 等） |
| REST API 端点 | 7 个 |
| 支持切换维度 | 地区 × 模型 × 聚类数，组合动态渲染 |
| API 响应时间 | < 100ms（读取预生成 CSV，无在线推理延迟） |
| 构建产物 | Vite 生产构建成功，bundle 压缩正常 |

---

## 🤔 面试常见问题与回答

### Q1: "介绍一下你做的全栈项目"
> "我独立开发了一个数据可视化平台，前端 React + Recharts，后端 FastAPI 提供接口，把 Python 数据分析模型包装成 REST API。用户可以在浏览器里点击切换地区和预测模型，图表实时刷新。整个链路我都自己搭，包括 Vite 代理配置、CORS 中间件、Axios 请求封装这些工程细节。"

### Q2: "你的 React 项目里怎么管理状态的？"
> "这个项目数据流比较简单，用 `useState` 管理当前选中的地区、模型、聚类数，`useEffect` 监听这些状态变化去触发 API 请求，拿到数据后更新图表数据 state。如果状态更复杂我会考虑 Zustand 或 Context，但这里没必要过度设计。"

### Q3: "跨域问题怎么解决的？"
> "开发环境用 Vite 的 `server.proxy` 把 `/api` 请求代理到后端 8000 端口，浏览器看到的永远是同源请求。后端 FastAPI 也配了 CORS 中间件，允许 `localhost:5173` 访问，两层都处理了，部署时只需改 origin 配置。"

### Q4: "Recharts 遇到过什么问题吗？"
> "遇到过数据格式问题。后端返回的是扁平数组 `[{year, region, value}]`，但 Recharts 的 LineChart 需要每行是一个 year 对象、不同 region 作为 key。我写了个 pivot 函数做转换，按 year 聚合后把各 region 的值挂到同一个对象上，才能渲染多条线。"

### Q5: "FastAPI 后端你怎么设计的？"
> "分了两层：`routers/` 只管路由匹配和入参校验（用 FastAPI 的 Query 类型约束），`services/` 负责读文件、调模型、返回结构化数据。这样路由层很薄，业务逻辑全在 services，方便单独测试和替换。"

### Q6: "项目有没有遇到印象深刻的 bug？"
> "有个生产环境的坑：FastAPI 的 `/docs` 默认从 `cdn.jsdelivr.net` 加载 Swagger UI，国内访问直接 ERR_CONNECTION_RESET。排查出来后改成安装 `swagger-ui-bundle` 包，把静态文件挂到本地路由，重写 `/docs` 端点指向本地 JS/CSS，解决了。这让我意识到依赖外部 CDN 的风险。"

---

## 💡 面试讲述策略（前端/全栈岗）

**开场白**（15 秒）
> "我做过一个全栈数据可视化项目，React 前端 + FastAPI 后端，自己设计 API、写组件、对接接口，整个链路都覆盖到了。"

**重点顺序**
1. **前端组件设计** — 三页面结构、交互逻辑、Recharts 用法
2. **接口层设计** — Axios 封装、Vite 代理、错误处理
3. **后端 API 设计** — FastAPI 分层、CORS、端点设计
4. **踩坑经验** — CDN 屏蔽问题（体现工程判断力）

**不要主动提** — 数据分析算法细节、Prophet/ARIMA 参数调优（除非面试官追问）
