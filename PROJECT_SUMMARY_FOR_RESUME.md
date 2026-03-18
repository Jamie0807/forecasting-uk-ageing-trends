# 项目简历文档 ·【前端 / 全栈开发岗】适用

## 📄 简历项目条目（直接复制使用）

**英国人口老龄化趋势预测与可视化平台** | React 18 · FastAPI · Recharts · Tailwind CSS · Vite

> 独立开发的全栈数据可视化 Web 应用。前端使用 React 18 + Vite 构建，设计 Dashboard、Forecast、Cluster 三大交互页面；后端基于 FastAPI 提供 7 个 REST API 端点，将 Python 数据分析模型（Prophet / ARIMA / KMeans）服务化；Axios 统一封装请求层，Recharts 实现动态折线图与聚类趋势图，Vite 反向代理解决跨域，全链路前后端分离。

---

## 🎯 核心职责

### 1. React 前端架构与页面开发 ★★★
- **主导** SPA 项目从零搭建：以 **React 18 + Vite 5** 为核心，集成 Tailwind CSS + PostCSS 完成样式工程化，形成统一的开发基线
- **设计** 三级页面路由体系（**React Router v6**），`/dashboard`、`/forecast`、`/cluster` 各页独立，路由与业务逻辑解耦
- **抽象** 5+ 个可复用 UI 组件（`Navbar`、`StatCard`、`MetricsTable` 等），遵循单一职责原则，props 接口清晰，复用率覆盖全部页面
- **运用** `useState` + `useEffect` 构建异步数据流：统一处理请求 loading、接口错误、空数据等边界状态，保障 UI 健壮性
- **打磨** 视觉细节：lucide-react 图标 + Tailwind 深色顶栏 + 响应式卡片布局，全站 UI 风格一致、可扩展

### 2. 全链路数据流设计与前后端集成 ★★★
- **打通** Python 模型到浏览器的完整数据链路：用户操作 → Axios 触发请求 → FastAPI 路由分发 → `services/` 读取模型结果序列化 JSON → React 状态更新 → Recharts 实时重绘，端到端零断层
- **封装** `src/api/client.js` 作为统一请求层：Axios 实例集中设置 `baseURL`、`timeout`，暴露 7 个语义化函数（`fetchProphetForecast(region)`、`fetchArimaForecast(region)`、`fetchCluster(nClusters)` 等），调用方零感知底层细节
- **实现** 地区 × 模型多维联动：任意维度切换均精准触发对应 Python 端点，图表与模型严格一一对应，杜绝脏数据渲染
- **配置** Vite `server.proxy` 将 `/api` 前缀请求透明转发至 Python 后端，开发阶段无跨域困扰，迁移部署环境仅需更改 `target`，改动成本极低

### 3. 交互式数据可视化 ★★★
- **深度运用** Recharts（`LineChart`、`ReferenceLine`、`Tooltip`、`Legend`），将 Python 模型输出的时序数据渲染为直观可交互图表
- **攻克数据格式转换难题**：后端返回扁平数组 `[{year, region, value}]`，手写 pivot 工具函数按 year 聚合、以 region 为 key 重构多系列格式，驱动 Recharts 多折线并排渲染
- **精细化 Forecast 页**：历史趋势（蓝色实线）与 Prophet / ARIMA 预测段（琥珀色虚线）双轨并排，`ReferenceLine` 精确标注预测起始年份，模型外推区间一目了然
- **实现 Cluster 页动态着色**：依据 Python KMeans 输出结果为各聚类折线动态分配颜色，`n_clusters`（2 / 3 / 4）切换时重新请求后端并无缝刷新图表，交互体验流畅
- **优化 Dashboard 多系列渲染**：多地区数据经 pivot 转换后供 Recharts 直接消费，图表渲染性能与数据准确性并重

### 4. FastAPI 后端分层架构 ★★
- **设计** `routers/`（HTTP 入参校验）+ `services/`（模型调用与数据处理）双层架构，关注点彻底分离；路由层极简，业务逻辑内聚于 services，便于独立测试与替换
- **规范** 7 个 RESTful 语义端点：`GET /api/forecast/prophet?region=England`、`GET /api/forecast/arima?region=Scotland`、`GET /api/cluster?n_clusters=3` 等，每个端点精确对应一个 Python 模型调用，接口设计自文档化
- **配置** CORS 中间件（`allow_origins`、`allow_methods`），精确授权 React 前端跨端口访问，兼顾安全性与开发便利性
- **定位并解决** Swagger UI CDN 屏蔽问题：排查出 `/docs` 默认依赖 `cdn.jsdelivr.net` 导致 `ERR_CONNECTION_RESET` 的根因，安装 `swagger-ui-bundle` 并将静态资源挂载到本地路由、重写 `/docs` 端点，彻底消除外部 CDN 依赖风险
- **提升** 开发效率：uvicorn `--reload` 热重载 + FastAPI 自动类型校验，接口联调周期大幅缩短

### 5. 前端工程化配置 ★★
- **`vite.config.js`**：配置 `server.proxy` 代理规则、开发端口（5173）、`@vitejs/plugin-react` 插件，统一开发与构建行为
- **`tailwind.config.js`**：扩展 `brand-900` 自定义主题色，确保品牌色在全项目中通过 Tailwind 工具类一致复用，避免硬编码色值散落各处
- **`postcss.config.js`**：集成 Tailwind CSS + Autoprefixer，自动添加浏览器前缀，样式兼容性无需手动维护
- **`frontend/package.json`**：设置 `"type": "module"` 消除 ESM / CJS 混用警告，锁定全部前端依赖版本，保障构建可复现性

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
