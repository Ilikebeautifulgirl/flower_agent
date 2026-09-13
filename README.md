<div align="center">

# 🌸 花小满 AI Agent

**基于 LangGraph + MCP + RAG 的鲜花批发商城多智能体客服系统**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange)](https://langchain-ai.github.io/langgraph/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[功能亮点](#功能亮点) · [架构设计](#架构设计) · [技术栈](#技术栈) · [快速开始](#快速开始) · [API 文档](#api-接口)

</div>

---


## 功能亮点

花小满 AI Agent 是一个面向鲜花批发商城的智能客服系统，采用 **多智能体协作架构**，将复杂的客服任务拆解给不同领域的专业 Agent 协同处理，支持图文多模态输入、对话式下单、智能配花等高级功能。

| 功能 | 说明 | 负责 Agent |
|------|------|-----------|
| 🌺 **花材知识问答** | 花语寓意、养护方法、品种等级、产地等花卉知识，基于 RAG + 知识图谱双检索 | ProductAgent |
| 🛒 **商品与订单查询** | 商城在售商品搜索、价格库存查询、个人订单记录、消费统计 | BillingAgent |
| 📦 **对话式下单** | 自然语言下单买花，自动查价、确认、创建订单，返回订单号 | OrderAgent |
| 💐 **智能配花方案** | 根据场景/预算/色系自动搭配花束配方，校验库存与真实价格，一键加入购物车 | FloristAgent |
| 🖼️ **图片识图** | 上传花卉照片自动识别品种，支持以图搜花、看图下单 | VisionAgent |
| 💰 **成本优化分析** | 采购浪费诊断、进货成本优化建议（FinOps 工作流） | FinOpsAgent |
| 🧠 **对话记忆** | Redis 短期记忆，多轮对话上下文自动关联 | ShortTermMemory |

---

## 架构设计

### 整体架构

```
用户 ──HTTP/SSE──> FastAPI (uvicorn:8100)
                        │
                        ▼
                 LangGraph 工作流
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    VisionAgent   Orchestrator    (条件路由)
                        │
          ┌─────────────┼─────────────┬─────────────┬─────────────┐
          ▼             ▼             ▼             ▼             ▼
    ProductAgent   BillingAgent   OrderAgent   FloristAgent   FinOpsAgent
     (RAG+图谱)    (MCP+MySQL)   (MCP+MySQL)   (MCP+MySQL)   (MCP+MySQL)
```

### 多智能体协作流程

1. **VisionAgent** — 入口节点，检测是否有图片上传，有图则调用视觉模型识别花卉品种后再进入主流程
2. **Orchestrator** — 编排器/路由官，基于 LLM 做意图分类，决定将问题分发给哪个专业 Agent
3. **专业 Agent** — 各自领域独立工作，通过 MCP 协议调用数据库工具，通过 RAG 检索知识库
4. **跨 Agent 工作流** — FinOps 等复杂场景支持多 Agent 串联协作（Billing → FinOps）

### 核心设计理念

- **单一职责**：每个 Agent 只负责一个领域，Prompt 更精准，回答质量更高
- **易于扩展**：新增业务能力只需加一个 Agent 文件 + 调整路由规则
- **工具即能力**：通过 MCP 协议封装数据库操作为工具，Agent 自主决定何时调用
- **安全隔离**：`user_id` 通过拦截器强制注入，杜绝越权查询他人数据
- **流式输出**：SSE 打字机效果，提升用户体验

---

## 技术栈

### 后端框架
- **Python 3.10+** — 主开发语言
- **FastAPI** — HTTP 服务层，支持 SSE 流式响应
- **Uvicorn** — ASGI 服务器

### AI 框架
- **LangGraph** — 多智能体工作流编排
- **LangChain** — LLM 应用开发框架
- **create_react_agent** — ReAct 模式 Agent 执行器

### 大模型
- **阿里云百炼 (DashScope)** — 模型服务提供商
- **qwen-plus** — 文本推理主模型
- **qwen-vl-plus** — 多模态视觉模型

### 数据存储
- **MySQL** — 订单、商品、用户数据
- **Redis** — 短期对话记忆
- **FAISS** — 向量数据库（RAG 检索）
- **Neo4j** — 知识图谱（花卉品种/等级/产地关系）

### 协议标准
- **MCP (Model Context Protocol)** — 模型上下文协议，Agent 与工具服务间的标准通信协议
- **SSE (Server-Sent Events)** — 服务器推送事件，流式对话输出

---

## 快速开始

### 环境要求

- Python 3.10 或更高版本
- Docker & Docker Compose（运行 MySQL / Redis / Neo4j）
- 阿里云百炼 API Key（[免费申请](https://dashscope.console.aliyun.com/)）

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/your-username/flower_agent.git
cd flower_agent

# 2. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key 和数据库密码

# 3. 启动中间件（MySQL / Redis / Neo4j）
docker compose up -d

# 4. 安装 Python 依赖
pip install -r requirements.txt

# 5. 初始化数据库
#    导入 mock 订单数据
docker cp database/init_mock_data.sql my-mysql:/tmp/init.sql
docker exec my-mysql sh -c "mysql -uroot -p你的密码 --default-character-set=utf8mb4 flower_cloud < /tmp/init.sql"

#    导入实例监控数据
docker cp database/init_instances.sql my-mysql:/tmp/init.sql
docker exec my-mysql sh -c "mysql -uroot -p你的密码 --default-character-set=utf8mb4 flower_cloud < /tmp/init.sql"

#    导入产品数据到 MySQL + 生成 RAG 文档
python database/import_products.py

#    灌入 Neo4j 知识图谱
python database/seed_neo4j.py

# 6. 启动服务
python -m uvicorn api_server:app --host 0.0.0.0 --port 8100
```

启动成功后，浏览器打开 `http://localhost:8100` 即可体验内置聊天界面。

### 命令行调试模式

```bash
python main.py
```

---

## 项目结构

```
flower_agent/
├── api_server.py                  # FastAPI HTTP 服务层（SSE 流式聊天 + 图片上传）
├── main.py                         # 命令行交互入口（开发调试用）
├── docker-compose.yml              # MySQL + Redis + Neo4j 一键启动
├── requirements.txt                # Python 依赖
├── .env.example                    # 环境变量模板
│
├── agents/                         # 各 Agent 节点（业务逻辑）
│   ├── orchestrator.py             #   编排器：意图分类路由
│   ├── product_agent.py            #   花材知识顾问：RAG + 知识图谱
│   ├── billing_agent.py            #   订单查询：MCP 工具调 MySQL
│   ├── order_agent.py              #   下单助手：MCP 工具创建订单
│   ├── florist_agent.py            #   配花师：花束方案设计
│   ├── finops_agent.py             #   成本优化：FinOps 分析
│   └── vision_agent.py             #   多模态识图：视觉模型
│
├── core/                           # 核心模块
│   ├── workflow/
│   │   ├── graph_manager.py        #   LangGraph 状态图组装
│   │   └── state.py                #   AgentState 数据结构定义
│   ├── memory/
│   │   └── short_term.py           #   Redis 短期对话记忆
│   ├── auth.py                     #   认证管理（HMAC Token）
│   └── recommend.py                #   推荐算法
│
├── tools/                          # 检索工具
│   ├── vector_tool.py              #   FAISS 向量检索（RAG）
│   └── graph_tool.py               #   Neo4j 知识图谱查询
│
├── mcp_servers/                    # MCP 服务端
│   └── flower_cloud_server.py    #   封装 MySQL 查询为 MCP 工具
│
├── database/                       # 数据库脚本与数据
│   ├── 在售产品列表.csv             #   示例产品数据（35 条）
│   ├── init_mock_data.sql          #   Mock 订单数据
│   ├── init_instances.sql          #   Mock 实例监控数据
│   ├── import_products.py          #   CSV → MySQL + 生成 RAG 文档
│   └── seed_neo4j.py               #   CSV → Neo4j 知识图谱
│
│
├── faiss_index/                    # FAISS 索引（首次运行自动构建）
├── uploads/                        # 用户上传图片
│
└── deploy/                         # 生产部署配置
    ├── deploy.sh                   #   一键部署脚本（Ubuntu）
    ├── flower_agent.service     #   systemd 进程守护
    ├── nginx_flower_agent.conf  #   Nginx 反向代理
    ├── init_flower_shop_db.sql            #   商城库初始化
    └── php_token_example.php       #   PHP 端 Token 认证示例
```

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | 内置网页测试客户端 |
| `GET` | `/health` | 健康检查 |
| `POST` | `/upload` | 上传图片（返回图片 URL） |
| `POST` | `/chat` | SSE 流式聊天接口 |

### /chat 请求示例

```json
{
  "user_id": "10001",
  "session_id": "session_abc123",
  "message": "卡罗拉的花语是什么？",
  "image_url": ""
}
```

### /chat 响应格式

SSE 流式返回，事件类型：
- `token` — 增量文本片段（打字机效果）
- `end` — 对话结束

---

## 安全说明

- `.env` 文件已加入 `.gitignore`，API Key、数据库密码等敏感信息不会提交到代码仓库
- `user_id` 支持 HMAC Token 验证，生产环境由后端签发，前端无法伪造
- 未配置 `AUTH_SECRET` 时回退到请求体 `user_id`（仅限开发测试环境）
- 所有涉及用户数据的 MCP 工具调用均通过拦截器强制注入服务端 `user_id`，防止越权

---

## 部署

项目提供了完整的生产部署方案：

- **`deploy/deploy.sh`** — Ubuntu 服务器一键部署脚本
- **`deploy/flower_agent.service`** — Systemd 服务配置，进程守护与自动重启
- **`deploy/nginx_flower_agent.conf`** — Nginx 反向代理配置（含 SSE 流式支持）

```bash
# 服务器上一键部署
bash deploy/deploy.sh
```

---

## License

[MIT](LICENSE)

---

<div align="center">

**如果这个项目对你有帮助，欢迎给个 Star ⭐**

</div>
