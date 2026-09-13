"""花小满 AI Agent - HTTP API 服务层（接入网站用）。

把 main.py 的命令行循环改造成 HTTP 接口：
    浏览器/小程序 ──HTTP──> FastAPI(本文件) ──> LangGraph 工作流

启动方式（在 flower_agent 目录下）：
    python -m uvicorn api_server:app --host 0.0.0.0 --port 8100

接口：
    GET  /health  健康检查（给运维/PHP 探活用）
    GET  /        内置网页测试客户端（浏览器打开 http://localhost:8100 即可聊）
    POST /chat    聊天接口，SSE 流式返回（打字机效果）

核心设计：
1. lifespan：图和 Redis 连接在服务启动时创建一次、全局复用（重建太重）
2. SSE 流式：用 graph.astream_events 抓住 LLM 逐 token 输出的事件实时推给前端
3. 过滤 orchestrator：路由 Agent 的"内心独白"不推给用户，只推最终回答的 token
4. 身份安全：user_id 由服务端解析（现在测试期从请求体来，
   生产环境必须改成从 PHP 传来的登录 token 里解，前端传什么都不算数）
"""

import json
import asyncio
import re
import uuid
import os
import hmac
import hashlib
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, File, UploadFile, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from core.workflow.graph_manager import AgentGraphManager
from core.memory.short_term import ShortTermMemory
from core.auth import AuthManager
from core.recommend import get_recommendations

# === 全局对象：服务启动时初始化，所有请求共享 ===
graph = None          # LangGraph 编译后的工作流图（重对象，只建一次）
memory = None         # Redis 短期记忆（redis.asyncio 客户端自带连接池，线程安全）
auth = None           # 认证管理器（验证码 + token）


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 生命周期钩子：startup 时建图，shutdown 时关连接。"""
    global graph, memory, auth
    print("🚀 服务启动：正在构建 LangGraph 工作流（首次会加载 FAISS/Neo4j，稍等）...")
    graph = AgentGraphManager().build_graph()
    memory = ShortTermMemory()
    await memory.initialize()
    auth = AuthManager()
    await auth.initialize()
    print("✅ 服务就绪：http://localhost:8100")
    yield
    await memory.close()
    print("👋 服务关闭，连接已释放")


app = FastAPI(title="花小满 AI Agent", lifespan=lifespan)

# === 图片上传目录：/upload 收到的图存这里，通过 /uploads/文件名 对外访问 ===
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.post("/upload")
async def upload_image(file: UploadFile = File(...),
                       authorization: str | None = Header(default=None)):
    """接收用户上传的图片：存到 uploads 目录，返回相对 URL 给前端。

    流程：前端选图 → 先调本接口拿 URL → 聊天请求带 image_url
    → vision_agent 节点识别（本地文件会转 base64 给视觉模型）。
    必须登录（请求头带 Bearer token）。
    """
    user_id = await auth.get_user_id(_extract_token(authorization))
    if user_id is None:
        return JSONResponse(status_code=401, content={"error": "请先登录"})
    ext = Path(file.filename or "img.jpg").suffix.lower() or ".jpg"
    name = f"{uuid.uuid4().hex[:12]}{ext}"
    data = await file.read()          # UploadFile.read 是异步的，不阻塞事件循环
    (UPLOAD_DIR / name).write_bytes(data)
    print(f"📤 收到上传图片: {file.filename} -> {name} ({len(data)} bytes)")
    return {"url": f"/uploads/{name}"}

# === CORS：开发期允许所有来源直连（生产环境应收紧成你网站的域名）===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SendCodeRequest(BaseModel):
    """POST /send_code 的请求体。"""
    phone: str                        # 手机号（11 位）


class LoginRequest(BaseModel):
    """POST /login 的请求体。"""
    phone: str                        # 手机号
    code: str                         # 短信验证码


class ChatRequest(BaseModel):
    """POST /chat 的请求体（user_id 不在请求体里，一律从 token 解析）。"""
    message: str                      # 用户这轮说的话
    session_id: str | None = None     # 会话 ID：前端首次可不传，服务端生成后通过 meta 事件返回
    image_url: str | None = None      # 本轮上传图片的 URL（/upload 返回的），识图用


def _extract_token(authorization: str | None) -> str:
    """从 Authorization 请求头取出 token。

    标准格式：Authorization: Bearer 3f9a...（Bearer 后面一个空格再跟 token）
    """
    if authorization and authorization.startswith("Bearer "):
        return authorization[7:].strip()
    return ""


@app.post("/send_code")
async def send_code(req: SendCodeRequest):
    """发送短信验证码（开发模式直接把验证码返回，方便页面测试）。"""
    phone = req.phone.strip()
    if not (len(phone) == 11 and phone.isdigit() and phone.startswith("1")):
        return {"ok": False, "message": "请输入正确的 11 位手机号"}
    return await auth.send_code(phone)


@app.post("/login")
async def login(req: LoginRequest):
    """验证码登录：手机号没注册过会自动注册，成功返回 token。"""
    phone = req.phone.strip()
    if not (len(phone) == 11 and phone.isdigit() and phone.startswith("1")):
        return {"ok": False, "message": "请输入正确的 11 位手机号"}
    return await auth.login(phone, req.code.strip())


@app.post("/logout")
async def logout(authorization: str | None = Header(default=None)):
    """退出登录：让 token 立即失效。"""
    token = _extract_token(authorization)
    await auth.logout(token)
    return {"ok": True}


@app.get("/api/recommendations")
async def recommendations(authorization: str | None = Header(default=None)):
    """个性化推荐：根据登录用户的商城进货历史推荐，新用户给热销。需登录。"""
    user = await auth.get_user_info(_extract_token(authorization))
    if user is None:
        return JSONResponse(status_code=401, content={"error": "请先登录"})
    # pymysql 是同步调用，放到线程池执行，避免阻塞事件循环
    result = await asyncio.to_thread(get_recommendations, user["phone"], 6)
    return result


class CheckoutRequest(BaseModel):
    """POST /api/checkout 的请求体：购物车里的商品。"""
    items: list[dict]                 # [{"name": "卡罗拉", "quantity": 2}, ...]


def _do_checkout(conn, user_id: int, items: list) -> dict:
    """同步结算：按商品名回库核验真实价格/库存，逐款建单。

    安全要点：价格一律以后端数据库查到的为准，前端传的 price 只展示、不入账，
    防止有人改请求体把 ¥50 的花改成 ¥0.01。
    """
    with conn.cursor() as cur:
        created, total_all = [], 0.0
        for it in items:
            name = str(it.get("name", "")).strip()
            qty = int(it.get("quantity", 0))
            if not name or qty <= 0:
                continue
            # 按名字查当前真实在售商品（取第一个精确/模糊匹配的在售款）
            cur.execute(
                """
                SELECT w.name, g.price, g.goods_stock
                FROM shop_goods g
                JOIN shop_goods_warehouse w ON g.goods_warehouse_id=w.id
                WHERE g.is_delete=0 AND w.is_delete=0 AND g.status=1
                  AND w.name=%s AND w.cover_pic LIKE 'http%%'
                ORDER BY g.sales DESC LIMIT 1
                """, (name,))
            goods = cur.fetchone()
            if not goods:
                return {"ok": False, "message": f"「{name}」已下架或不存在，请刷新购物车"}
            price = float(goods["price"])
            if int(goods["goods_stock"]) < qty:
                return {"ok": False,
                        "message": f"「{name}」库存不足（仅剩 {goods['goods_stock']} 件）"}
            total = round(price * qty, 2)
            cur.execute(
                """INSERT INTO agent_orders (user_id, product_name, price, quantity, total_price, status)
                   VALUES (%s, %s, %s, %s, %s, '待支付')""",
                (user_id, goods["name"], price, qty, total))
            created.append({"order_id": cur.lastrowid, "name": goods["name"],
                            "price": str(price), "quantity": qty, "total": str(total)})
            total_all += total
        conn.commit()
    if not created:
        return {"ok": False, "message": "购物车是空的"}
    return {"ok": True, "orders": created, "total": str(round(total_all, 2))}


@app.post("/api/checkout")
async def checkout(req: CheckoutRequest, authorization: str | None = Header(default=None)):
    """购物车结算：核验价格并创建待支付订单。需登录。"""
    user = await auth.get_user_info(_extract_token(authorization))
    if user is None:
        return JSONResponse(status_code=401, content={"error": "请先登录"})
    conn = _shop_conn()
    try:
        result = await asyncio.to_thread(_do_checkout, conn, user["id"], req.items)
    finally:
        conn.close()
    return result


class PayRequest(BaseModel):
    """POST /api/pay 的请求体。"""
    order_id: int | None = None       # 单笔支付
    order_ids: list[int] | None = None  # 购物车多笔合并支付
    method: str = "mock"              # 支付方式：mock=开发模拟；wechat/alipay=生产（待接入）


def _shop_conn():
    """连商城库（agent_orders 表所在库）。"""
    import pymysql
    return pymysql.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        port=int(os.getenv("MYSQL_PORT", "3306")),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "change_me_to_your_password"),
        database=os.getenv("MYSQL_SHOP_DB", "flower_shop_db"),
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )


def _process_payment(conn, order_ids: list[int], user_id: int, method: str) -> dict:
    """同步的支付处理（在线程池里跑），支持单笔或多笔合并支付。

    【开发模式 mock】直接把订单状态改成"已支付"，不产生真实交易。
    【生产】method=wechat/alipay 时应在这里：
        1. 调用微信/支付宝"统一下单"接口，生成预支付交易单
        2. 返回 prepay_id / 支付二维码链接给前端拉起收银台
        3. 用户付完后由支付平台异步回调 notify 接口，再更新订单状态
       绝不能在前端点"支付"时就直接标已支付（现在 mock 才这么做）。
    """
    if not order_ids:
        return {"ok": False, "message": "缺少要支付的订单"}
    with conn.cursor() as cur:
        # 只能支付自己的、当前还是"待支付"的订单（防越权 + 防重复支付）
        placeholders = ",".join(["%s"] * len(order_ids))
        cur.execute(
            f"SELECT id, total_price, status FROM agent_orders "
            f"WHERE user_id=%s AND id IN ({placeholders})",
            (user_id, *order_ids))
        orders = cur.fetchall()
        found = {o["id"] for o in orders}
        missing = [oid for oid in order_ids if oid not in found]
        if missing:
            return {"ok": False, "message": f"订单 {missing} 不存在或不属于当前用户"}
        pending = [o for o in orders if o["status"] == "待支付"]
        if not pending:
            return {"ok": False, "message": "这些订单都已支付，无需重复支付"}
        amount = round(sum(float(o["total_price"]) for o in pending), 2)
        pay_ids = [o["id"] for o in pending]

        if method == "mock":
            cur.execute(
                f"UPDATE agent_orders SET status='已支付' WHERE id IN ({placeholders})",
                pay_ids)
            conn.commit()
            return {"ok": True, "paid": True, "order_ids": pay_ids,
                    "amount": str(amount),
                    "message": f"模拟支付成功 ¥{amount}（开发模式，未产生真实扣款）"}
        # 生产支付通道：接入后在这里返回拉起支付所需参数
        return {"ok": False,
                "message": f"{method} 支付通道尚未接入，请先用模拟支付体验完整流程"}


@app.post("/api/pay")
async def pay(req: PayRequest, authorization: str | None = Header(default=None)):
    """结算支付：把待支付订单标记为已支付（开发模式为模拟支付）。需登录。"""
    user = await auth.get_user_info(_extract_token(authorization))
    if user is None:
        return JSONResponse(status_code=401, content={"error": "请先登录"})
    ids = req.order_ids or ([req.order_id] if req.order_id else [])
    conn = _shop_conn()
    try:
        result = await asyncio.to_thread(_process_payment, conn, ids, user["id"], req.method)
    finally:
        conn.close()
    return result


def _sse(data: dict) -> str:
    """把字典包装成一条 SSE 帧：'data: {...}\n\n' 是 SSE 协议的固定格式。"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# 图节点名 → 前端展示的"处理者"标签（让用户看到当前是哪位"专家"在干活）
AGENT_LABELS = {
    "orchestrator": "🧭 正在分析你的问题…",
    "product_agent": "🌸 商品知识顾问",
    "billing_agent": "💰 订单与商品专员",
    "finops_agent": "📊 成本优化专家",
    "order_agent": "🛒 下单助手",
    "florist_agent": "💐 配花师",
}


def _tool_friendly(tool_name: str, raw_input) -> str:
    """把工具调用翻译成用户能看懂的"正在做什么"，不暴露函数名和参数。"""
    if tool_name == "search_shop_goods":
        kw = (raw_input or {}).get("keyword", "") if isinstance(raw_input, dict) else ""
        return f"正在搜索「{kw}」…" if kw else "正在搜索商品…"
    if tool_name == "query_user_orders":
        return "正在查询进货订单…"
    if tool_name == "get_order_statistics":
        return "正在统计消费情况…"
    if tool_name == "query_user_mall_orders":
        return "正在查询商城订单…"
    if tool_name == "query_user_instances":
        return "正在查询资源实例…"
    if tool_name == "analyze_instance_usage":
        return "正在分析资源使用…"
    if tool_name == "query_vector_db":
        return "正在检索知识库…"
    if tool_name == "query_knowledge_graph":
        return "正在查询知识图谱…"
    if tool_name == "create_agent_order":
        return "正在创建订单…"
    if tool_name == "submit_bouquet":
        return "正在按库存和价格校验配花方案…"
    return "正在处理…"


def _parse_shop_goods(output: str) -> list:
    """解析 search_shop_goods 的文本输出，提取结构化商品（花名+价格+图片配对）。

    输入格式示例：
        商城「卡罗拉」相关在售商品（按销量排序，共 3 件）：
        1. 卡罗拉，售价 ¥12.00，库存 231，已售 129，图片：https://...
    输出：[{name, price, stock, sales, image}, ...]
    """
    items = []
    # 逐行匹配 "数字. 花名，售价 ¥价格，库存 X，已售 Y，图片：URL"
    pattern = re.compile(
        r"\d+\.\s*(.+?)，售价\s*¥([\d.]+)，库存\s*(\d+)，已售\s*(\d+)，图片：(https?://\S+)"
    )
    for line in output.split("\n"):
        m = pattern.search(line)
        if m:
            items.append({
                "name": m.group(1).strip(),
                "price": m.group(2),
                "stock": m.group(3),
                "sales": m.group(4),
                "image": m.group(5).strip(),
            })
    return items


async def _chat_stream(req: ChatRequest, user_id: str):
    """真正的流式生成器：逐 token 推给前端。

    user_id 已经在外面通过 token 验证过，这里直接信任。
    """
    session_id = req.session_id or f"session_{uuid.uuid4().hex[:8]}"

    # --- 第 1 步：读历史记忆，拼上下文（和 main.py 完全一样的逻辑）---
    history = await memory.get_messages(user_id, session_id)
    memory_context = "\n".join(f"{m['role']}: {m['content']}" for m in history)

    state = {
        "messages": [HumanMessage(content=req.message)],
        "user_id": user_id,
        "session_id": session_id,
        "memory_context": memory_context,
        "next_agent": "",
        "metadata": {},
        "image_url": req.image_url or "",   # 本轮图片 URL（vision_agent 消费）
    }

    # 先告诉前端会话 ID（前端要保存，下轮带来）
    yield _sse({"type": "meta", "session_id": session_id})

    full_reply = ""   # 累积完整回复，流结束后存 Redis
    current_node = ""  # 当前处理节点（去重用：同一节点的 start 事件只推一次）
    image_urls = []    # 工具查到的商品图 URL（去重收集，end 前统一发给前端）
    products = []      # 结构化商品列表：[{name, price, stock, sales, image}]，用于花名+图片配对渲染
    bouquet = None     # 配花方案（submit_bouquet 校验后的 JSON），存在时优先渲染方案卡片
    thinking_buf = ""  # 累积当前"思考步骤"的推理文本（非最终回答的 LLM 输出）
    try:
        # --- 第 2 步：流式执行工作流 ---
        async for event in graph.astream_events(state, version="v2"):
            ev = event["event"]

            # ---------- ① 节点开始：显示当前哪位专家在干活 ----------
            if ev == "on_chain_start":
                node = event.get("name", "")
                if node in AGENT_LABELS and node != current_node:
                    current_node = node
                    yield _sse({"type": "agent", "name": AGENT_LABELS[node]})
                continue

            # ---------- ② 工具开始调用：只推"正在查询"状态，不暴露函数细节 ----------
            if ev == "on_tool_start":
                tool_name = event.get("name", "unknown_tool")
                # 不把原始入参推给前端，只给一个友好的"正在做什么"提示
                friendly = _tool_friendly(tool_name, (event.get("data") or {}).get("input", {}))
                yield _sse({"type": "tool_start", "name": tool_name, "label": friendly})
                if thinking_buf.strip():
                    yield _sse({"type": "thinking", "content": thinking_buf.strip()})
                    thinking_buf = ""
                continue

            # ---------- ③ 工具执行完毕：推结构化商品数据 + 友好结果摘要 ----------
            if ev == "on_tool_end":
                tool_name = event.get("name", "unknown_tool")
                output = (event.get("data") or {}).get("output", "")
                content = getattr(output, "content", output)
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            parts.append(block)
                    output_str = "\n".join(parts)
                else:
                    output_str = str(content)

                # search_shop_goods：解析出结构化商品（花名+价格+图片配对）
                if tool_name == "search_shop_goods":
                    parsed = _parse_shop_goods(output_str)
                    products.extend(parsed)
                    for p in parsed:
                        if p["image"] and p["image"] not in image_urls:
                            image_urls.append(p["image"])
                    # 给前端一个简短摘要，不暴露原始输出
                    summary = f"查到 {len(parsed)} 款在售商品" if parsed else "未找到相关商品"
                    yield _sse({"type": "tool_end", "name": tool_name, "summary": summary})
                elif tool_name == "submit_bouquet":
                    # 配花方案：工具返回的是校验后的 JSON，立刻推给前端渲染方案卡片
                    try:
                        bouquet_data = json.loads(output_str)
                        bouquet = bouquet_data   # 有方案后，结尾不再重复推零散商品卡片
                        yield _sse({"type": "bouquet", "data": bouquet_data})
                        n = len(bouquet_data.get("items", []))
                        yield _sse({"type": "tool_end", "name": tool_name,
                                    "summary": f"配花方案已生成（{n} 种花材，合计 ¥{bouquet_data.get('total')}）"})
                    except (json.JSONDecodeError, ValueError):
                        yield _sse({"type": "tool_end", "name": tool_name, "summary": "方案调整中…"})
                else:
                    # 其他工具：只给一个"完成"提示，不推原始数据
                    yield _sse({"type": "tool_end", "name": tool_name, "summary": "查询完成"})
                continue

            if ev != "on_chat_model_stream":
                continue

            # ---------- ④ LLM token 流：区分"思考"和"最终回答" ----------
            node = (event.get("metadata") or {}).get("langgraph_node", "")
            # 只看 create_react_agent 内部的 "agent" 节点（orchestrator 的路由独白不给用户看）
            if node != "agent":
                continue

            chunk = event["data"]["chunk"]
            text = getattr(chunk, "content", "") or ""
            tool_calls = getattr(chunk, "tool_calls", None)

            # 判断这个 chunk 是"推理步骤"还是"最终回答"：
            # - chunk 带 tool_calls → LLM 正在决定调哪个工具，属于思考过程
            # - 不带 tool_calls 且有文字 → LLM 在组织最终回答
            if tool_calls:
                # 推理步骤的文字（有些模型会输出"我需要搜索一下"这类内心独白）
                if text:
                    thinking_buf += text
            else:
                if not text:
                    continue
                # 最终回答：先把攒下的思考文本推出去，再推回答 token
                if thinking_buf.strip():
                    yield _sse({"type": "thinking", "content": thinking_buf.strip()})
                    thinking_buf = ""
                full_reply += text
                yield _sse({"type": "token", "content": text})

        # --- 第 3 步：把本轮问答存进 Redis（流结束后才存，保证完整）---
        history.append({"role": "user", "content": req.message})
        history.append({"role": "assistant", "content": full_reply})
        await memory.save_messages(user_id, session_id, history)

        # 工具查到的商品：end 前统一发结构化数据（花名+价格+图片配对）
        # 但本轮已经生成了配花方案卡片时，不再补发零散商品，避免重复
        if not bouquet and products:
            yield _sse({"type": "products", "items": products[:8]})
        elif not bouquet and image_urls:
            # 兜底：只有图片 URL 没有结构化数据时，降级发图片列表
            yield _sse({"type": "images", "urls": image_urls[:8]})

        yield _sse({"type": "end"})
    except Exception as exc:
        yield _sse({"type": "error", "message": str(exc)})


@app.post("/chat")
async def chat(
    req: ChatRequest,
    authorization: str | None = Header(default=None),
):
    """聊天接口：SSE 流式响应。必须先登录，请求头带 Authorization: Bearer {token}。"""
    if not req.message.strip():
        return {"error": "消息不能为空"}
    # 必须在开启 SSE 流之前验证 token：验证失败直接回 401，前端跳登录
    token = _extract_token(authorization)
    user_id = await auth.get_user_id(token)
    if user_id is None:
        return JSONResponse(status_code=401, content={"error": "未登录或登录已过期，请重新登录"})
    return StreamingResponse(
        _chat_stream(req, str(user_id)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    """健康检查：PHP/运维探活用。"""
    return {"status": "ok", "memory": memory.available if memory else False}


# === 内置网页测试客户端（GET / 直接返回一个能聊天的页面）===
# 好处：浏览器打开 http://localhost:8100 就是同源请求，连 CORS 都不用配
# 前端设计：Agent 风格 = 气泡对话 + 处理者状态条（当前哪个专家在干活）+ 建议问题
TEST_PAGE = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>花小满 · AI 智能客服</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:"PingFang SC","Microsoft YaHei",sans-serif;
     background:linear-gradient(160deg,#fdf2f6 0%,#f5f0ff 100%);height:100vh;
     display:flex;justify-content:center;align-items:center;gap:24px;padding:0 24px}
/* ---------- 外壳 ---------- */
#app{width:420px;height:88vh;min-width:320px;background:#fff;
     display:flex;flex-direction:column;overflow:hidden}
/* ---------- 顶栏 ---------- */
#head{padding:14px 18px;background:linear-gradient(90deg,#e75480,#b56ac9);
      color:#fff;display:flex;align-items:center;gap:10px}
#head .avatar{width:38px;height:38px;border-radius:50%;background:#fff3;
      display:flex;align-items:center;justify-content:center;font-size:22px}
#head .info{flex:1;min-width:0}
#head .info b{font-size:16px;white-space:nowrap}
#head .info .st{font-size:11px;opacity:.85;display:flex;align-items:center;gap:4px;
                white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#head .info .st i{width:7px;height:7px;border-radius:50%;background:#7CFC9B;
      display:inline-block;animation:pulse 1.6s infinite}
@keyframes pulse{50%{opacity:.4}}
#newbtn{background:#fff2;border:1px solid #fff5;color:#fff;border-radius:16px;
        padding:5px 12px;font-size:12px;cursor:pointer}
#newbtn:hover{background:#fff4}
/* ---------- 消息区 ---------- */
#chat{flex:1;overflow-y:auto;padding:16px 14px;display:flex;flex-direction:column;gap:14px}
.row{display:flex;gap:8px;align-items:flex-end}
.row.me{flex-direction:row-reverse}
.bubble{max-width:78%;padding:10px 13px;border-radius:14px;font-size:14px;line-height:1.65;
        white-space:pre-wrap;word-break:break-word}
.row.ai .bubble{background:#f7f4fa;border-bottom-left-radius:4px}
.row.me .bubble{background:linear-gradient(135deg,#e75480,#c86dd7);color:#fff;
        border-bottom-right-radius:4px}
.aicon{width:30px;height:30px;border-radius:50%;background:#fbe3ee;flex-shrink:0;
       display:flex;align-items:center;justify-content:center;font-size:16px}
/* 处理者状态条：显示当前哪个 Agent 在干活 */
#agentbar{display:none;align-self:flex-start;font-size:12px;color:#a05;font-weight:600;
          background:#fdeef4;border-radius:12px;padding:4px 12px;
          animation:fadein .3s}
#agentbar .dots::after{content:"…";animation:blink 1s steps(3) infinite}
@keyframes blink{50%{opacity:.2}}
@keyframes fadein{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
/* 打字中三个跳动圆点 */
.typing{display:inline-flex;gap:4px;padding:4px 2px}
.typing i{width:6px;height:6px;border-radius:50%;background:#c9a;animation:jump 1s infinite}
.typing i:nth-child(2){animation-delay:.15s}.typing i:nth-child(3){animation-delay:.3s}
@keyframes jump{40%{transform:translateY(-5px)}}
/* 光标 */
.cursor{display:inline-block;width:2px;height:14px;background:#e75480;
        animation:blink .8s steps(1) infinite;vertical-align:middle}
/* ---------- 欢迎区建议问题 ---------- */
#welcome{align-self:flex-start;max-width:88%}
#welcome .chips{display:flex;flex-wrap:wrap;gap:8px;margin-top:10px}
.chip{font-size:12.5px;color:#b03a76;background:#fdf0f5;border:1px solid #f5cfe0;
      border-radius:16px;padding:6px 12px;cursor:pointer;transition:.15s}
.chip:hover{background:#e75480;color:#fff;border-color:#e75480}
/* ---------- 输入区 ---------- */
#bar{display:flex;gap:8px;padding:12px;border-top:1px solid #f0e3ec;background:#fff}
#cam{border:1.5px solid #ecd9e6;background:#fff;border-radius:12px;width:44px;
     font-size:18px;cursor:pointer;flex-shrink:0}
#cam.hasfile{background:#e75480;color:#fff;border-color:#e75480}
#inp{flex:1;border:1.5px solid #ecd9e6;border-radius:20px;padding:10px 16px;
     font-size:14px;outline:none}
#inp:focus{border-color:#e75480}
#send{border:none;border-radius:20px;padding:10px 20px;font-size:14px;cursor:pointer;
      background:linear-gradient(135deg,#e75480,#c86dd7);color:#fff;font-weight:600}
#send:disabled{opacity:.5;cursor:not-allowed}
/* 商品图片缩略图行（回答下方） */
.imgs{display:flex;flex-wrap:wrap;gap:8px}
.imgs img{width:86px;height:86px;object-fit:cover;border-radius:10px;cursor:pointer;
          border:1px solid #f0dbe8}
/* 用户上传的图（气泡内预览） */
.upimg{max-width:180px;max-height:140px;border-radius:10px;display:block;margin-top:6px}
/* ---------- Agent 专属：工具状态条（极简，不暴露函数细节） ---------- */
.toolstatus{align-self:flex-start;display:flex;align-items:center;gap:6px;font-size:12px;
             color:#7a3a5c;background:#fdeef4;border-radius:10px;padding:5px 12px;animation:fadein .25s}
.toolstatus .spin{width:10px;height:10px;border:2px solid #e75480;border-top-color:transparent;
                  border-radius:50%;animation:rot .7s linear infinite}
@keyframes rot{to{transform:rotate(360deg)}}
.toolstatus.done .spin{display:none}
.toolstatus.done{background:#e8f5e9;color:#2d7a4f}
/* ---------- Agent 专属：思考气泡（推理过程） ---------- */
.thinking{align-self:flex-start;max-width:86%;font-size:12px;color:#999;font-style:italic;
          background:#f7f4fa;border-radius:10px;padding:6px 12px;border-left:3px solid #d4b5cf}
/* ---------- Agent 专属：执行时间线（顶部步骤条） ---------- */
.trace{align-self:flex-start;display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px}
.trace .step{font-size:10.5px;background:#f3eef7;color:#8a6a9a;border-radius:10px;
             padding:3px 9px;white-space:nowrap}
.trace .step.done{background:#e6f4ea;color:#2d7a4f}
.trace .step.active{background:#e75480;color:#fff;animation:pulse 1s infinite}
/* ---------- 商品卡片：花名+价格+图片配对展示 ---------- */
.products{display:flex;flex-direction:column;align-self:flex-start;max-width:90%}
.pcard{display:flex;gap:10px;padding:8px 0;border-bottom:1px solid #f0e3ec;
       align-items:center;cursor:pointer;transition:.15s}
.pcard:hover{opacity:.7}
.pcard img{width:48px;height:48px;object-fit:cover;border-radius:6px;flex-shrink:0}
.pcard .info{flex:1;min-width:0}
.pcard .pname{font-size:13px;font-weight:600;color:#333;white-space:nowrap;
              overflow:hidden;text-overflow:ellipsis}
.pcard .pprice{font-size:13px;color:#e75480;font-weight:700;margin-top:1px}
.pcard .pmeta{font-size:11px;color:#999;margin-top:1px}
.paddbtn{border:1px solid #e75480;background:#fff;color:#e75480;border-radius:14px;
         padding:4px 12px;font-size:12px;cursor:pointer;flex-shrink:0}
.paddbtn:active{background:#fde7f0}
/* ---------- 配花方案卡片 ---------- */
.bouquet{align-self:flex-start;width:88%;max-width:420px;background:#fffafd;
         border:1px solid #f0cfe0;border-radius:14px;padding:14px 16px}
.bqhead{display:flex;align-items:center;gap:6px;font-size:14px;font-weight:700;
        color:#7a3a5c;margin-bottom:2px}
.bqshare{margin-left:auto;border:1px solid #ecc3da;background:#fff;color:#c86dd7;
         border-radius:14px;padding:3px 11px;font-size:11px;font-weight:600;cursor:pointer}
.bqshare:active{background:#fdf2f7}
.bqoccasion{font-size:11px;color:#b598aa;margin-bottom:10px}
.bqrow{display:flex;gap:10px;padding:9px 0;border-bottom:1px solid #f6ecf3;align-items:center}
.bqrow img{width:46px;height:46px;object-fit:cover;border-radius:6px;flex-shrink:0;background:#f6ecf3}
.bqrow .bqi{flex:1;min-width:0}
.bqrow .bqn{font-size:13px;font-weight:600;color:#333;
            white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bqrow .bqntext{}.bqnname{}
.bqrow .bqnote{font-size:10.5px;color:#a98aa0;margin-top:2px;
               white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.bqrow .bqprice{font-size:11px;color:#e75480;margin-top:2px}
.bqstepper{display:flex;align-items:center;gap:7px;flex-shrink:0}
.bqstepper button{width:22px;height:22px;border:1px solid #e0c9d8;background:#fff;
                  border-radius:50%;cursor:pointer;font-size:13px;line-height:1;color:#7a3a5c}
.bqstepper .bqq{min-width:20px;text-align:center;font-size:13px}
.bqfoot{display:flex;justify-content:space-between;align-items:center;padding:12px 0 4px}
.bqtotal{font-size:15px}
.bqtotal b{color:#e75480;font-size:20px}
.bqbudget{font-size:10.5px;color:#b598aa;margin-left:6px}
.bqbudget.over{color:#e07b39}
.bqbtns{display:flex;gap:8px;margin-top:10px}
.bqbtn{flex:1;border:none;border-radius:20px;padding:10px;font-size:13px;cursor:pointer;font-weight:600}
.bqbtn.outline{background:#fff;color:#e75480;border:1.5px solid #e75480}
.bqbtn.solid{background:linear-gradient(135deg,#e75480,#c86dd7);color:#fff}
/* ---------- 订单确认卡片 ---------- */
.ordercard{align-self:flex-start;max-width:90%;background:linear-gradient(135deg,#fff5f8,#f3eef7);
           border:1px solid #f0cfe0;border-radius:14px;padding:14px}
.ordercard .otitle{font-size:14px;font-weight:700;color:#7a3a5c;margin-bottom:8px}
.ordercard .orow{display:flex;justify-content:space-between;font-size:13px;padding:3px 0;color:#555}
.ordercard .ototal{font-size:18px;font-weight:700;color:#e75480;margin-top:8px;padding-top:8px;
                   border-top:1px dashed #e0c0d5}
.ordercard .obtns{display:flex;gap:8px;margin-top:12px}
.ordercard .obtn{flex:1;border:none;border-radius:18px;padding:8px;font-size:13px;
                 cursor:pointer;font-weight:600}
.ordercard .obtn.confirm{background:linear-gradient(135deg,#e75480,#c86dd7);color:#fff}
.ordercard .obtn.cancel{background:#fff;color:#888;border:1px solid #ddd}
/* ---------- 右侧侧边栏（购物车/推荐/留言） ---------- */
#sidebar{height:88vh;width:340px;background:#fff;
         display:flex;flex-direction:column;overflow:hidden}
#sidehead{padding:14px 16px;background:linear-gradient(90deg,#e75480,#b56ac9);color:#fff;
           display:flex;align-items:center;justify-content:space-between}
#sidehead b{font-size:15px}
#sideclose{display:none}
#sidetabs{display:flex;border-bottom:1px solid #f0e3ec}
#sidetabs .tab{flex:1;padding:11px 0;text-align:center;font-size:13px;color:#888;
               cursor:pointer;border-bottom:2px solid transparent;transition:.15s}
#sidetabs .tab.active{color:#e75480;border-bottom-color:#e75480;font-weight:600}
#carttabcount{background:#ff4757;color:#fff;font-size:10px;border-radius:8px;
              padding:1px 5px;margin-left:4px;display:none}
#carttabcount.show{display:inline}
#sidebody{flex:1;overflow-y:auto;padding:12px}
.sidepage{display:none}
.sidepage.active{display:block}
/* 侧边栏打开按钮已隐藏（常驻显示不需要） */
#sideopen{display:none}
/* 购物车 */
.cartitem{display:flex;gap:10px;padding:10px 0;
          border-bottom:1px solid #f0e3ec;align-items:center}
.cartitem img{width:48px;height:48px;object-fit:cover;border-radius:6px}
.cartitem .cinfo{flex:1;min-width:0}
.cartitem .cname{font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.cartitem .cprice{font-size:13px;color:#e75480;font-weight:700}
.cartitem .cqty{display:flex;align-items:center;gap:6px;margin-top:4px}
.cartitem .cqty button{width:22px;height:22px;border:1px solid #e0c0d5;background:#fff;
                       border-radius:4px;cursor:pointer;font-size:14px;color:#7a3a5c}
.cartitem .cqty span{font-size:12px;min-width:16px;text-align:center}
.cartitem .cdel{background:none;border:none;color:#ccc;font-size:16px;cursor:pointer}
.cartitem .cdel:hover{color:#e75480}
.cartempty{text-align:center;color:#bbb;padding:40px 0;font-size:13px}
.carttotal{display:flex;justify-content:space-between;align-items:center;padding:12px 0;
           border-bottom:1px solid #f0e3ec;margin-bottom:10px}
.carttotal .lbl{font-size:13px;color:#7a3a5c}
.carttotal .amt{font-size:20px;color:#e75480;font-weight:700}
.cartcheckout{width:100%;border:none;border-radius:20px;padding:11px;font-size:14px;
              cursor:pointer;background:linear-gradient(135deg,#e75480,#c86dd7);color:#fff;font-weight:600}
.cartcheckout:disabled{opacity:.5;cursor:not-allowed}
/* 推荐页 */
.recitem{display:flex;gap:10px;padding:10px 0;
         border-bottom:1px solid #f0e3ec;align-items:center;cursor:pointer;transition:.15s}
.recitem:hover{opacity:.7}
.recitem img{width:48px;height:48px;object-fit:cover;border-radius:6px}
.recitem .rname{font-size:13px;font-weight:600}
.recitem .rreason{font-size:10px;color:#c86dd7;margin-top:3px}
.recitem .rprice{font-size:13px;color:#e75480;font-weight:700;margin-top:2px}
/* 留言页 */
.msgform{display:flex;flex-direction:column;gap:8px;margin-bottom:14px}
.msgform textarea{border:1px solid #ecd9e6;border-radius:10px;padding:10px;font-size:13px;
                  resize:none;height:70px;outline:none;font-family:inherit}
.msgform textarea:focus{border-color:#e75480}
.msgform button{align-self:flex-end;border:none;border-radius:16px;padding:7px 18px;
                font-size:13px;cursor:pointer;background:linear-gradient(135deg,#e75480,#c86dd7);
                color:#fff;font-weight:600}
.msglist .msgitem{padding:10px 0;border-bottom:1px solid #f0e3ec;margin-bottom:0}
.msglist .mhead{display:flex;justify-content:space-between;font-size:11px;color:#999;margin-bottom:4px}
.msglist .mcontent{font-size:13px;color:#333;line-height:1.6;word-break:break-word}
.msgempty{text-align:center;color:#bbb;padding:30px 0;font-size:13px}
/* toast 提示 */
.toast{position:fixed;top:20px;left:50%;transform:translateX(-50%);background:rgba(0,0,0,.75);
       color:#fff;padding:8px 18px;border-radius:18px;font-size:13px;z-index:200;
       animation:toastin .25s}
@keyframes toastin{from{opacity:0;transform:translate(-50%,-10px)}to{opacity:1;transform:translate(-50%,0)}}
/* ---------- 登录遮罩（简洁表单，无卡片边框） ---------- */
#loginmask{position:fixed;inset:0;z-index:300;
           background:linear-gradient(160deg,#fdf2f6 0%,#f5f0ff 100%);
           display:flex;align-items:center;justify-content:center}
#loginmask.hide{display:none}
.loginbox{width:300px}
.loginlogo{font-size:40px;text-align:center;margin-bottom:6px}
.logintitle{text-align:center;font-size:20px;font-weight:700;color:#7a3a5c;margin-bottom:4px}
.loginsub{text-align:center;font-size:12px;color:#aa8da0;margin-bottom:26px}
.loginbox input{width:100%;border:none;border-bottom:1.5px solid #e0c9d8;
                padding:11px 2px;font-size:14px;outline:none;background:transparent;margin-bottom:20px}
.loginbox input:focus{border-bottom-color:#e75480}
.coderow{display:flex;gap:10px;align-items:flex-start}
.coderow input{flex:1}
.codebtn{border:none;background:none;color:#e75480;font-size:13px;cursor:pointer;
         padding:11px 0 0;white-space:nowrap;font-weight:600}
.codebtn:disabled{color:#c9b3c2;cursor:not-allowed}
.loginbtn{width:100%;border:none;border-radius:22px;padding:12px;font-size:15px;
          cursor:pointer;background:linear-gradient(135deg,#e75480,#c86dd7);color:#fff;
          font-weight:600;margin-top:10px}
.loginbtn:disabled{opacity:.6;cursor:not-allowed}
.loginhint{font-size:11px;color:#b598aa;text-align:center;margin-top:14px;line-height:1.7}
.loginhint b{color:#e75480}
.loginerr{color:#e74c3c;font-size:12px;text-align:center;min-height:16px;margin-bottom:8px}
/* 顶栏用户区 */
#userinfo{display:flex;align-items:center;gap:8px;font-size:12px;opacity:.92;flex-shrink:0}
#userinfo button{white-space:nowrap}
#logoutbtn{background:#fff2;border:1px solid #fff5;color:#fff;border-radius:14px;
           padding:5px 12px;font-size:12px;cursor:pointer}
/* 推荐标题 */
.rectitle{font-size:13px;font-weight:700;color:#7a3a5c;padding:2px 2px 10px}
/* ---------- 结算支付弹窗 ---------- */
#paymask{position:fixed;inset:0;z-index:400;background:rgba(60,30,50,.45);
         display:none;align-items:center;justify-content:center}
#paymask.show{display:flex}
.paybox{width:340px;max-height:86vh;overflow:auto;background:#fff;border-radius:16px}
.payhead{display:flex;justify-content:space-between;align-items:center;
         padding:16px 18px;font-size:16px;font-weight:700;color:#5a2a44}
.payclose{cursor:pointer;font-size:22px;color:#b598aa;line-height:1}
.paylist{padding:0 18px;max-height:240px;overflow:auto}
.payrow{display:flex;gap:10px;align-items:center;padding:10px 0;
        border-bottom:1px solid #f6ecf3}
.payrow img{width:42px;height:42px;object-fit:cover;border-radius:6px;background:#f6ecf3}
.payrow .pn{flex:1;font-size:13px}
.payrow .pq{font-size:12px;color:#a98aa0}
.payrow .pp{font-size:13px;color:#e75480;font-weight:600;white-space:nowrap}
.paytotal{display:flex;justify-content:space-between;align-items:center;
          padding:14px 18px;font-size:14px}
.paytotal b{font-size:20px;color:#e75480}
.paymethods{padding:4px 18px 14px}
.paym{display:flex;align-items:center;gap:10px;padding:11px 12px;margin-bottom:8px;
      border:1.5px solid #ecdde8;border-radius:10px;cursor:pointer;font-size:13px}
.paym small{margin-left:auto;color:#b598aa;font-size:11px}
.paym.active{border-color:#e75480;background:#fdf2f7}
.pmicon{font-size:18px}
.paysubmit{display:block;width:calc(100% - 36px);margin:0 18px 18px;border:none;border-radius:22px;
           padding:13px;font-size:15px;cursor:pointer;font-weight:600;
           background:linear-gradient(135deg,#e75480,#c86dd7);color:#fff}
.paysubmit:disabled{opacity:.6}
</style></head><body>
<!-- 登录遮罩：未登录时全屏覆盖 -->
<div id="loginmask">
  <div class="loginbox">
    <div class="loginlogo">🌸</div>
    <div class="logintitle">花小满</div>
    <div class="loginsub">手机号验证码登录，新用户自动注册</div>
    <div class="loginerr" id="loginerr"></div>
    <input id="phone" type="tel" maxlength="11" placeholder="请输入手机号"
           oninput="this.value=this.value.replace(/\\D/g,'')"/>
    <div class="coderow">
      <input id="smscode" type="tel" maxlength="6" placeholder="验证码"
             oninput="this.value=this.value.replace(/\\D/g,'')"/>
      <button class="codebtn" id="codebtn" onclick="sendCode()">获取验证码</button>
    </div>
    <button class="loginbtn" id="loginbtn" onclick="doLogin()">登 录</button>
    <div class="loginhint">未注册的手机号验证后将<b>自动注册</b><br>演示阶段验证码会显示在输入框旁</div>
  </div>
</div>
<div id="app">
  <div id="head">
    <div class="avatar">🌸</div>
    <div class="info"><b id="nicklabel"></b>
      <div class="st"><i></i>小花 AI 助手 · 在线</div></div>
    <div id="userinfo">
      <button id="newbtn" onclick="newSession()">＋ 新会话</button>
      <button id="logoutbtn" onclick="doLogout()">退出</button>
    </div>
  </div>
  <div id="chat">
    <div class="row ai" id="welcome">
      <div class="aicon">🌸</div>
      <div class="bubble">你好，我是小花 🌷<br>懂花材、会查价、能帮你查订单。<br>
        <div class="chips">
          <span class="chip" onclick="quick(this)">卡罗拉多少钱？</span>
          <span class="chip" onclick="quick(this)">店里有没有绣球？</span>
          <span class="chip" onclick="quick(this)">绣球有哪些在售？</span>
          <span class="chip" onclick="quick(this)">我的商城订单有哪些？</span>
        </div>
      </div>
    </div>
    <div id="agentbar"></div>
  </div>
  <div id="bar">
    <button id="cam" title="上传花卉图片，让小花识别" onclick="document.getElementById('file').click()">📷</button>
    <input type="file" id="file" accept="image/*" style="display:none" onchange="pickFile(this)"/>
    <input id="inp" placeholder="问问花价、发图识花…（回车发送）"/>
    <button id="send" onclick="send()">发送</button>
  </div>
</div>

<!-- 侧边栏打开按钮 -->
<button id="sideopen" onclick="toggleSidebar()">购物车 / 推荐 / 留言<span class="badge" id="cartbadge" style="display:none">0</span></button>

<!-- 右侧侧边栏：购物车 | 推荐 | 留言 -->
<div id="sidebar">
  <div id="sidehead"><b>花小满</b><button id="sideclose" onclick="toggleSidebar()">×</button></div>
  <div id="sidetabs">
    <div class="tab active" onclick="switchTab('cart')">🛒 购物车<span id="carttabcount"></span></div>
    <div class="tab" onclick="switchTab('rec')">✨ 推荐</div>
    <div class="tab" onclick="switchTab('msg')">💬 留言</div>
  </div>
  <div id="sidebody">
    <!-- 购物车 -->
    <div class="sidepage active" id="page-cart">
      <div id="cartlist"></div>
      <div id="cartfoot" style="display:none">
        <div class="carttotal"><span class="lbl">合计</span><span class="amt" id="cartamt">¥0.00</span></div>
        <button class="cartcheckout" onclick="checkoutCart()">去结算（下单）</button>
      </div>
    </div>
    <!-- 推荐 -->
    <div class="sidepage" id="page-rec">
      <div id="rectitle" class="rectitle">为你推荐</div>
      <div id="reclist"></div>
    </div>
    <!-- 留言 -->
    <div class="sidepage" id="page-msg">
      <div class="msgform">
        <textarea id="msginput" placeholder="想对小花说点什么？（建议、吐槽、需求都可以）"></textarea>
        <button onclick="submitMsg()">发表留言</button>
      </div>
      <div class="msglist" id="msglist"></div>
    </div>
  </div>
</div>

<!-- 结算 / 支付弹窗 -->
<div id="paymask">
  <div class="paybox">
    <div class="payhead">
      <span id="paytitle">确认订单</span>
      <span class="payclose" onclick="closePay()">×</span>
    </div>
    <!-- 第一步：订单确认 -->
    <div id="paystep1">
      <div class="paylist" id="paylist"></div>
      <div class="paytotal"><span>应付总额</span><b id="paytotal">¥0.00</b></div>
      <div class="paymethods">
        <label class="paym active" onclick="selectMethod(this,'mock')">
          <span class="pmicon">🧪</span><span>模拟支付</span><small>开发演示·不扣款</small></label>
        <label class="paym" onclick="selectMethod(this,'wechat')">
          <span class="pmicon">💚</span><span>微信支付</span><small>待接入</small></label>
        <label class="paym" onclick="selectMethod(this,'alipay')">
          <span class="pmicon">🔵</span><span>支付宝</span><small>待接入</small></label>
      </div>
      <button class="paysubmit" id="paysubmit" onclick="submitPay()">确认支付</button>
    </div>
    <!-- 第二步：支付结果 -->
    <div id="paystep2" style="display:none;text-align:center;padding:30px 20px">
      <div style="font-size:52px" id="payresulticon">✅</div>
      <div style="font-size:18px;font-weight:700;margin:14px 0 6px" id="payresulttitle">支付成功</div>
      <div style="font-size:13px;color:#9a7f92;line-height:1.7" id="payresultmsg"></div>
      <button class="paysubmit" style="margin-top:24px" onclick="closePay()">完成</button>
    </div>
  </div>
</div>

<script>
let sessionId=null, busy=false, selectedFile=null, currentTrace=null;
const chat=document.getElementById('chat'),inp=document.getElementById('inp'),
      sendBtn=document.getElementById('send'),agentbar=document.getElementById('agentbar'),
      cam=document.getElementById('cam');
function scrollEnd(){chat.scrollTop=chat.scrollHeight}

// 选完图：按钮变红提示"已带图"，气泡发送时会预览
function pickFile(input){
  selectedFile=input.files[0]||null;
  cam.classList.toggle('hasfile',!!selectedFile);
  cam.title=selectedFile?('将发送图片: '+selectedFile.name):'上传花卉图片，让小花识别';
}

// 追加一条消息气泡，AI 气泡返回正文元素（流式往里写）
function bubble(role,text){
  const row=document.createElement('div');row.className='row '+(role==='me'?'me':'ai');
  if(role==='ai'){const a=document.createElement('div');a.className='aicon';a.textContent='🌸';row.appendChild(a)}
  const b=document.createElement('div');b.className='bubble';b.textContent=text;
  row.appendChild(b);chat.appendChild(row);scrollEnd();return b;
}
// 打字动画（首 token 到达前）
function showTyping(){
  const row=document.createElement('div');row.className='row ai';row.id='typingRow';
  row.innerHTML='<div class="aicon">🌸</div><div class="bubble"><span class="typing"><i></i><i></i><i></i></span></div>';
  chat.appendChild(row);scrollEnd();
}
function hideTyping(){const t=document.getElementById('typingRow');if(t)t.remove()}

// ---- Agent 专属组件 ----
// 工具状态条：只显示"正在搜索XX…" / "✅ 查到3款商品"，不暴露函数和参数
function addToolStatus(label){
  const el=document.createElement('div');el.className='toolstatus';
  el.innerHTML=`<span class="spin"></span><span class="lbl"></span>`;
  el.querySelector('.lbl').textContent=label;
  chat.appendChild(el);scrollEnd();
  return el;
}
function finishToolStatus(el,summary){
  el.classList.add('done');
  el.querySelector('.spin')?.remove();
  el.querySelector('.lbl').textContent='✅ '+summary;
}
// 思考气泡：Agent 的推理过程
function addThinking(text){
  const el=document.createElement('div');el.className='thinking';
  el.textContent='💭 '+text;
  chat.appendChild(el);scrollEnd();
}
// 执行时间线
function addTrace(){
  const el=document.createElement('div');el.className='trace';
  chat.appendChild(el);currentTrace=el;return el;
}
function traceStep(label,done){
  if(!currentTrace)currentTrace=addTrace();
  currentTrace.querySelectorAll('.step.active').forEach(s=>{s.classList.remove('active');s.classList.add('done')});
  const step=document.createElement('span');step.className='step '+(done?'done':'active');
  step.textContent=label;currentTrace.appendChild(step);scrollEnd();
}
// 商品卡片：花名+价格+图片配对（识图/查价结果都走这里）
function renderProducts(items){
  const wrap=document.createElement('div');wrap.className='products';
  items.forEach(p=>{
    const card=document.createElement('div');card.className='pcard';
    card.innerHTML=`<img src="${p.image}" alt="" onerror="this.style.background='#f3eef7'"/>
      <div class="info">
        <div class="pname"></div>
        <div class="pprice">¥${p.price}</div>
        <div class="pmeta">库存 ${p.stock} · 已售 ${p.sales}</div>
      </div>
      <button class="paddbtn">加入购物车</button>`;
    card.querySelector('.pname').textContent=p.name;  // textContent 防 XSS
    // 点"加入购物车"按钮直接加购（拍照识花后的卡片也是同样入口）
    card.querySelector('.paddbtn').onclick=(e)=>{e.stopPropagation();addToCart(p);};
    wrap.appendChild(card);
  });
  chat.appendChild(wrap);scrollEnd();
}

// 配花方案卡片：可微调数量、一键全部加购、直接结算
function renderBouquet(d){
  const box=document.createElement('div');box.className='bouquet';
  const items=d.items.map(p=>Object.assign({},p,{qty:p.quantity}));
  const budget=parseFloat(d.budget)||0;
  box.innerHTML=`
    <div class="bqhead"><span>💐 配花方案</span>
      <button class="bqshare" title="分享方案">🔗 分享</button>
    </div>
    <div class="bqoccasion"></div>
    <div class="bqrows"></div>
    <div class="bqfoot">
      <span class="bqtotal">合计 <b class="bqamt">¥0.00</b>
        <span class="bqbudget"></span></span>
    </div>
    <div class="bqbtns">
      <button class="bqbtn outline bqadd">全部加入购物车</button>
      <button class="bqbtn solid bqpay">立即结算</button>
    </div>`;
  box.querySelector('.bqoccasion').textContent=d.occasion||'';
  const rowsEl=box.querySelector('.bqrows');
  const amtEl=box.querySelector('.bqamt'),budEl=box.querySelector('.bqbudget');

  function refresh(){
    let total=0;
    box.querySelectorAll('.bqrow').forEach((row,idx)=>{
      total+=parseFloat(items[idx].price)*items[idx].qty;
      row.querySelector('.bqq').textContent=items[idx].qty;
      row.querySelector('.bqsub').textContent='¥'+(parseFloat(items[idx].price)*items[idx].qty).toFixed(2);
    });
    amtEl.textContent='¥'+total.toFixed(2);
    if(budget>0){
      const over=total>budget+1e-6;
      budEl.textContent=over?`超出预算 ¥${(total-budget).toFixed(2)}`:`预算 ¥${budget.toFixed(0)} 内`;
      budEl.className='bqbudget'+(over?' over':'');
    }else{budEl.textContent='';}
  }
  items.forEach((p,idx)=>{
    const row=document.createElement('div');row.className='bqrow';
    row.innerHTML=`<img src="${p.image}" onerror="this.style.visibility='hidden'"/>
      <div class="bqi">
        <div class="bqn"></div>
        <div class="bqnote"></div>
        <div class="bqprice">¥${p.price}/件 · 小计 <span class="bqsub"></span></div>
      </div>
      <div class="bqstepper">
        <button class="bqminus">−</button><span class="bqq"></span><button class="bqplus">+</button>
      </div>`;
    row.querySelector('.bqn').textContent=p.name;
    row.querySelector('.bqnote').textContent=p.note||'';
    row.querySelector('.bqminus').onclick=()=>{if(items[idx].qty>1){items[idx].qty--;refresh();}};
    row.querySelector('.bqplus').onclick=()=>{
      if(items[idx].qty<(p.stock||9999)){items[idx].qty++;refresh();}
      else toast(`${p.name} 库存仅剩 ${p.stock}`);
    };
    rowsEl.appendChild(row);
  });
  refresh();

  // 把当前方案按数量合并进购物车（同名累加）
  function mergeIntoCart(){
    items.forEach(p=>{
      const ex=cart.find(c=>c.name===p.name);
      if(ex){ex.qty+=p.qty;}else{cart.push({name:p.name,price:p.price,image:p.image,qty:p.qty});}
    });
    saveCart();renderCart();
  }
  box.querySelector('.bqadd').onclick=()=>{mergeIntoCart();toast('方案已全部加入购物车 🛒');};
  box.querySelector('.bqpay').onclick=()=>{
    // 直接结算方案里的花（不改动购物车已有内容）
    checkoutCart(items.map(p=>({name:p.name,price:p.price,image:p.image,qty:p.qty})));
  };
  // 分享：用卡片上当前（可能微调过）的数量组文案
  box.querySelector('.bqshare').onclick=async()=>{
    const total=items.reduce((s,p)=>s+parseFloat(p.price)*p.qty,0);
    // 纯文本方案：微信/朋友圈/短信/备忘录里都能直接粘贴，排版也不乱
    const lines=[
      `💐 小花配花方案${d.occasion?` · ${d.occasion}`:''}`,
      '———————————',
    ];
    items.forEach(p=>{
      lines.push(`${p.name} ×${p.qty}　¥${(parseFloat(p.price)*p.qty).toFixed(2)}`);
      if(p.note)lines.push(`　└ ${p.note}`);
    });
    lines.push('———————————');
    lines.push(`合计 ¥${total.toFixed(2)}`);
    if(budget>0){
      lines.push(total<=budget+1e-6?`（预算 ¥${budget.toFixed(0)} 内）`:`（略超预算 ¥${budget.toFixed(0)}）`);
    }
    lines.push('来自「花小满」AI 配花师 🌸');
    const text=lines.join('\\n');
    // 手机端：navigator.share 直接调起微信/QQ 等系统分享面板
    if(navigator.share){
      try{
        await navigator.share({title:'我的专属配花方案',text});
        return;
      }catch(e){
        // 用户取消分享时浏览器会抛 AbortError，不算错误，静默返回
        if(e.name==='AbortError')return;
      }
    }
    // 桌面端或不支持原生分享：降级为复制到剪贴板
    try{
      await navigator.clipboard.writeText(text);
      toast('方案文案已复制，快去粘贴分享吧 🔗');
    }catch(e){
      // 老浏览器没有剪贴板 API：弹一个可全选的文本框兜底
      window.prompt('复制下面的方案文案：',text);
    }
  };
  chat.appendChild(box);scrollEnd();
}

async function send(text){
  if(busy)return;
  const msg=(text||inp.value).trim();if(!msg)return;
  inp.value='';busy=true;sendBtn.disabled=true;
  document.getElementById('welcome')?.remove();
  const myBubble=bubble('me',msg);
  // 带图发送：气泡里先显示本地预览，同时把图传到 /upload 换服务端 URL
  let imageUrl=null;
  if(selectedFile){
    const im=document.createElement('img');im.className='upimg';
    im.src=URL.createObjectURL(selectedFile);myBubble.appendChild(im);
    const fd=new FormData();fd.append('file',selectedFile);
    const up=await fetch('/upload',{method:'POST',body:fd,headers:authHeader()});
    if(up.status===401){forceLogout();return;}
    imageUrl=(await up.json()).url;
    selectedFile=null;cam.classList.remove('hasfile');document.getElementById('file').value='';
  }
  showTyping();
  // 每轮新建一个时间线（放在 AI 回答区最前面）
  addTrace();

  const body=JSON.stringify({message:msg,session_id:sessionId,image_url:imageUrl});
  const resp=await fetch('/chat',{method:'POST',
    headers:Object.assign({'Content-Type':'application/json'},authHeader()),body});
  if(resp.status===401){forceLogout();return;}
  const reader=resp.body.getReader(),dec=new TextDecoder();
  let buf='',answer=null,cur='';
  while(true){
    const{done,value}=await reader.read();if(done)break;
    buf+=dec.decode(value,{stream:true});
    const parts=buf.split('\\n\\n');buf=parts.pop();
    for(const p of parts){
      if(!p.startsWith('data: '))continue;
      const ev=JSON.parse(p.slice(6));
      if(ev.type==='meta')sessionId=ev.session_id;
      // ---- 处理者状态条 ----
      if(ev.type==='agent'){
        hideTyping();
        agentbar.style.display='block';
        agentbar.innerHTML=ev.name+'<span class="dots"></span>';
        // 时间线加一步：当前哪个专家上岗
        traceStep(ev.name.replace(/[🧭🌸💰📊]/g,'').trim());
        if(!answer){answer=bubble('ai','');}
        scrollEnd();
      }
      // ---- 思考过程（推理文本） ----
      if(ev.type==='thinking'){
        hideTyping();
        addThinking(ev.content);
      }
      // ---- 工具调用开始（只显示友好状态，不暴露函数） ----
      if(ev.type==='tool_start'){
        hideTyping();agentbar.style.display='none';
        traceStep(ev.label.replace(/[🔍📊🔧]/g,'').trim());
        addToolStatus(ev.label);
        if(!answer){answer=bubble('ai','');}
        scrollEnd();
      }
      // ---- 工具调用结束（显示摘要） ----
      if(ev.type==='tool_end'){
        // 找到最后一个未完成的 toolstatus 标记完成
        const statuses=chat.querySelectorAll('.toolstatus:not(.done)');
        if(statuses.length)finishToolStatus(statuses[statuses.length-1],ev.summary);
      }
      // ---- 最终回答 token ----
      if(ev.type==='token'){
        hideTyping();agentbar.style.display='none';
        if(!answer){answer=bubble('ai','');}
        cur+=ev.content;answer.textContent=cur;
        const c=document.createElement('span');c.className='cursor';
        answer.appendChild(c);scrollEnd();
      }
      // ---- 结构化商品卡片（花名+价格+图片配对） ----
      if(ev.type==='products'){
        renderProducts(ev.items);
      }
      // ---- 配花方案卡片 ----
      if(ev.type==='bouquet'){
        renderBouquet(ev.data);
      }
      // ---- 商品图（兜底，无结构化数据时） ----
      if(ev.type==='images'){
        const wrap=document.createElement('div');wrap.className='imgs';
        ev.urls.forEach(u=>{const im=document.createElement('img');im.src=u;
          im.onclick=()=>window.open(u);wrap.appendChild(im)});
        chat.appendChild(wrap);scrollEnd();
      }
      if(ev.type==='error'){answer=answer||bubble('ai','');answer.textContent+='⚠️ '+ev.message}
      // ---- 本轮结束 ----
      if(ev.type==='end'){
        hideTyping();agentbar.style.display='none';
        // 时间线最后一步标完成
        traceStep('完成',true);
        answer=answer||bubble('ai','（空回复）');
        answer.textContent=cur;
        busy=false;sendBtn.disabled=false;inp.focus();
      }
    }
  }
  busy=false;sendBtn.disabled=false;   // 兜底恢复
}
function quick(el){send(el.textContent)}
function newSession(){sessionId=null;location.reload()}
inp.addEventListener('keydown',e=>{if(e.key==='Enter')send()});

// ========== 侧边栏：购物车 / 推荐 / 留言 ==========
const CART_KEY='flower_cart', MSG_KEY='flower_msgs';
let cart=JSON.parse(localStorage.getItem(CART_KEY)||'[]');
let msgs=JSON.parse(localStorage.getItem(MSG_KEY)||'[]');

function toast(msg){
  const t=document.createElement('div');t.className='toast';t.textContent=msg;
  document.body.appendChild(t);setTimeout(()=>t.remove(),1800);
}
function toggleSidebar(){/* 侧边栏已常驻显示，无需切换 */}
function switchTab(name){
  document.querySelectorAll('#sidetabs .tab').forEach((t,i)=>{
    t.classList.toggle('active',['cart','rec','msg'][i]===name);
  });
  document.querySelectorAll('.sidepage').forEach(p=>p.classList.remove('active'));
  document.getElementById('page-'+name).classList.add('active');
  if(name==='rec')renderRec();
  if(name==='msg')renderMsgs();
}
// ---- 购物车 ----
function saveCart(){localStorage.setItem(CART_KEY,JSON.stringify(cart));updateCartBadge()}
function updateCartBadge(){
  const n=cart.reduce((s,i)=>s+i.qty,0);
  const b=document.getElementById('cartbadge');
  if(b){if(n>0){b.style.display='flex';b.textContent=n;}else{b.style.display='none';}}
  const tc=document.getElementById('carttabcount');
  if(tc){if(n>0){tc.textContent=n;tc.classList.add('show');}else{tc.classList.remove('show');}}
}
function addToCart(p){
  const ex=cart.find(i=>i.name===p.name&&i.price===p.price);
  if(ex){ex.qty++;}else{cart.push({name:p.name,price:p.price,image:p.image,qty:1});}
  saveCart();renderCart();toast(`已加入购物车：${p.name}`);
}
function changeQty(idx,d){
  cart[idx].qty+=d;
  if(cart[idx].qty<=0)cart.splice(idx,1);
  saveCart();renderCart();
}
function delCart(idx){cart.splice(idx,1);saveCart();renderCart()}
function renderCart(){
  const list=document.getElementById('cartlist');
  if(!cart.length){
    list.innerHTML='<div class="cartempty">🛒 购物车空空如也<br>去问问小花有什么花吧</div>';
    document.getElementById('cartfoot').style.display='none';return;
  }
  list.innerHTML='';
  let total=0;
  cart.forEach((it,idx)=>{
    total+=parseFloat(it.price)*it.qty;
    const el=document.createElement('div');el.className='cartitem';
    el.innerHTML=`<img src="${it.image}" onerror="this.style.background='#f3eef7'"/>
      <div class="cinfo"><div class="cname">${it.name}</div>
        <div class="cprice">¥${it.price}</div>
        <div class="cqty"><button onclick="changeQty(${idx},-1)">−</button>
          <span>${it.qty}</span><button onclick="changeQty(${idx},1)">+</button></div></div>
      <button class="cdel" onclick="delCart(${idx})">×</button>`;
    list.appendChild(el);
  });
  document.getElementById('cartfoot').style.display='block';
  document.getElementById('cartamt').textContent='¥'+total.toFixed(2);
}
// ---- 购物车结算 → 弹窗确认 → 支付 ----
let payMethod='mock', pendingOrderIds=[], checkoutScope=null;
async function checkoutCart(scopeItems){
  // scopeItems 不传=结算整个购物车；传入（如配花方案"立即结算"）=只结算这批
  const source=scopeItems||cart;
  checkoutScope=scopeItems?scopeItems.slice():null;
  if(!source.length)return;
  // 先展示明细（价格为预览，实际以后端核验为准）
  const list=document.getElementById('paylist');
  list.innerHTML=source.map(i=>`
    <div class="payrow">
      <img src="${i.image||''}" onerror="this.style.visibility='hidden'"/>
      <div class="pn">${i.name}<div class="pq">×${i.qty}</div></div>
      <div class="pp">¥${(parseFloat(i.price)*i.qty).toFixed(2)}</div>
    </div>`).join('');
  document.getElementById('paytotal').textContent=
    '¥'+source.reduce((s,i)=>s+parseFloat(i.price)*i.qty,0).toFixed(2);
  // 重置弹窗到第一步
  document.getElementById('paystep1').style.display='';
  document.getElementById('paystep2').style.display='none';
  document.getElementById('paysubmit').disabled=false;
  document.getElementById('paysubmit').textContent='确认支付';
  selectMethod(document.querySelector('.paym'),'mock');
  document.getElementById('paymask').classList.add('show');
}
function closePay(){document.getElementById('paymask').classList.remove('show');}
function selectMethod(el,m){
  payMethod=m;
  document.querySelectorAll('.paym').forEach(x=>x.classList.remove('active'));
  el.classList.add('active');
}
async function submitPay(){
  // 第一步：调结算接口，后端核验真实价格并建单
  const btn=document.getElementById('paysubmit');
  btn.disabled=true;btn.textContent='正在下单…';
  let r;
  try{
    const source=checkoutScope||cart;
    r=await fetch('/api/checkout',{method:'POST',
      headers:Object.assign({'Content-Type':'application/json'},authHeader()),
      body:JSON.stringify({items:source.map(i=>({name:i.name,quantity:i.qty}))})}).then(x=>x.json());
  }catch(e){r={ok:false,message:'网络错误'};}
  if(!r.ok){btn.disabled=false;btn.textContent='确认支付';toast(r.message||'下单失败');return;}
  pendingOrderIds=r.orders.map(o=>o.order_id);
  // 用后端返回的真实金额刷新弹窗
  document.getElementById('paytotal').textContent='¥'+r.total;
  // 第二步：发起支付
  btn.textContent='支付中…';
  let p;
  try{
    p=await fetch('/api/pay',{method:'POST',
      headers:Object.assign({'Content-Type':'application/json'},authHeader()),
      body:JSON.stringify({order_ids:pendingOrderIds,method:payMethod})}).then(x=>x.json());
  }catch(e){p={ok:false,message:'网络错误'};}
  showPayResult(p, r);
}
function showPayResult(p, checkout){
  document.getElementById('paystep1').style.display='none';
  document.getElementById('paystep2').style.display='';
  const icon=document.getElementById('payresulticon'),
        title=document.getElementById('payresulttitle'),
        msg=document.getElementById('payresultmsg');
  if(p.ok){
    icon.textContent='✅';title.textContent='支付成功';
    msg.textContent=p.message;
    // 支付成功：购物车结算才清空购物车；方案直接结算不动购物车
    if(!checkoutScope){cart=[];saveCart();renderCart();}
    checkoutScope=null;
  }else{
    icon.textContent='⚠️';title.textContent='支付未完成';
    msg.textContent=p.message||'支付失败';
  }
}
// ---- 推荐页：登录后调个性化接口，用商城真实图片 ----
async function renderRec(){
  const list=document.getElementById('reclist');
  if(list.dataset.done)return;
  list.dataset.done='1';
  document.getElementById('rectitle').textContent='正在为你挑选…';
  list.innerHTML='';
  let d;
  try{
    const resp=await fetch('/api/recommendations',{headers:authHeader()});
    if(resp.status===401){forceLogout();return;}
    d=await resp.json();
  }catch(e){
    document.getElementById('rectitle').textContent='推荐加载失败';return;
  }
  document.getElementById('rectitle').textContent=d.title||'为你推荐';
  (d.items||[]).forEach(p=>{
    const el=document.createElement('div');el.className='recitem';
    el.innerHTML=`<img src="${p.image}" onerror="this.style.background='#f3eef7'"/>
      <div style="flex:1;min-width:0">
        <div class="rname"></div>
        <div class="rreason">${p.reason||''}</div>
        <div class="rprice">¥${p.price} <small style="color:#b598aa;font-weight:400">库存${p.stock}</small></div>
      </div>`;
    el.querySelector('.rname').textContent=p.name;  // textContent 防 XSS
    // 点击推荐 → 问 Agent 这款花的详情
    el.onclick=()=>{inp.value=`${p.name}怎么样？`;send();};
    list.appendChild(el);
  });
}
// ---- 留言页 ----
function submitMsg(){
  const ta=document.getElementById('msginput');
  const text=ta.value.trim();
  if(!text){toast('留言不能为空');return;}
  msgs.unshift({text,time:new Date().toLocaleString('zh-CN')});
  localStorage.setItem(MSG_KEY,JSON.stringify(msgs));
  ta.value='';renderMsgs();toast('留言成功，小花会看到的 🌸');
}
function renderMsgs(){
  const list=document.getElementById('msglist');
  if(!msgs.length){list.innerHTML='<div class="msgempty">还没有留言，来说点什么吧~</div>';return;}
  list.innerHTML='';
  msgs.forEach(m=>{
    const el=document.createElement('div');el.className='msgitem';
    el.innerHTML=`<div class="mhead"><span>访客</span><span>${m.time}</span></div>
      <div class="mcontent"></div>`;
    el.querySelector('.mcontent').textContent=m.text;  // textContent 防 XSS
    list.appendChild(el);
  });
}
// 初始化
renderCart();updateCartBadge();

// ========== 登录 / 注册（手机号+验证码） ==========
const TOKEN_KEY='flower_token', NICK_KEY='flower_nick';
let codeTimer=null;
function authHeader(){const t=localStorage.getItem(TOKEN_KEY);return t?{Authorization:'Bearer '+t}:{};}
function loginErr(msg){document.getElementById('loginerr').textContent=msg||'';}
// 获取验证码
async function sendCode(){
  const phone=document.getElementById('phone').value.trim();
  if(!/^1\\d{10}$/.test(phone)){loginErr('请输入正确的 11 位手机号');return;}
  loginErr('');
  const btn=document.getElementById('codebtn');btn.disabled=true;
  try{
    const r=await fetch('/send_code',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({phone})});
    const d=await r.json();
    if(!d.ok){loginErr(d.message||'发送失败');btn.disabled=false;return;}
    // 开发模式：后端把验证码带回来了，自动填进去并提示（真实短信不会有这个字段）
    if(d.dev_code){
      document.getElementById('smscode').value=d.dev_code;
      toast('开发模式：验证码已自动填入 '+d.dev_code);
    }else{toast('验证码已发送');}
    // 60 秒倒计时
    let n=60;btn.textContent=n+'s 后重发';
    codeTimer=setInterval(()=>{n--;if(n<=0){clearInterval(codeTimer);
      btn.disabled=false;btn.textContent='获取验证码';}else{btn.textContent=n+'s 后重发';}},1000);
  }catch(e){loginErr('网络错误，请重试');btn.disabled=false;}
}
// 登录
async function doLogin(){
  const phone=document.getElementById('phone').value.trim();
  const code=document.getElementById('smscode').value.trim();
  if(!/^1\\d{10}$/.test(phone)){loginErr('请输入正确的 11 位手机号');return;}
  if(code.length!==6){loginErr('请输入 6 位验证码');return;}
  loginErr('');
  const btn=document.getElementById('loginbtn');btn.disabled=true;btn.textContent='登录中…';
  try{
    const r=await fetch('/login',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify({phone,code})});
    const d=await r.json();
    if(!d.ok){loginErr(d.message||'登录失败');btn.disabled=false;btn.textContent='登 录';return;}
    localStorage.setItem(TOKEN_KEY,d.token);
    localStorage.setItem(NICK_KEY,d.nickname);
    enterApp(d.nickname);
  }catch(e){loginErr('网络错误，请重试');btn.disabled=false;btn.textContent='登 录';}
}
// 退出（用户主动点）
async function doLogout(){
  try{await fetch('/logout',{method:'POST',headers:authHeader()});}catch(e){}
  forceLogout();
}
// token 失效/退出后的统一处理
function forceLogout(){
  localStorage.removeItem(TOKEN_KEY);localStorage.removeItem(NICK_KEY);
  document.getElementById('loginmask').classList.remove('hide');
  location.reload();
}
// 登录成功，进入主界面
function enterApp(nick){
  document.getElementById('nicklabel').textContent=nick||'';
  document.getElementById('loginmask').classList.add('hide');
  const btn=document.getElementById('loginbtn');btn.disabled=false;btn.textContent='登 录';
}
// 验证码框回车直接登录
document.getElementById('smscode').addEventListener('keydown',e=>{if(e.key==='Enter')doLogin()});
document.getElementById('phone').addEventListener('keydown',e=>{if(e.key==='Enter')
  document.getElementById('smscode').focus()});
// 页面加载：本地有 token 就直接进（过期了调接口会 401 被踢回登录页）
(function(){
  const t=localStorage.getItem(TOKEN_KEY);
  if(t){enterApp(localStorage.getItem(NICK_KEY)||'');}
})();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return TEST_PAGE
