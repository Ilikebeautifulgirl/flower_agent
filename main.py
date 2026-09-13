"""阶段 4：用 LangGraph 工作流重构成单 Agent 版（入口文件）

这个文件是整个程序的入口，负责：
1. 创建并初始化工作流图
2. 维护全局对话状态 state
3. 接收用户输入，把消息塞进 state，调用图，再显示 AI 回复
4. 维持多轮对话上下文（state["messages"] 不断累积）

核心知识点：
- async/await：异步编程，因为 LangGraph 用的是异步调用 graph.ainvoke()
- HumanMessage：LangChain 标准的用户消息对象
- AgentState：自定义的全局状态字典，在整个图的所有节点间流动
"""

import asyncio  # Python 标准库，提供 async/await 异步编程支持
import uuid      # 生成唯一 ID（用来给会话起名字）
from langchain_core.messages import HumanMessage  # 用户消息类型
from core.memory.short_term import ShortTermMemory

# 从我们自己的模块里导入图管理器和状态类型
from core.workflow.graph_manager import AgentGraphManager
from core.workflow.state import AgentState


async def main():
    """
    异步主函数 - 整个程序的核心循环。

    为什么用 async def 而不是 def？
    因为 LangGraph 的 graph.ainvoke() 是异步的，必须用 await 调用。
    异步的好处：未来如果有多个用户同时聊天，程序不会卡住。
    """
    memory = ShortTermMemory()
    await memory.initialize()
    if memory.available:
        print("记忆功能已启用")
    else:
        print("记忆功能已禁用")
    # === 第 1 步：构建工作流图 ===
    # AgentGraphManager() 会初始化所有节点（包括 product_agent）
    # build_graph() 把节点连成一张图并编译，返回可执行的 graph 对象
    graph_manager = AgentGraphManager()
    graph = graph_manager.build_graph()

    # === 第 2 步：准备用户身份信息 ===
    # user_id 是用户唯一标识，阶段 7 之后用来查"MySQL 订单表"
    # session_id 是本次会话的标识，阶段 9 之后用来在 Redis 里存对话历史
    user_id = "10001"   # 演示用户：flower_shop_db 真实商城库中订单最多的一位
    session_id = f"session_{uuid.uuid4().hex[:8]}"  # 取 UUID 前 8 位作为会话名

    # === 第 3 步：初始化全局状态（这个"箱子"会在所有节点间传递）===
    # state 是一个字典，符合 AgentState 的结构定义
    # 每个字段都会被图中的节点读取或更新
    state: AgentState = {
        "messages": [],          # 消息列表：HumanMessage / AIMessage 等会不断追加进来
        "user_id": user_id,       # 用户 ID：用于权限隔离（防止查别人订单）
        "session_id": session_id, # 会话 ID：用于在 Redis 里存/取对话历史
        "memory_context": "",     # 记忆上下文：阶段 9 之后会填入从 Redis 提取的历史摘要
        "next_agent": "",         # 下一个 Agent 名字：阶段 5 之后 Orchestrator 会填它做路由
        "metadata": {}            # 元数据：阶段 11 之后用来标记跨 Agent 工作流（如 is_finops_workflow）
    }

    # === 第 4 步：启动欢迎界面 ===
    print("🤖 云平台客服助手（LangGraph 工作流版）已启动（输入 quit 退出）")
    print("=" * 50)
    print(f"   用户: {user_id}")
    print(f"   会话: {session_id}")
    print("=" * 50)

    # === 第 5 步：进入主对话循环 ===
    while True:
        # 接收用户输入，strip() 去掉首尾空格
        user_input = input("你: ").strip()

        # 输入 quit / exit / q 就退出循环
        # .lower() 转小写，让 QUIT / Quit 也能退出
        if user_input.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break

        # 空输入跳过，不浪费一次 AI 调用
        if not user_input:
            continue

        # 把用户消息包装成 HumanMessage 后追加到 state["messages"]
        # 为什么用 append 而不是赋值？因为要保留历史对话
        # 这样 AI 能看到之前聊过什么（多轮上下文）
        history = await memory.get_messages(user_id,session_id)
        state["memory_context"] = "\n".join(f"{m['role']}: {m['content']}" for m in history)
        state["messages"].append(HumanMessage(content=user_input))

        # === 第 6 步：执行工作流！===
        # graph.ainvoke() 是异步调用，会沿着 START → product_agent → END 走一圈
        # 节点处理完后返回更新后的 state（包含 AI 的回复消息）
        result = await graph.ainvoke(state)

        # === 第 7 步：取出 AI 的最后一条回复 ===
        # result["messages"] 是一个列表，最后一条就是 AI 最新的回复
        # [-1] 是 Python 切片语法，表示取列表最后一个元素
        ai_message = result["messages"][-1]

        # === 第 8 步：把更新后的 messages 同步回 state ===
        # 这一步很关键！不同步的话，下一轮循环时 state["messages"] 还是旧的
        # 会导致 AI 看不到刚才自己说了什么
        state["messages"] = result["messages"]

        # 打印 AI 的回答
        print(f"AI: {ai_message.content}")
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": ai_message.content})
        await memory.save_messages(user_id,session_id,history)
# Python 标准入口：当直接运行此文件时（python main.py），执行下面的代码
# asyncio.run() 启动异步主函数
if __name__ == "__main__":
    asyncio.run(main())
