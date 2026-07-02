"""Agent03 Gradio WebUI."""
from __future__ import annotations

import os

import gradio as gr
from dotenv import load_dotenv

from agent import DB_PATH, MODEL_PATH, agent03_loop, get_hrida_status

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


def respond(message: str, history: list[dict]) -> tuple[str, list[dict]]:
    history = history or []
    if not message.strip():
        return "", history
    history.append({"role": "user", "content": message})
    try:
        reply = agent03_loop(message.strip())
    except Exception as exc:
        reply = f"运行出错: {exc}"
    history.append({"role": "assistant", "content": reply})
    return "", history


with gr.Blocks(title="Agent03 双模型协作查询") as demo:
    gr.Markdown(
        f"""
# Agent03: DeepSeek + Hrida 双模型协作查询

| 模型 | 角色 | 状态 |
| --- | --- | --- |
| DeepSeek V4 Pro | 推理层: 意图解析、复杂度判断、结果翻译 | 云端 API |
| Hrida-T2SQL-3B | SQL 生成层: 递归 CTE、窗口函数、复杂 SQL | {get_hrida_status()} |

当前数据库: `{DB_PATH.name}`  
本地模型: `{MODEL_PATH.name}`
"""
    )

    chatbot = gr.Chatbot(label="对话", height=460)
    with gr.Row():
        msg = gr.Textbox(
            label="输入自然语言查询",
            placeholder="例如: 大金的子孙有哪些",
            scale=5,
        )
        send = gr.Button("发送", variant="primary", scale=1)

    gr.Examples(
        examples=[
            "查询所有宠物的名字、品种和主人",
            "每种宠物类型分别有多少只",
            "大金的子孙有哪些",
            "每只宠物体重增长趋势",
            "大黄体重连续下降超过3天的记录",
        ],
        inputs=msg,
    )

    send.click(respond, [msg, chatbot], [msg, chatbot])
    msg.submit(respond, [msg, chatbot], [msg, chatbot])


if __name__ == "__main__":
    try:
        demo.launch(server_name="127.0.0.1", server_port=7860)
    except OSError:
        demo.launch(server_name="127.0.0.1")
