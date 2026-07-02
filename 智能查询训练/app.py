"""
Gradio WebUI：智能数据库查询 Agent。
"""
import queue
import threading

import gradio as gr

from agent import agent_loop_with_logs, get_database_schema


EXAMPLES = [
    "查询所有学生的姓名和年龄",
    "年龄大于20岁的学生有几个？",
    "查询每个班级的学生人数",
    "查询 Python 成绩最高的前三名学生",
    "查询软件一班学生的数据库成绩",
    "查询不存在的字段 average_age，看看 Agent 能不能修正",
]


def respond(message: str, history: list[dict] | None):
    history = history or []
    if not message.strip():
        yield "", history, ""
        return

    history.append({"role": "user", "content": message})
    yield "", history, "[Agent] 已收到问题，开始处理..."

    event_queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
    log_lines = ["[Agent] 已收到问题，开始处理..."]

    def on_log(line: str) -> None:
        event_queue.put(("log", line))

    def worker() -> None:
        try:
            reply, _logs = agent_loop_with_logs(message, on_log=on_log)
            event_queue.put(("done", reply))
        except Exception as exc:
            event_queue.put(("error", f"执行出错：{exc}"))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        event_type, payload = event_queue.get()
        if event_type == "log":
            log_lines.append(payload or "")
            yield "", history, "\n".join(log_lines)
            continue

        if event_type == "done":
            history.append({"role": "assistant", "content": payload or "没有生成有效回复。"})
            yield "", history, "\n".join(log_lines)
            break

        log_lines.append(payload or "执行出错。")
        history.append({"role": "assistant", "content": payload or "执行出错。"})
        yield "", history, "\n".join(log_lines)
        break


with gr.Blocks(title="智能数据库查询 Agent") as demo:
    gr.Markdown(
        """
        # Agent02：智能数据库查询 Agent
        **DeepSeek Function Calling + SQLite + run_sql 工具**
        """
    )

    with gr.Accordion("当前数据库结构", open=True):
        gr.Code(value=get_database_schema(), language="sql", label="Schema")

    chatbot = gr.Chatbot(label="查询对话", height=460)

    process_log = gr.Textbox(
        value="",
        label="执行过程",
        lines=12,
        interactive=False,
    )

    with gr.Row():
        msg = gr.Textbox(
            label="输入自然语言查询",
            placeholder="例如：年龄大于20岁的学生有几个？",
            scale=5,
        )
        send = gr.Button("发送", variant="primary", scale=1)

    gr.Examples(examples=EXAMPLES, inputs=msg, label="验收示例")

    send.click(respond, [msg, chatbot], [msg, chatbot, process_log])
    msg.submit(respond, [msg, chatbot], [msg, chatbot, process_log])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7861)
