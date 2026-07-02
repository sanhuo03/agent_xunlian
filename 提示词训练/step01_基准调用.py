"""
步骤1：基准调用（裸调用）
训练目的：跑通 API 调用，对比同一问题两次输出的差异，理解 LLM 的随机性
"""
import os
import difflib
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


def ask(prompt, system="你是一个有帮助的助手", temperature=0.7):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def calc_similarity(a, b):
    """用 difflib 计算两段文本的相似度"""
    if not a or not b:
        return 0
    return difflib.SequenceMatcher(None, a, b).ratio() * 100


# ===== Gradio 并排对比 UI =====
with gr.Blocks(title="步骤1：基准调用") as demo:
    gr.Markdown("# 步骤1：基准调用（裸调用）")
    gr.Markdown("DeepSeek V4 Pro — 同一问题问两遍，并排对比差异")

    state = gr.State({"q": "", "a1": ""})

    with gr.Row():
        inp = gr.Textbox(label="输入问题", placeholder="输入你的问题...", scale=4, autofocus=True)
        send_btn = gr.Button("发送", variant="primary", scale=1)

    reask_btn = gr.Button("再问一次 — 对比两次回答", variant="secondary", visible=False)

    with gr.Row(visible=False) as compare_panel:
        with gr.Column():
            gr.Markdown("### 第1次回答")
            out1 = gr.Markdown()
        with gr.Column():
            gr.Markdown("### 第2次回答")
            out2 = gr.Markdown()

    sim_display = gr.Markdown(visible=False)


    def on_send(msg):
        if not msg.strip():
            return msg, None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), {"q": "", "a1": ""}
        a1 = ask(msg)
        return (
            "",                                                    # 清空输入框
            f"```\n{a1}\n```",                                     # 第1次回答
            None,                                                   # 第2次回答清空
            gr.update(visible=True),                                # 显示"再问一次"按钮
            gr.update(visible=True),                                # 显示对比面板
            gr.update(visible=False, value=None),                   # 隐藏相似度
            {"q": msg, "a1": a1},                                   # 更新状态
        )

    def on_reask(st):
        q = st["q"]
        a1 = st["a1"]
        if not q:
            return None, None, None, st
        a2 = ask(q)
        sim = calc_similarity(a1, a2)
        if sim >= 90:
            emoji = "[几乎一样]"
        elif sim >= 60:
            emoji = "[部分差异]"
        else:
            emoji = "[差异很大]"
        sim_text = f"### 相似度: {sim:.1f}%  {emoji}\n\n> 观察：同一问题两次输出的措辞是否不同？语义是否一致？"
        return (
            f"```\n{a1}\n```",
            f"```\n{a2}\n```",
            gr.update(visible=True, value=sim_text),
            st,
        )

    send_btn.click(
        on_send, inputs=[inp],
        outputs=[inp, out1, out2, reask_btn, compare_panel, sim_display, state]
    )
    reask_btn.click(
        on_reask, inputs=[state],
        outputs=[out1, out2, sim_display, state]
    )


demo.launch(server_name="127.0.0.1", server_port=7860)
