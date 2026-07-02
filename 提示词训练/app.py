"""
提示词工程实战训练集成版 WebUI

功能：
1. 在同一个 Gradio 页面中集成 6 种 Prompt 模式
2. 提供 temperature 滑块，观察不同温度下输出差异
3. 提供模式对比 Tab，总结各模式相对裸调用的优势
"""
import json
import os
from typing import Any

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

MODEL_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


def ask(prompt: str, system: str = "你是一个有帮助的助手。", temperature: float = 0.7) -> str:
    """统一的大模型调用函数。"""
    if not os.getenv("DEEPSEEK_API_KEY") or not os.getenv("DEEPSEEK_BASE_URL"):
        return "请先在 .env 文件中配置 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL。"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        return f"调用失败：{exc}"


def safe_json_pretty(text: str) -> str:
    """尝试格式化 JSON，失败时返回原文和提示。"""
    try:
        data: Any = json.loads(text)
        return "JSON 解析成功：\n\n```json\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n```"
    except Exception as exc:
        return f"JSON 解析失败：{exc}\n\n原始输出：\n\n{text}"


def baseline(prompt: str, temperature: float) -> str:
    return ask(prompt, system="你是一个有帮助的助手。", temperature=temperature)


def role_play(prompt: str, role_desc: str, temperature: float) -> str:
    system = f"你现在的角色设定如下：{role_desc}\n请严格按照这个身份、语气和风格回答用户问题。"
    return ask(prompt, system=system, temperature=temperature)


def structured_output(prompt: str, temperature: float) -> str:
    system = """你是一个技术分析师。请只输出严格 JSON，不要输出任何解释性文字。
JSON 格式必须如下：
{
  "topic": "主题",
  "summary": "一句话总结",
  "items": [
    {
      "name": "名称",
      "strength": "优势",
      "weakness": "劣势",
      "use_case": "适用场景"
    }
  ]
}"""
    raw = ask(prompt, system=system, temperature=temperature)
    return safe_json_pretty(raw)


def few_shot(prompt: str, temperature: float) -> str:
    system = """你是一个反义词和词语解释专家。请学习下面的回答格式：
问：开心
答：
词语：开心
反义词：难过
解释：开心表示心情愉快，难过表示心情低落。

问：快速
答：
词语：快速
反义词：缓慢
解释：快速强调速度快，缓慢强调速度慢。

问：光明
答：
词语：光明
反义词：黑暗
解释：光明表示明亮，黑暗表示缺少光亮。

现在请按照同样格式回答用户的问题，不要增加无关内容。"""
    return ask(prompt, system=system, temperature=temperature)


def chain_of_thought(prompt: str, temperature: float) -> str:
    system = """你是一个严谨的数学和逻辑老师。
请分步骤分析问题，写清楚关键推理过程，最后单独用“最终答案：”给出结论。"""
    return ask(prompt, system=system, temperature=temperature)


def format_constraint(prompt: str, temperature: float) -> str:
    system = """你是一个学习资料推荐助手。请严格按照下面模板输出，不要输出其他内容：
1. 《书名》 - 作者 [难度：★☆☆/★★☆/★★★]
   推荐理由：一句话说明
   适合人群：一句话说明

2. 《书名》 - 作者 [难度：★☆☆/★★☆/★★★]
   推荐理由：一句话说明
   适合人群：一句话说明

3. 《书名》 - 作者 [难度：★☆☆/★★☆/★★★]
   推荐理由：一句话说明
   适合人群：一句话说明"""
    return ask(prompt, system=system, temperature=temperature)


def temperature_demo(prompt: str) -> str:
    rows = []
    for temp in [0, 0.7, 1.0]:
        for idx in range(1, 4):
            result = ask(prompt, temperature=temp)
            rows.append(f"### temperature={temp} 第{idx}次\n{result}")
    return "\n\n---\n\n".join(rows)


COMPARISON = [
    ["裸调用", "不设置特殊 System Prompt", "简单直接", "输出不可控，格式和语气不稳定"],
    ["角色扮演", "设定身份、语气、风格", "适合教学、客服、专家问答等场景", "可能为了风格牺牲部分信息密度"],
    ["结构化输出", "要求只输出 JSON 等结构", "程序可以直接解析和处理", "需要配合代码校验，避免格式漂移"],
    ["Few-shot", "在提示词中提供示例", "不用微调也能让模型模仿格式", "示例质量会直接影响输出质量"],
    ["思维链", "要求逐步推理后再回答", "适合数学、逻辑、分析类问题", "输出更长，简单问题会显得啰嗦"],
    ["格式约束", "给固定模板让模型填空", "适合生成清单、报告、推荐表", "模板太死时灵活性较低"],
]


with gr.Blocks(title="提示词工程实战训练") as demo:
    gr.Markdown("# 提示词工程实战训练")
    gr.Markdown("DeepSeek + Gradio：对比 6 种 Prompt 模式，并观察 temperature 对输出稳定性的影响。")

    with gr.Accordion("temperature 参数说明", open=True):
        gr.Markdown(
            """
- `0`：确定性最强，适合代码、JSON、严谨问答。
- `0.7`：平衡准确性和表达变化，适合一般问答、解释、推荐。
- `1.0`：随机性更强，适合创意写作，但更容易跑偏。
"""
        )

    with gr.Tabs():
        with gr.Tab("1 裸调用"):
            base_prompt = gr.Textbox(value="介绍一下 Python", label="输入问题", lines=3)
            base_temp = gr.Slider(0, 1, value=0.7, step=0.1, label="temperature")
            base_btn = gr.Button("发送", variant="primary")
            base_out = gr.Markdown(label="输出")
            base_btn.click(baseline, [base_prompt, base_temp], base_out)

        with gr.Tab("2 角色扮演"):
            role_prompt = gr.Textbox(value="介绍一下 Python", label="输入问题", lines=3)
            role_desc = gr.Textbox(
                value="你是一位资深 Python 讲师，讲话幽默风趣，喜欢用生活化例子解释概念，回答不超过 200 字。",
                label="角色设定",
                lines=3,
            )
            role_temp = gr.Slider(0, 1, value=0.7, step=0.1, label="temperature")
            role_btn = gr.Button("发送", variant="primary")
            role_out = gr.Markdown(label="输出")
            role_btn.click(role_play, [role_prompt, role_desc, role_temp], role_out)

        with gr.Tab("3 结构化输出"):
            json_prompt = gr.Textbox(value="帮我比较 Python、Java、JavaScript 三种编程语言", label="输入问题", lines=3)
            json_temp = gr.Slider(0, 1, value=0, step=0.1, label="temperature")
            json_btn = gr.Button("生成 JSON", variant="primary")
            json_out = gr.Markdown(label="输出")
            json_btn.click(structured_output, [json_prompt, json_temp], json_out)

        with gr.Tab("4 Few-shot"):
            few_prompt = gr.Textbox(value="焦虑", label="输入词语", lines=2)
            few_temp = gr.Slider(0, 1, value=0.3, step=0.1, label="temperature")
            few_btn = gr.Button("发送", variant="primary")
            few_out = gr.Markdown(label="输出")
            few_btn.click(few_shot, [few_prompt, few_temp], few_out)

        with gr.Tab("5 思维链"):
            cot_prompt = gr.Textbox(
                value="小明有5个苹果，给了小红2个，又买了3个，然后吃掉1个，最后把剩下的一半给了小刚。小明还剩几个苹果？",
                label="输入推理题",
                lines=4,
            )
            cot_temp = gr.Slider(0, 1, value=0, step=0.1, label="temperature")
            cot_btn = gr.Button("逐步推理", variant="primary")
            cot_out = gr.Markdown(label="输出")
            cot_btn.click(chain_of_thought, [cot_prompt, cot_temp], cot_out)

        with gr.Tab("6 格式约束"):
            fmt_prompt = gr.Textbox(value="推荐3本 Python 入门书籍", label="输入问题", lines=3)
            fmt_temp = gr.Slider(0, 1, value=0.3, step=0.1, label="temperature")
            fmt_btn = gr.Button("按模板生成", variant="primary")
            fmt_out = gr.Markdown(label="输出")
            fmt_btn.click(format_constraint, [fmt_prompt, fmt_temp], fmt_out)

        with gr.Tab("温度实验"):
            temp_prompt = gr.Textbox(value="用一句话介绍 Python", label="输入问题", lines=2)
            temp_btn = gr.Button("分别用 0 / 0.7 / 1.0 各运行 3 次", variant="primary")
            temp_out = gr.Markdown(label="输出")
            temp_btn.click(temperature_demo, temp_prompt, temp_out)

        with gr.Tab("模式对比"):
            gr.Dataframe(
                headers=["模式", "做法", "相对裸调用的优势", "注意点"],
                value=COMPARISON,
                interactive=False,
                wrap=True,
            )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
