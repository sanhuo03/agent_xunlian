"""
步骤5：思维链（Chain-of-Thought）
训练目的：让 AI "先想再答"，写出推理过程，而不是直接给答案
           对比 CoT vs 直接回答的正确率
"""
import os
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


# ===== Chain-of-Thought =====
print("=" * 50)
print("步骤5：思维链（Chain-of-Thought）")
print("=" * 50)

question = "小明有5个苹果，给了小红2个，又买了3个，然后吃掉1个，最后把剩下的一半给了小刚。小明还剩几个苹果？"

print(f"题目：{question}")
print()

# CoT 版本
result_cot = ask(
    prompt=question,
    system="你是一个数学老师。请一步一步推理，每一步都写清楚，最后给出答案。",
    temperature=0,
)

print("【CoT 推理过程】")
print(result_cot)
print()

# 直接回答版本（对比）
result_direct = ask(
    prompt=f"{question} 直接给答案，只输出数字。",
    system="你是计算器，直接输出计算结果。",
    temperature=0,
)

print(f"【直接回答】{result_direct}")
print()
print(">>> 正确答案：2.5个苹果")
print(">>> 观察：CoT 的推理过程对吗？直接回答的结果对吗？")
