"""
步骤4：Few-shot（少样本）
训练目的：通过 2~5 个示例，让 AI"照猫画虎"学会你想要的格式
           对比有示例 vs 无示例的差异
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


# ===== Few-shot =====
print("=" * 50)
print("步骤4：Few-shot（少样本）")
print("=" * 50)

# 有示例版本
result_with = ask(
    prompt="焦虑的反义词",
    system="""你是一个反义词专家。先看几个例子：
问：开心 → 答：难过
问：快速 → 答：缓慢
问：光明 → 答：黑暗
问：勇敢 → 答：胆怯
现在回答用户的问题，只输出答案，格式：XX""",
)

print("【有 Few-shot】焦虑的反义词：", result_with)

# 无示例版本（对比）
result_without = ask(
    prompt="焦虑的反义词",
    system="只输出反义词，格式：XX",
)

print("【无 Few-shot】焦虑的反义词：", result_without)
print()
print(">>> 观察：有示例时格式是否更严格？无示例时 AI 是否多说了废话？")
