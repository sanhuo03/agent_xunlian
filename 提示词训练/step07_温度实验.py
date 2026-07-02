"""
步骤7：温度实验（temperature）
训练目的：理解 temperature 参数如何影响 AI 输出的"确定性 vs 创造性"
           同一问题，3种温度各跑3次，横向对比

temperature 值含义：
  0    → 像严谨的会计师——每次都给你同样的标准答案
  0.7  → 像有创意的文案——每次给的都不一样，可能有惊喜
  1.0  → 像喝多了的朋友——开始自由发挥，甚至跑偏
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


# ===== 温度实验 =====
print("=" * 70)
print("步骤7：温度实验（temperature）")
print("=" * 70)

prompt = "用一句话介绍Python"

temperatures = [0, 0.7, 1.0]
runs = 3

print(f'问题："{prompt}"')
print()

for temp in temperatures:
    label = {0: "[严谨模式]", 0.7: "[平衡模式]", 1.0: "[创意模式]"}[temp]
    print(f"{'='*50}")
    print(f"Temperature = {temp}  ({label})")
    print(f"{'='*50}")
    for i in range(1, runs + 1):
        result = ask(prompt=prompt, temperature=temp)
        print(f"  第{i}次: {result}")
    print()

print("=" * 50)
print("对比总结")
print("=" * 50)
print("temperature=0   → 3次输出几乎一模一样（确定性最强）")
print("temperature=0.7 → 3次输出略有不同，但主题一致")
print("temperature=1.0 → 3次输出差异最大，甚至可能跑偏")
print()
print(">>> 规律：temperature 越高 = 越随机 = 越有创意但越不可控")
print(">>> 建议：写代码/JSON 用0，写故事/广告用0.7~1.0")
