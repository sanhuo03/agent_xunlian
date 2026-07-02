"""
步骤6：格式约束
训练目的：给 AI 一个精确的输出模板，AI 只能"填空"，不能自由发挥
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


# ===== 格式约束 =====
print("=" * 50)
print("步骤6：格式约束")
print("=" * 50)

result = ask(
    prompt="推荐3本Python入门书籍",
    system="""你用中文回答，严格按以下格式，不要输出其他内容：
1. 《书名》- 作者 [难度: ★☆☆~★★★]
   > 一句话介绍

2. 《书名》- 作者 [难度: ★☆☆~★★★]
   > 一句话介绍

3. 《书名》- 作者 [难度: ★☆☆~★★★]
   > 一句话介绍""",
    temperature=0.3,
)

print(result)
print()
print(">>> 观察：3本书的格式是否完全一致？有没有偏离模板？")
