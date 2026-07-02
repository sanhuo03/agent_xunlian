"""
步骤2：角色扮演
训练目的：用 system prompt 给 AI 一个"人设"，控制语气、风格
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


# ===== 角色扮演 =====
print("=" * 50)
print("步骤2：角色扮演")
print("=" * 50)

result = ask(
    prompt="介绍一下Python",
    system="你是一位资深Python讲师，讲话风格幽默风趣，喜欢用美食打比方。回答不超过200字。",
)
print(result)
