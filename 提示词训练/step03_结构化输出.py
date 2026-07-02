"""
步骤3：结构化输出（JSON）
训练目的：强制 AI 输出 JSON，让代码能直接读取，对接程序
"""
import os
import json
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


# ===== 结构化输出（JSON） =====
print("=" * 50)
print("步骤3：结构化输出（JSON）")
print("=" * 50)

result = ask(
    prompt="帮我比较 Python、Java、JavaScript 三种编程语言",
    system="""你是一个技术分析师。请用严格的JSON格式回答，格式如下：
{
  "comparison": [
    {"language": "语言名", "type": "类型", "strength": "优势", "weakness": "劣势", "use_case": "典型场景"}
  ]
}
只输出JSON，不要有任何其他文字。""",
    temperature=0,  # 结构化输出用0，保证确定性
)

print("原始输出：")
print(result)
print()

try:
    data = json.loads(result)
    print("[成功] JSON 解析成功！逐条展示：")
    for item in data["comparison"]:
        print(f"  {item['language']}: {item['strength']}")
except json.JSONDecodeError:
    print("[失败] JSON解析失败，原始输出：", result[:200])
