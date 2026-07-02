"""
智能数据库查询 Agent。

核心能力：
1. 大模型把自然语言问题转换为 SQL。
2. Agent 通过 run_sql 工具执行 SQL。
3. SQL 报错时，把错误返回给大模型，由模型修正后重试。
4. 查询结果再由大模型组织成人类可读回答。
"""
import json
import os
import sqlite3
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


BASE_DIR = os.path.dirname(__file__)
load_dotenv(os.path.join(BASE_DIR, ".env"))

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
MAX_AGENT_TURNS = int(os.getenv("MAX_AGENT_TURNS", "4"))

DB_PATH = os.getenv("DB_PATH", "students.db")
DB_FILE = DB_PATH if os.path.isabs(DB_PATH) else os.path.join(BASE_DIR, DB_PATH)

client: OpenAI | None = None


def log_event(logs: list[str] | None, message: str, on_log=None) -> None:
    print(message)
    if logs is not None:
        logs.append(message)
    if on_log is not None:
        on_log(message)


def get_client() -> OpenAI:
    global client
    if client is None:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return client


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def get_database_schema() -> str:
    """读取当前数据库结构，让 Agent 不依赖写死的一张表。"""
    if not os.path.exists(DB_FILE):
        return f"数据库文件不存在：{DB_FILE}，请先运行 python init_db.py"

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
    )
    table_names = [row["name"] for row in cursor.fetchall()]

    parts: list[str] = []
    for table in table_names:
        cursor.execute(f"PRAGMA table_info({table})")
        columns = cursor.fetchall()
        column_desc = ", ".join(
            f"{col['name']} {col['type']}{' PRIMARY KEY' if col['pk'] else ''}"
            for col in columns
        )
        parts.append(f"- {table}({column_desc})")

    conn.close()
    return "\n".join(parts) if parts else "当前数据库没有可查询的数据表。"


def run_sql(sql: str, logs: list[str] | None = None, on_log=None) -> str:
    """
    Function Calling 工具：执行 SQL 并返回 JSON 字符串。

    为了课堂验收更安全，默认只允许 SELECT 查询。
    如果 SQL 错误，会把错误信息返回给 Agent，触发下一轮修正。
    """
    log_event(logs, f"[run_sql] 准备执行 SQL:\n{sql}", on_log)

    normalized = sql.strip().rstrip(";")
    if not normalized:
        return json.dumps(
            {"success": False, "error": "SQL 为空，请重新生成 SQL。"},
            ensure_ascii=False,
        )

    first_word = normalized.split()[0].lower()
    if first_word not in {"select", "with"}:
        return json.dumps(
            {"success": False, "error": "当前工具只允许执行 SELECT 查询。"},
            ensure_ascii=False,
        )

    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(normalized)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []

        result_rows: list[dict[str, Any]] = []
        for row in rows[:50]:
            result_rows.append({column: row[column] for column in columns})

        result = {
            "success": True,
            "columns": columns,
            "rows": result_rows,
            "row_count": len(rows),
            "truncated": len(rows) > 50,
        }
        log_event(logs, f"[run_sql] 执行成功，返回 {len(rows)} 行", on_log)
        log_event(logs, f"[run_sql] 执行结果：{json.dumps(result, ensure_ascii=False)}", on_log)
        return json.dumps(result, ensure_ascii=False)
    except Exception as exc:
        log_event(logs, f"[run_sql] 执行失败：{exc}", on_log)
        return json.dumps(
            {"success": False, "error": str(exc), "failed_sql": sql},
            ensure_ascii=False,
        )
    finally:
        try:
            conn.close()
        except Exception:
            pass


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "执行一条 SQLite SELECT SQL，并返回查询结果或错误信息。",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "要执行的 SQLite SELECT 查询语句。",
                    }
                },
                "required": ["sql"],
            },
        },
    }
]


def build_system_prompt() -> str:
    schema = get_database_schema()
    return f"""
你是一个智能数据库查询 Agent，负责把用户的自然语言问题转换为 SQLite SQL。

数据库结构如下：
{schema}

规则：
1. 只生成 SQLite 兼容 SQL。
2. 查询必须通过 run_sql 工具执行，不要凭空编造数据。
3. 如果 run_sql 返回 success=false，说明 SQL 执行失败，你需要根据错误信息修正 SQL 并再次调用 run_sql。
4. 如果查询涉及多个表，请根据外键关系生成 JOIN。
5. 最终回答要使用中文，简洁说明查询结论；必要时列出关键数据。
6. 不要向用户展示无关的内部推理过程。
""".strip()


def call_tool(tool_name: str, arguments: str, logs: list[str] | None = None, on_log=None) -> str:
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError as exc:
        return json.dumps({"success": False, "error": f"工具参数不是合法 JSON：{exc}"}, ensure_ascii=False)

    log_event(logs, f"[Agent] 调用工具：{tool_name}({json.dumps(args, ensure_ascii=False)})", on_log)

    if tool_name == "run_sql":
        sql = args.get("sql", "")
        log_event(logs, f"[Agent] 生成 SQL：{sql}", on_log)
        return run_sql(sql, logs, on_log)
    return json.dumps({"success": False, "error": f"未知工具：{tool_name}"}, ensure_ascii=False)


def agent_loop(user_input: str) -> str:
    reply, _logs = agent_loop_with_logs(user_input)
    return reply


def agent_loop_with_logs(user_input: str, on_log=None) -> tuple[str, str]:
    logs: list[str] = []

    if not DEEPSEEK_API_KEY or "请填写" in DEEPSEEK_API_KEY:
        final_reply = "请先在 .env 文件中填写 DEEPSEEK_API_KEY，然后重新启动程序。"
        log_event(logs, f"[Agent] 最终回复：{final_reply}", on_log)
        return final_reply, "\n".join(logs)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_input},
    ]

    for turn in range(1, MAX_AGENT_TURNS + 1):
        log_event(logs, f"[Agent] 第 {turn}/{MAX_AGENT_TURNS} 轮思考", on_log)
        response = get_client().chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = response.choices[0].message

        if message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": call.function.name,
                                "arguments": call.function.arguments,
                            },
                        }
                        for call in message.tool_calls
                    ],
                }
            )

            for call in message.tool_calls:
                tool_result = call_tool(call.function.name, call.function.arguments, logs, on_log)
                log_event(logs, f"[Agent] 工具返回：{tool_result[:300]}", on_log)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_result,
                    }
                )
            continue

        final_reply = message.content or "没有生成有效回复。"
        log_event(logs, f"[Agent] 最终回复：{final_reply}", on_log)
        return final_reply, "\n".join(logs)

    final_reply = "Agent 已达到最大重试轮次，请换一种更明确的问法。"
    log_event(logs, f"[Agent] 最终回复：{final_reply}", on_log)
    return final_reply, "\n".join(logs)


if __name__ == "__main__":
    while True:
        question = input("\n请输入查询问题，输入 q 退出：").strip()
        if question.lower() == "q":
            break
        print(agent_loop(question))
