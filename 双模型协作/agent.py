"""
Agent03: DeepSeek + Hrida 双模型协作数据库查询 Agent.

分工:
- DeepSeek: 理解用户问题、判断查询复杂度、简单 SQL 生成、结果翻译。
- Hrida-T2SQL-3B: 复杂 SQL 生成，重点处理递归 CTE、窗口函数等场景。
- SQLite: 执行 SQL，并把结果返回给模型解释。
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

CUDA_BIN_CANDIDATES = [
    Path(os.getenv("CUDA_PATH", "")) / "bin" if os.getenv("CUDA_PATH") else None,
    Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"),
    Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"),
]
for cuda_bin in CUDA_BIN_CANDIDATES:
    if cuda_bin and cuda_bin.exists():
        os.environ["PATH"] = str(cuda_bin) + os.pathsep + os.environ.get("PATH", "")
        try:
            os.add_dll_directory(str(cuda_bin))
        except (AttributeError, OSError):
            pass

DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-pro")
DB_PATH = (BASE_DIR / os.getenv("DB_PATH", "data.db")).resolve()
MODEL_PATH = (BASE_DIR / os.getenv("MODEL_PATH", "models/Hrida-T2SQL-3B-V0.1_Q4_K_M.gguf")).resolve()

deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

hrida = None
HRIDA_AVAILABLE = False


def _strip_code_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:sql|json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_sql(text: str) -> str:
    text = _strip_code_block(text)
    match = re.search(r"(WITH\s+RECURSIVE[\s\S]+|WITH\s+[\s\S]+|SELECT[\s\S]+|INSERT[\s\S]+|UPDATE[\s\S]+|DELETE[\s\S]+)", text, re.IGNORECASE)
    if match:
        text = match.group(1)
    return text.strip().rstrip(";") + ";"


def _load_hrida() -> tuple[bool, str]:
    global hrida, HRIDA_AVAILABLE
    if HRIDA_AVAILABLE and hrida is not None:
        return True, "Hrida-T2SQL-3B 已加载"
    if not MODEL_PATH.exists():
        return False, f"模型文件不存在: {MODEL_PATH}"
    try:
        from llama_cpp import Llama

        hrida = Llama(
            model_path=str(MODEL_PATH),
            n_ctx=4096,
            n_gpu_layers=-1,
            n_threads=max((os.cpu_count() or 4) - 1, 1),
            verbose=False,
        )
        HRIDA_AVAILABLE = True
        return True, "Hrida-T2SQL-3B 加载成功"
    except ImportError:
        return False, "llama-cpp-python 未安装"
    except Exception as exc:
        return False, f"模型加载失败: {exc}"


def get_hrida_status() -> str:
    ok, message = _load_hrida()
    return message if ok else f"Hrida 不可用: {message}"


def get_schema() -> str:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"数据库不存在: {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        schema_lines: list[str] = []
        for table_name, ddl in rows:
            count = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()[0]
            columns = conn.execute(f'PRAGMA table_info("{table_name}")').fetchall()
            column_text = ", ".join(f"{col[1]} {col[2]}" for col in columns)
            schema_lines.append(f"表名: {table_name}")
            schema_lines.append(f"字段: {column_text}")
            schema_lines.append(f"建表语句: {ddl}")
            schema_lines.append(f"数据行数: {count}")
            schema_lines.append("---")
        return "\n".join(schema_lines)
    finally:
        conn.close()


def analyze_complexity(user_input: str, schema_text: str) -> dict[str, Any]:
    prompt = f"""你是 SQL 查询复杂度分析助手。请判断用户问题应该走 simple 还是 complex。

判断规则:
- complex: 需要递归 CTE、祖先/子孙/上下级层级查询、窗口函数 LAG/LEAD/RANK/ROW_NUMBER、连续区间 Gaps and Islands。
- simple: 普通 SELECT、WHERE、JOIN、GROUP BY、COUNT、AVG、SUM 等。

用户问题:
{user_input}

数据库 Schema:
{schema_text}

只输出 JSON，不要输出 Markdown:
{{"complexity":"simple 或 complex","reason":"一句话说明原因","keywords":["关键词"]}}
"""
    response = deepseek.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = _strip_code_block(response.choices[0].message.content or "")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"complexity": "simple", "reason": "JSON 解析失败，默认走简单查询", "keywords": []}
    if data.get("complexity") not in {"simple", "complex"}:
        data["complexity"] = "simple"
    data.setdefault("reason", "")
    data.setdefault("keywords", [])
    return data


def deepseek_generate_sql(user_input: str, schema_text: str, error: str | None = None, bad_sql: str | None = None) -> str:
    repair_text = ""
    if error:
        repair_text = f"""

上一次 SQL:
{bad_sql}

执行错误:
{error}

请修正 SQL。
"""
    prompt = f"""你是 SQLite 专家。根据用户问题和数据库 Schema 生成可执行 SQL。

要求:
- 只输出 SQL，不要解释，不要 Markdown。
- 使用 SQLite 语法。
- 不要编造表名或字段名。
- 查询结果字段可以使用中文别名。

数据库 Schema:
{schema_text}

用户问题:
{user_input}
{repair_text}
"""
    response = deepseek.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return _extract_sql(response.choices[0].message.content or "")


def hrida_generate_sql(question: str, schema_text: str, error: str | None = None, bad_sql: str | None = None) -> str:
    ok, message = _load_hrida()
    if not ok:
        raise RuntimeError(message)

    repair_text = ""
    if error:
        repair_text = f"""
The previous SQL failed.
Previous SQL:
{bad_sql}
Error:
{error}
Please generate a corrected SQL query.
"""

    prompt = f"""### Instruction:
You are an expert Text-to-SQL model. Generate one precise SQLite query.

### Database schema:
{schema_text}

### Question:
{question}
{repair_text}

### SQL:
"""
    result = hrida.create_chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=600,
        temperature=0,
        stop=["</s>", "<|end|>"],
    )
    return _extract_sql(result["choices"][0]["message"]["content"])


def execute_sql(sql: str) -> tuple[bool, str, str | None]:
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        if sql.lstrip().upper().startswith(("SELECT", "WITH")):
            rows = cursor.fetchall()
            columns = [item[0] for item in cursor.description or []]
            if not rows:
                return True, "查询结果为空。", None
            header = " | ".join(columns)
            body = "\n".join(" | ".join(str(value) for value in row) for row in rows[:80])
            return True, f"{header}\n{'-' * 60}\n{body}\n共 {len(rows)} 行", None
        conn.commit()
        return True, f"执行成功，影响 {cursor.rowcount} 行。", None
    except Exception as exc:
        return False, "", str(exc)
    finally:
        conn.close()


def translate_result(question: str, sql: str, raw_result: str) -> str:
    prompt = f"""请把数据库查询结果翻译成用户能看懂的中文回答。

用户问题:
{question}

执行 SQL:
{sql}

原始结果:
{raw_result}

要求:
- 直接回答问题。
- 简洁说明关键数据。
- 如果结果为空，说明没有找到符合条件的数据。
"""
    response = deepseek.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def agent03_loop(user_input: str) -> str:
    logs: list[str] = []

    logs.append("[步骤1] 读取数据库 Schema")
    schema_text = get_schema()
    table_count = schema_text.count("表名:")
    logs.append(f"[步骤1] 共发现 {table_count} 张表")

    logs.append("[步骤2] DeepSeek 判断查询复杂度")
    analysis = analyze_complexity(user_input, schema_text)
    complexity = analysis["complexity"]
    logs.append(f"[步骤2] complexity={complexity}; reason={analysis.get('reason', '')}")

    sql = ""
    generator = "DeepSeek"
    last_error: str | None = None
    for attempt in range(1, 4):
        try:
            if complexity == "complex":
                ok, status = _load_hrida()
                if ok:
                    generator = "Hrida-T2SQL-3B"
                    logs.append(f"[步骤3] 第 {attempt} 次由 Hrida 生成复杂 SQL")
                    sql = hrida_generate_sql(user_input, schema_text, last_error, sql or None)
                else:
                    generator = "DeepSeek 兜底"
                    logs.append(f"[步骤3] Hrida 不可用({status})，第 {attempt} 次由 DeepSeek 兜底生成 SQL")
                    sql = deepseek_generate_sql(user_input, schema_text, last_error, sql or None)
            else:
                generator = "DeepSeek"
                logs.append(f"[步骤3] 第 {attempt} 次由 DeepSeek 生成简单 SQL")
                sql = deepseek_generate_sql(user_input, schema_text, last_error, sql or None)
        except Exception as exc:
            return "SQL 生成失败: " + str(exc) + "\n\n内部日志:\n" + "\n".join(logs)

        logs.append(f"[步骤3] 生成模型: {generator}")
        logs.append(f"[步骤3] SQL: {sql}")
        logs.append("[步骤4] 执行 SQL")
        ok, raw_result, error = execute_sql(sql)
        if ok:
            logs.append("[步骤4] SQL 执行成功")
            logs.append("[步骤5] DeepSeek 翻译查询结果")
            answer = translate_result(user_input, sql, raw_result)
            return f"{answer}\n\n---\nSQL:\n{sql}\n\n原始结果:\n{raw_result}\n\n内部日志:\n" + "\n".join(logs)
        last_error = error
        logs.append(f"[步骤4] SQL 执行失败: {error}")

    return f"SQL 执行失败，已重试 3 次。\n最后 SQL:\n{sql}\n错误信息: {last_error}\n\n内部日志:\n" + "\n".join(logs)


if __name__ == "__main__":
    print("Agent03: DeepSeek + Hrida 双模型协作")
    print("Hrida 状态:", get_hrida_status())
    print("数据库路径:", DB_PATH)
    print("模型路径:", MODEL_PATH)
    print("输入 quit 退出。")
    while True:
        question = input("\n查询> ").strip()
        if question.lower() in {"quit", "exit"}:
            break
        if question:
            print(agent03_loop(question))
