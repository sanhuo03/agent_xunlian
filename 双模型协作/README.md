# Agent03: DeepSeek + Hrida 双模型协作

本项目实现 DeepSeek + Hrida-T2SQL-3B 双模型协作的智能数据库查询 Agent。

## 分工

- DeepSeek V4 Pro: 意图解析、复杂度判断、简单 SQL 生成、结果翻译。
- Hrida-T2SQL-3B: 复杂 SQL 生成，重点处理递归 CTE、窗口函数、Gaps and Islands 等查询。
- SQLite: 执行 SQL。
- Gradio: WebUI 聊天界面。

## 运行

```powershell
conda activate lzx
python -m pip install -r requirements.txt
python download_model.py
python app.py
```

打开浏览器访问:

```text
http://127.0.0.1:7860
```

注意：要在 `lzx` 环境中安装 CUDA 11.8 版 `llama-cpp-python`，并已下载 Hrida 模型到 `models/` 目录。

## 文件说明

- `.env`: DeepSeek API、数据库路径、模型路径配置。
- `agent.py`: 双模型协作核心逻辑。
- `app.py`: Gradio WebUI。
- `download_model.py`: 下载 Hrida GGUF 模型。
- `data.db`: 11 表宠物测试数据库。
- `models/Hrida-T2SQL-3B-V0.1_Q4_K_M.gguf`: Hrida 本地模型文件。
- `langgraph_design.md`: Agent03 LangGraph 设计参考文档。
- `实验报告.md`: 实验过程和对比记录。
