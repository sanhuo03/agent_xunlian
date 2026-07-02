"""下载 Hrida-T2SQL-3B GGUF 模型到 models/ 目录。"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "HridaAI/Hrida-T2SQL-3B-V0.1-GGUF"
FILENAME = "Hrida-T2SQL-3B-V0.1_Q4_K_M.gguf"
MODEL_DIR = Path(__file__).resolve().parent / "models"


def main() -> int:
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("已设置国内镜像: https://hf-mirror.com")

    MODEL_DIR.mkdir(exist_ok=True)
    print(f"开始下载 {FILENAME} 到 {MODEL_DIR}")
    try:
        path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=MODEL_DIR,
        )
    except KeyboardInterrupt:
        print("下载已中断，重新运行脚本可继续。")
        return 130
    except Exception as exc:
        print(f"下载失败: {exc}")
        print("也可以手动下载后放入 models/ 目录:")
        print(f"https://hf-mirror.com/{REPO_ID}/resolve/main/{FILENAME}")
        return 1

    print(f"下载完成: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
