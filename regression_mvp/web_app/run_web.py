from __future__ import annotations

import os
import sys
from pathlib import Path

# 切换工作目录到项目根，确保 pydantic-settings 能加载到根目录的 .env
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn


# PyCharm 里直接 Run 当前文件即可启动只读管理页。
HOST = "127.0.0.1"
PORT = 6400
RELOAD = False


def main() -> None:
    uvicorn.run("regression_mvp.web_app.app:app", host=HOST, port=PORT, reload=RELOAD)


if __name__ == "__main__":
    main()
