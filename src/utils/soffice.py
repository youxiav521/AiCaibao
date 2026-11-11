"""LibreOffice(soffice) 调用封装：将 Office 文档转为 PDF。"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from loguru import logger


def resolve_soffice_bin() -> str:
    """解析 soffice 可执行文件路径。

    优先读 SOFFICE_BIN 环境变量，回退 PATH 查找与 Windows 常见安装路径。
    """
    candidates = [
        os.environ.get("SOFFICE_BIN"),
        "soffice",
        "soffice.exe",
        "libreoffice",
    ]
    if platform.system().lower() == "windows":
        candidates.extend(
            [
                r"C:\Program Files\LibreOffice\program\soffice.exe",
                r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
            ]
        )
    for cand in candidates:
        if not cand:
            continue
        if os.path.isabs(cand) and os.path.exists(cand):
            return cand
        found = shutil.which(cand)
        if found:
            return found
    raise FileNotFoundError(
        "未找到 LibreOffice(soffice)；请设置 SOFFICE_BIN 环境变量，"
        "或安装 LibreOffice 并将其加入 PATH"
    )


def convert_to_pdf(
    src_path: str | Path,
    outdir: Optional[str | Path] = None,
    *,
    max_retries: int = 3,
    timeout: int = 60,
) -> Path:
    """用 soffice 将 Office 文档转为 PDF，返回生成的 PDF 路径。

    Args:
        src_path: 源 Office 文件路径（Word/Excel 等）。
        outdir: 输出目录，None 时用源文件所在目录；目标 PDF 已存在则直接返回。
        max_retries: 最大重试次数（指数退避 1s/2s/4s）。
        timeout: 单次 soffice 调用超时秒数。

    每次尝试用独立 UserInstallation 目录，避免 LibreOffice 多实例冲突。
    """
    src = Path(src_path)
    out_dir = Path(outdir) if outdir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / f"{src.stem}.pdf"
    if target_path.exists():
        return target_path

    soffice_bin = resolve_soffice_bin()
    is_windows = platform.system().lower() == "windows"
    base_delay = 1.0
    last_error: Optional[Exception] = None

    for attempt in range(max_retries):
        user_install_dir = Path(tempfile.gettempdir()) / f"soffice_user_{uuid.uuid4().hex[:8]}"
        user_install_uri = user_install_dir.resolve().as_uri()
        try:
            run_kwargs = {}
            if is_windows and hasattr(subprocess, "CREATE_NO_WINDOW"):
                run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            subprocess.run(
                [
                    soffice_bin,
                    "--headless",
                    f"-env:UserInstallation={user_install_uri}",
                    "--convert-to",
                    "pdf",
                    str(src),
                    "--outdir",
                    str(out_dir),
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                **run_kwargs,
            )
            if target_path.exists():
                return target_path
            raise FileNotFoundError("soffice 返回成功但未生成目标 PDF 文件")
        except subprocess.CalledProcessError as exc:
            last_error = exc
            error_msg = exc.stderr.decode(errors="ignore")[:200] if exc.stderr else str(exc)
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "soffice 转 PDF 失败，将重试 | attempt={}, delay_s={:.1f}, error={}",
                    attempt + 1, delay, error_msg,
                )
                time.sleep(delay)
            else:
                logger.error("soffice 转 PDF 达最大重试次数 | attempts={}", max_retries)
        except subprocess.TimeoutExpired as exc:
            last_error = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "soffice 转 PDF 超时(>{}s)，将重试 | attempt={}, delay_s={:.1f}",
                    timeout, attempt + 1, delay,
                )
                time.sleep(delay)
            else:
                logger.error("soffice 转 PDF 达最大重试次数 | attempts={}", max_retries)
        except (OSError, FileNotFoundError) as exc:
            last_error = exc
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "soffice 转 PDF 异常，将重试 | attempt={}, delay_s={:.1f}, error={}",
                    attempt + 1, delay, exc,
                )
                time.sleep(delay)
            else:
                logger.error("soffice 转 PDF 达最大重试次数 | attempts={}", max_retries)
        finally:
            if user_install_dir.exists():
                shutil.rmtree(user_install_dir, ignore_errors=True)

    error_detail = ""
    if isinstance(last_error, subprocess.CalledProcessError) and last_error.stderr:
        error_detail = last_error.stderr.decode(errors="ignore")
    elif isinstance(last_error, subprocess.TimeoutExpired):
        error_detail = f"转换超时 (>{timeout}s)"
    else:
        error_detail = str(last_error) if last_error else "未知错误"
    raise RuntimeError(
        f"soffice 转 PDF 失败（已重试 {max_retries} 次）：{error_detail}"
    ) from last_error
