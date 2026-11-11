"""MinerU 解析服务客户端

功能：
- 通过 HTTP 调用外部 MinerU 解析服务（例如 `http://HOST:PORT/file_parse`）。
- 使用短生命周期临时文件作为解析工作区，解析结果与图片统一持久化到 MinIO。

注意：
- 默认端点来自环境变量 MINERU_API_BASE（如 http://10.192.72.13:8000），
"""
import base64
import os
import platform
import asyncio
import shutil
import subprocess
import tempfile
import uuid
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import mimetypes
import zipfile
import io
from src.applications.parsing.utils.convert_minerU_script import convert_mineru_response_async
import httpx
import time
from loguru import logger
from src.applications.parsing.utils.heading_utils import assign_heading_levels, assign_ids_in_flat
from src.applications.parsing.utils.table_classification_utils import (
    classify_tables_async,
    STATEMENT_TABLE_TYPE_LABELS_SET,
    _reclassify_statement_outliers_by_page_cluster,
    _LARGE_TABLE_SET_THRESHOLD,
)
from src.applications.parsing.utils.table_header_detection import (
    adetect_headers_for_items,
    build_header_cells_from_pending,
)
from src.applications.parsing.utils.convert_minerU_script import (
    apply_cross_page_table_merge_from_pdf_info,
    build_table_2d_for_items,
    _mineru_table_block_is_deleted,
)
from src.applications.parsing.utils.mineru_parsers import merge_table as merge_pdf_info_tables
from src.applications.parsing.service.parse_excel_service import ExcelParsingService
from src.applications.parsing.utils.element_merge_utils import merge_short_elements
from src.applications.parsing.service.parse_markdown_service import MarkdownParsingService
from src.applications.parsing.service.parse_txt_service import TxtParsingService
from src.applications.parsing.service.parse_json_service import JsonParsingService
from src.applications.parsing.utils.mineru_storage import (
    MINERU_OUTPUT_BUCKET,
    extract_pdf_info_from_mineru_result,
    load_mineru_items,
    load_mineru_result,
    upload_mineru_items,
    upload_mineru_image,
    upload_mineru_result,
    upload_original_file,
)
from src.applications.parsing.cell_traceability.schema_match.pipeline import cell_traceability
from src.applications.parsing.utils.split_pdf import split_pdf_bytes
import fitz  # PyMuPDF
from src.applications.parsing.utils.storage_utils import generate_file_id
from src.base.storage_service import storage_service
from src.utils import json_utils, StopWatch

# PDF 切分默认每份页数（环境变量 PAGES_PER_PART 缺失或非法时使用）
DEFAULT_PAGES_PER_PART = 100

# Split 并发解析上限：避免外层 split 并发 × 内层 LLM 并发 数量爆炸打爆下游。
# 10 个 split × 各 10 并发 = 峰值 100 路 LLM 调用，极易触发限流雪崩。
# 默认 3：3 split × 10 内层 = 峰值 30 路，与单文件解析量级相当。
DEFAULT_SPLIT_MAX_CONCURRENCY = 3

def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _sort_items_by_bbox_reading_order(items: Any) -> Any:
    if not isinstance(items, list) or len(items) < 2:
        return items

    indexed = list(enumerate(items))

    def _key(pair: tuple[int, Any]) -> tuple[int, float, float, int]:
        idx, it = pair
        if not isinstance(it, dict):
            return (10**9, float("inf"), float("inf"), idx)

        page_key = _safe_int(it.get("page"), 10**9)
        bbox = it.get("bBox") or it.get("bbox")
        if isinstance(bbox, dict):
            top = _safe_float(bbox.get("t"), float("inf"))
            left = _safe_float(bbox.get("l"), float("inf"))
            coord_origin = bbox.get("coord_origin") or bbox.get("coordOrigin")
            if isinstance(coord_origin, str) and coord_origin.upper() == "BOTTOMLEFT":
                top = -top
            return (page_key, top, left, idx)

        return (page_key, float("inf"), float("inf"), idx)

    indexed.sort(key=_key)
    return [it for _, it in indexed]


  
# MinerU 3.x VLM 推理服务地址（兼容旧 SG_LANG_SERVER_URL 环境变量）
MINERU_API_BASE = os.environ.get("MINERU_API_BASE")
if not MINERU_API_BASE:
    raise RuntimeError("MINERU_API_BASE must be set")

# 3.x 推荐 backend: pipeline / hybrid-auto-engine / hybrid-http-client / vlm-auto-engine / vlm-http-client
mineru_backend = os.environ.get("MINERU_BACKEND") or "hybrid-http-client"
logger.info(f"MinerU URL: {MINERU_API_BASE}, backend: {mineru_backend}")

# 表头识别批量模式：将多张表拼入同一个 LLM prompt，大幅减少调用次数。
# 设为 false 可回退到每张表单独调用（便于调试或对比）。
TABLE_HEADER_BATCH_MODE: bool = os.environ.get("TABLE_HEADER_BATCH_MODE", "true").strip().lower() not in ("false", "0", "no")

class MinerUParsingService:
    """面向 MinerU HTTP 服务的解析客户端。"""

    @staticmethod
    def _env_flag_enabled(value: Optional[str], default: bool = True) -> bool:
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 2400,
        endpoint_path: Optional[str] = None,
        bearer_token: Optional[str] = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("MINERU_API_BASE") or  "").rstrip("/")
        self.timeout = timeout
        self.http_trust_env = self._env_flag_enabled(
            os.environ.get("MINERU_HTTP_TRUST_ENV"),
            default=False,
        )
        path = (endpoint_path or os.environ.get("MINERU_ENDPOINT_PATH") or "file_parse").lstrip("/")
        self.endpoint = f"{self.base_url}/{path}"
        file_parse_path = (os.environ.get("MINERU_FILE_PARSE_ENDPOINT_PATH") or "file_parse").lstrip("/")
        tasks_path = (os.environ.get("MINERU_TASKS_ENDPOINT_PATH") or "tasks").lstrip("/")
        self.file_parse_endpoint = f"{self.base_url}/{file_parse_path}"
        self.tasks_endpoint = f"{self.base_url}/{tasks_path}"
        self.headers: Dict[str, str] = {}
        if bearer_token:
            self.headers["Authorization"] = f"Bearer {bearer_token}"
        self.ENABLE_STATEMENTS_CELL_LOCATION = self._env_flag_enabled(
            os.environ.get("ENABLE_STATEMENTS_CELL_LOCATION"),
            default=True,
        )

    @staticmethod
    def _is_true(value: Optional[str], default: bool = False) -> bool:
        if value is None:
            return default
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    async def _wait_task_result(
        self,
        client: httpx.AsyncClient,
        task_id: str,
    ) -> httpx.Response:
        poll_interval = _safe_float(os.environ.get("MINERU_TASK_POLL_INTERVAL_SECONDS"), 1.5)
        if poll_interval <= 0:
            poll_interval = 1.5
        timeout_seconds = _safe_float(os.environ.get("MINERU_TASK_TIMEOUT_SECONDS"), float(self.timeout))
        if timeout_seconds <= 0:
            timeout_seconds = float(self.timeout)
        deadline = time.monotonic() + timeout_seconds

        status_url = f"{self.tasks_endpoint}/{task_id}"
        result_url = f"{self.tasks_endpoint}/{task_id}/result"
        success_states = {"success", "succeeded", "completed", "done", "finished"}
        failed_states = {"failed", "error", "cancelled", "canceled"}

        while True:
            if time.monotonic() > deadline:
                raise TimeoutError(f"MinerU task timeout, task_id={task_id}, timeout={timeout_seconds}s")

            status_resp = await client.get(status_url, headers=self.headers)
            status_resp.raise_for_status()

            status_data: Dict[str, Any] = {}
            try:
                parsed = status_resp.json()
                if isinstance(parsed, dict):
                    status_data = parsed
            except Exception:
                status_data = {}

            state = str(
                status_data.get("status")
                or status_data.get("state")
                or status_data.get("task_status")
                or ""
            ).strip().lower()

            if state in success_states:
                result_resp = await client.get(result_url, headers=self.headers)
                result_resp.raise_for_status()
                return result_resp

            if state in failed_states:
                detail = status_data.get("error") or status_data.get("message") or status_data
                raise RuntimeError(f"MinerU task failed, task_id={task_id}, detail={detail}")

            # 某些部署可能不返回统一状态字段，直接尝试获取结果（未完成时通常 4xx）
            result_probe = await client.get(result_url, headers=self.headers)
            if result_probe.status_code < 400:
                return result_probe

            await asyncio.sleep(poll_interval)


    def _should_enable_heading(self, file_name: str) -> bool:
        """基于原始文件名判断是否启用“标题分级”。

        仅当文件名包含财报关键词时启用：年报、年度报告、半年报、半年度报告、季报、季度报告、财报摘要、年度报告摘要、半年度报告摘要、季度报告摘要。
        """
        try:
            # file_path_or_name 原始文件名
            name = Path(file_name).stem
        except Exception:
            return False
        if not isinstance(name, str) or not name:
            return False            
        keywords = [
            "年报", "年度报告", "半年报", "半年度报告",
            "季报", "季度报告", "财报摘要",
            "年度报告摘要", "半年度报告摘要", "季度报告摘要",
        ]
        return any(kw in name for kw in keywords)

    def _maybe_convert_to_pdf(self, file_path: Path) -> Path:
        """若输入不是 .pdf，则尝试转为 .pdf 并返回新路径；否则原样返回。
        对于其他文件，使用 LibreOffice(soffice) 转换。

        转换失败将抛出 ValueError。
        """
        suffix_lower = file_path.suffix.lower()
        if suffix_lower == ".pdf":
            return file_path

        # 对图片类型，直接返回原文件，避免将图片包裹进 PDF 导致 MinerU 仅识别为单一 image
        image_suffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        if suffix_lower in image_suffixes:
            return file_path

        target_path = file_path.with_suffix(".pdf")
        if target_path.exists():
            return target_path

        is_windows = platform.system().lower() == "windows"
        candidates = [
            os.environ.get("SOFFICE_BIN"),
            "soffice",
            "soffice.exe",
            "libreoffice",
        ]
        if is_windows:
            # 覆盖 LibreOffice 在 Windows 的常见安装路径
            candidates.extend(
                [
                    r"C:\Program Files\LibreOffice\program\soffice.exe",
                    r"C:\Program Files (x86)\LibreOffice\program\soffice.exe",
                ]
            )
        candidates = [c for c in candidates if c]

        soffice_bin: Optional[str] = None
        for cand in candidates:
            path = cand if os.path.isabs(cand) and os.path.exists(cand) else shutil.which(cand)
            if path:
                soffice_bin = path
                break

        if not soffice_bin:
            raise ValueError(
                "检测到非 PDF 文件，但未找到 LibreOffice(soffice) 以自动转换为 PDF；请先安装 LibreOffice 或手动将文件转为 PDF 后再上传。"
            )

        # 重试配置
        max_retries = 3
        base_delay = 1.0  # 初始等待秒数
        convert_timeout = 60  # 单次转换超时（秒）

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            # 为每次尝试创建独立的临时用户目录，避免 LibreOffice 多实例冲突
            user_install_dir = Path(tempfile.gettempdir()) / f"soffice_user_{uuid.uuid4().hex[:8]}"
            user_install_uri = user_install_dir.resolve().as_uri()
            try:
                # 使用 soffice 无头模式转换为 PDF
                run_kwargs: Dict[str, Any] = {}
                if is_windows and hasattr(subprocess, "CREATE_NO_WINDOW"):
                    run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
                subprocess.run(
                    [
                        soffice_bin,
                        "--headless",
                        f"-env:UserInstallation={user_install_uri}",
                        "--convert-to",
                        "pdf",
                        str(file_path),
                        "--outdir",
                        str(file_path.parent),
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=convert_timeout,
                    **run_kwargs,
                )
                # 验证目标文件是否真正生成（soffice 有时返回 0 但未生成文件）
                if target_path.exists():
                    break  # 转换成功，跳出循环
                else:
                    raise FileNotFoundError("soffice 返回成功但未生成目标 PDF 文件")
            except subprocess.CalledProcessError as e:
                last_error = e
                error_msg = e.stderr.decode(errors='ignore')[:200] if e.stderr else str(e)
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # 指数退避: 1s, 2s, 4s
                    logger.warning(
                        f"LibreOffice 转换失败 (尝试 {attempt + 1}/{max_retries})，"
                        f"{delay:.1f}s 后重试: {error_msg}"
                    )
                    time.sleep(delay)
            except subprocess.TimeoutExpired as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"LibreOffice 转换超时 (>{convert_timeout}s，尝试 {attempt + 1}/{max_retries})，"
                        f"{delay:.1f}s 后重试"
                    )
                    time.sleep(delay)
            except (OSError, FileNotFoundError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"LibreOffice 转换异常 (尝试 {attempt + 1}/{max_retries})，"
                        f"{delay:.1f}s 后重试: {e}"
                    )
                    time.sleep(delay)
            finally:
                # 清理临时用户目录
                if user_install_dir.exists():
                    try:
                        shutil.rmtree(user_install_dir, ignore_errors=True)
                    except Exception:
                        pass
        else:
            # 所有重试均失败
            error_detail = ""
            if isinstance(last_error, subprocess.CalledProcessError) and last_error.stderr:
                error_detail = last_error.stderr.decode(errors='ignore')
            elif isinstance(last_error, subprocess.TimeoutExpired):
                error_detail = f"转换超时 (>{convert_timeout}s)"
            else:
                error_detail = str(last_error) if last_error else "未知错误"
            raise ValueError(
                f"文件转换为 PDF 失败（已重试 {max_retries} 次）：{error_detail}"
            ) from last_error

        if not target_path.exists():
            raise ValueError("文件转换后未找到目标 PDF，请检查 LibreOffice 是否可用。")

        return target_path

    async def parse_to_json_async(
        self,
        file_path: str,
        image_output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        file_stem = path.stem  # 用于图片子目录，避免多请求并发覆盖
        owned_image_dir: tempfile.TemporaryDirectory[str] | None = None

        # 若输入不是 PDF，优先转换为 PDF 再上传
        try:
            path = await asyncio.to_thread(self._maybe_convert_to_pdf, path)
        except Exception as e:
            # 明确提示转换失败原因
            raise e

        # MinerU 的 /file_parse 使用字段名 `files`（List[UploadFile]）
        guessed_type, _ = mimetypes.guess_type(str(path))
        content_type = (
            "application/pdf" if path.suffix.lower() == ".pdf" else (guessed_type or "application/octet-stream")
        )
        file_tuple = (path.name, open(path, "rb"), content_type)
        files_payload = [("files", file_tuple)]
      
        if image_output_dir is None:
            owned_image_dir = tempfile.TemporaryDirectory(prefix=f"mineru_images_{file_stem}_")
            image_output_dir = owned_image_dir.name
        os.makedirs(image_output_dir, exist_ok=True)
        
        data_payload = {
            "response_format_zip": "true",
            "return_md": "true",           
            "table_enable": "true",        
            "return_content_list": "true",
            "return_middle_json": "true",
            "return_model_output": "true",
            "return_images": "true",
            "output_dir": image_output_dir,
            "backend": mineru_backend,
            "server_url": MINERU_API_BASE,
            "parse_method": os.environ.get("MINERU_PARSE_METHOD") or "auto",
            "formula_enable": os.environ.get("MINERU_FORMULA_ENABLE") or "true",
        }
        
        use_task_api = self._is_true(os.environ.get("MINERU_USE_TASK_API"), default=False)
        endpoint_lower = self.endpoint.rstrip("/").lower()
        if endpoint_lower.endswith("/tasks"):
            use_task_api = True

        try:
            start_time = time.time()
            async with httpx.AsyncClient(timeout=self.timeout, trust_env=self.http_trust_env) as client:
                if use_task_api:
                    logger.info(f"请求 MinerU: POST {self.tasks_endpoint} files=[{path.name}]")
                    submit_resp = await client.post(
                        self.tasks_endpoint,
                        files=files_payload,
                        data=data_payload,
                        headers=self.headers,
                    )
                    submit_resp.raise_for_status()
                    submit_data = submit_resp.json() if submit_resp.content else {}
                    if not isinstance(submit_data, dict):
                        raise ValueError(f"MinerU task submit response is not dict: {submit_data}")
                    task_id = (
                        submit_data.get("task_id")
                        or submit_data.get("id")
                        or submit_data.get("taskId")
                    )
                    if not task_id:
                        raise ValueError(f"MinerU task submit response missing task_id: {submit_data}")
                    resp = await self._wait_task_result(client, str(task_id))
                else:
                    request_endpoint = self.endpoint or self.file_parse_endpoint
                    logger.info(f"请求 MinerU: POST {request_endpoint} files=[{path.name}]")
                    resp = await client.post(
                        request_endpoint,
                        files=files_payload,
                        data=data_payload,
                        headers=self.headers,
                    )

            duration = time.time() - start_time
            logger.info(f"MinerU 解析请求耗时: {duration:.2f} 秒")

        finally:
            try:
                file_tuple[1].close()
            except Exception:
                pass

        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            detail = (resp.text or "").strip()
            message = f"MinerU API error: {e}; body={detail[:512]}"
            raise httpx.HTTPStatusError(message, request=e.request, response=e.response) from e

        try:
            # 优先按 ZIP 解析 (当 response_format_zip=true 时)
            ct = (resp.headers.get("Content-Type") or "").lower()
            if "application/zip" in ct or "zip" in ct:
                try:
                    with zipfile.ZipFile(io.BytesIO(resp.content)) as z, tempfile.TemporaryDirectory(
                        prefix=f"mineru_extract_{file_stem}_"
                    ) as temp_extract_dir_name:
                        temp_extract_dir = Path(temp_extract_dir_name)
                        z.extractall(temp_extract_dir)
                        logger.info("MinerU 返回 ZIP 已解压至: {}", temp_extract_dir)

                        json_files = list(temp_extract_dir.rglob("*.json"))
                        logger.info(
                            "ZIP 中发现 {} 个 JSON 文件: {}",
                            len(json_files),
                            [jf.name for jf in json_files],
                        )

                        md_files = list(temp_extract_dir.rglob("*.md"))
                        if md_files:
                            logger.info(
                                "ZIP 中发现 {} 个 MD 文件: {}",
                                len(md_files),
                                [mf.name for mf in md_files],
                            )

                        merged_data = {
                            "content_list": None,
                            "model_output": None,
                            "middle_json": None,
                            "md_content": None,
                        }

                        for mf in md_files:
                            try:
                                md_text = mf.read_text(encoding="utf-8").strip()
                                if md_text and not merged_data["md_content"]:
                                    merged_data["md_content"] = md_text
                                    logger.info(
                                        "已从独立 MD 文件读取 md_content: {} ({} 字符)",
                                        mf.name,
                                        len(md_text),
                                    )
                            except Exception as e:
                                logger.warning("读取 MD 文件 {} 失败: {}", mf.name, e)

                        for jf in json_files:
                            try:
                                file_data = json_utils.load(jf)
                                jf_name_lower = jf.name.lower()
                                jf_stem_lower = jf.stem.lower()

                                if "content_list" in jf_name_lower or jf_stem_lower == "content_list":
                                    if isinstance(file_data, list):
                                        merged_data["content_list"] = json_utils.dumps(file_data, ensure_ascii=False)
                                    elif isinstance(file_data, dict) and "content_list" in file_data:
                                        merged_data["content_list"] = json_utils.dumps(file_data["content_list"], ensure_ascii=False)
                                    else:
                                        merged_data["content_list"] = json_utils.dumps(file_data, ensure_ascii=False)
                                    logger.info("已读取 content_list: {}", jf.name)

                                elif "model" in jf_name_lower or jf_stem_lower == "model":
                                    merged_data["model_output"] = json_utils.dumps(file_data, ensure_ascii=False)
                                    logger.info("已读取 model_output: {}", jf.name)

                                elif "middle" in jf_name_lower or jf_stem_lower == "middle_json":
                                    merged_data["middle_json"] = json_utils.dumps(file_data, ensure_ascii=False)
                                    logger.info("已读取 middle_json: {}", jf.name)

                                if isinstance(file_data, dict) and not merged_data["md_content"]:
                                    if "md_content" in file_data:
                                        merged_data["md_content"] = file_data["md_content"]
                                        logger.info("已从 {} 提取 md_content", jf.name)
                                    if "results" in file_data:
                                        results = file_data.get("results", {})
                                        if isinstance(results, dict) and results:
                                            first_key = next(iter(results))
                                            nested = results[first_key]
                                            if isinstance(nested, dict):
                                                if "md_content" in nested and not merged_data["md_content"]:
                                                    merged_data["md_content"] = nested["md_content"]
                                                if "content_list" in nested and not merged_data["content_list"]:
                                                    content_list = nested["content_list"]
                                                    merged_data["content_list"] = content_list if isinstance(content_list, str) else json_utils.dumps(content_list, ensure_ascii=False)
                                                if "model_output" in nested and not merged_data["model_output"]:
                                                    model_output = nested["model_output"]
                                                    merged_data["model_output"] = model_output if isinstance(model_output, str) else json_utils.dumps(model_output, ensure_ascii=False)
                                                if "middle_json" in nested and not merged_data["middle_json"]:
                                                    middle_json = nested["middle_json"]
                                                    merged_data["middle_json"] = middle_json if isinstance(middle_json, str) else json_utils.dumps(middle_json, ensure_ascii=False)
                            except Exception as e:
                                logger.warning("读取 JSON 文件 {} 失败: {}", jf.name, e)

                        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
                        copied_images_count = 0
                        for root, _dirs, files in os.walk(temp_extract_dir):
                            for file in files:
                                if Path(file).suffix.lower() in image_extensions:
                                    src_img = os.path.join(root, file)
                                    dst_img = os.path.join(image_output_dir, file)
                                    shutil.move(src_img, dst_img)
                                    copied_images_count += 1

                        logger.info("已移动 {} 张图片到 {}", copied_images_count, image_output_dir)

                        if merged_data["content_list"] or merged_data["model_output"]:
                            logger.info(
                                "ZIP 解析成功，content_list: {}, model_output: {}, middle_json: {}, md_content: {}",
                                bool(merged_data["content_list"]),
                                bool(merged_data["model_output"]),
                                bool(merged_data["middle_json"]),
                                bool(merged_data["md_content"]),
                            )
                            return merged_data
                        logger.warning("ZIP 解析未找到 content_list 或 model_output，返回 merged_data")
                        return merged_data
                except Exception as e:
                    logger.error("解析 MinerU ZIP 响应失败: {}", e)
                    # 失败继续尝试 JSON 解析

            # 按 JSON 解析
            if "application/json" in ct or "+json" in ct:
                try:
                    data = json_utils.response_json(resp)
                    # 兼容返回 list 的情况，统一包一层
                    if isinstance(data, list):
                        return {"data": data}
                    if isinstance(data, dict):
                        return data
                except Exception:
                    pass

            # 兜底：当作文本返回
            text = resp.text if hasattr(resp, "text") else None
            if text is not None:
                return {"raw_text": text}
            return {"raw_bytes_len": len(resp.content or b"")}
        finally:
            if owned_image_dir is not None:
                owned_image_dir.cleanup()

    async def parse_to_items_async(
        self,
        file_path: str,
        file_id: Any = None,
        output_dir: Optional[str] = None,
        image_output_dir: Optional[str] = None,
        use_llm_heading: bool = False
    ) -> tuple[List[Dict[str, Any]], Any]:
        _ = output_dir
        payload: Any = await self.parse_to_json_async(
            file_path,
            image_output_dir=image_output_dir,
        )

        if isinstance(payload, dict) and "raw_text" in payload:
            raw = payload.get("raw_text")
            try:
                payload = json_utils.loads(raw) if isinstance(raw, str) else payload
            except Exception:
                payload = []

        items = await convert_mineru_response_async(
            payload,
            file_id=file_id,
            use_llm_heading=use_llm_heading
        )
        return items, payload

    @staticmethod
    def _build_mineru_result_payload(result_key: str, md_content: str) -> Dict[str, Any]:
        return {
            "results": {
                result_key: {
                    "md_content": md_content,
                }
            }
        }

    def _parse_local_file_to_items_sync(
        self,
        file_path: str,
        ext_lower: str,
        mineru_dir: Path | None,
        result_key: str,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        in_path = Path(file_path)

        if ext_lower in {".xlsx", ".csv"}:
            logger.info("检测到 Excel 文件，走 ExcelParsingService 解析路径")
            items = ExcelParsingService().parse_to_items(file_path)
                        # 拼接所有表格的 md 为完整 md_content
            md_parts = []
            for item in items:
                sheet_name = item.get("sheetName", "")
                md = item.get("md", "")
                if md:
                    # 添加工作表标题和表格内容
                    md_parts.append(f"## {sheet_name}\n\n{md}")
            
            full_md_content = "\n\n---\n\n".join(md_parts)
        elif ext_lower in {".md", ".markdown"}:
            logger.info("检测到 Markdown 文件，走 MarkdownParsingService 解析路径")
            items = MarkdownParsingService().parse_to_items(file_path)
            
            # 构造完整 md_content
            full_md_content = "\n\n".join([item.get("md", "") for item in items if item.get("md")])
        elif ext_lower == ".txt":
            logger.info("检测到 TXT 文件，走 TxtParsingService 解析路径")
            items = TxtParsingService().parse_to_items(file_path)
            
            # 构造完整 md_content
            full_md_content = "\n\n".join([item.get("value", "") for item in items if item.get("value")])
        elif ext_lower == ".json":
            logger.info("检测到 JSON 文件，走 JsonParsingService 解析路径")
            items = JsonParsingService().parse_to_items(file_path)
            
            # 构造完整 md_content (JSON 的 md 字段通常是原字符串)
            full_md_content = "\n\n".join([item.get("value", "") for item in items if item.get("value")])
        else:
            raise ValueError(f"不支持的本地解析文件后缀: {ext_lower}")

        mineru_json = self._build_mineru_result_payload(result_key, full_md_content)
        return items, mineru_json

    @staticmethod
    def _normalize_mineru_result_payload_sync(
        raw_mineru_data: Any,
        items: List[Dict[str, Any]],
        result_key: str,
    ) -> Dict[str, Any]:
        md_content = None
        middle_json_str = None

        if isinstance(raw_mineru_data, dict):
            if "results" in raw_mineru_data:
                results = raw_mineru_data.get("results", {})
                if results:
                    first_key = next(iter(results))
                    file_data = results[first_key]
                    if isinstance(file_data, dict):
                        md_content = file_data.get("md_content")
                        middle_json_str = file_data.get("middle_json")
            elif "md_content" in raw_mineru_data:
                md_content = raw_mineru_data.get("md_content")
                middle_json_str = raw_mineru_data.get("middle_json")
            else:
                md_content = "\n\n".join([item.get("md", "") for item in items if item.get("md")])
        elif isinstance(raw_mineru_data, list):
            md_content = "\n\n".join([item.get("md", "") for item in items if item.get("md")])

        if not md_content:
            md_content = "\n\n".join([item.get("md", "") for item in items if item.get("md")])

        content_list_str = None
        model_output_str = None
        if isinstance(raw_mineru_data, dict):
            content_list_str = raw_mineru_data.get("content_list")
            model_output_str = raw_mineru_data.get("model_output")
            if not middle_json_str:
                middle_json_str = raw_mineru_data.get("middle_json")

        pdf_mineru_json = {
            "results": {
                result_key: {
                    "md_content": md_content,
                }
            }
        }
        result_data = pdf_mineru_json["results"][result_key]
        if middle_json_str:
            result_data["middle_json"] = middle_json_str
        if content_list_str:
            result_data["content_list"] = content_list_str
        if model_output_str:
            result_data["model_output"] = model_output_str

        return pdf_mineru_json

    @staticmethod
    def _postprocess_items_sync(
        items: List[Dict[str, Any]],
        resolved_file_id: Any,
        folder_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        # 纯文本解析产物（TXT/JSON）没有真实页码与坐标，统一补齐为
        # page=1 + 基于 startRow/endRow 的伪 bBox，保持与 MinerU text item 契约一致。
        if isinstance(items, list):
            for _it in items:
                if not isinstance(_it, dict) or _it.get("type") != "text":
                    continue
                if not isinstance(_it.get("page"), int):
                    _it["page"] = 1
                if not isinstance(_it.get("bBox"), dict) and not isinstance(_it.get("bbox"), dict):
                    start_row = _it.get("startRow") or _it.get("endRow") or 1
                    end_row = _it.get("endRow") or start_row
                    try:
                        top = float(start_row)
                    except (TypeError, ValueError):
                        top = 0.0
                    try:
                        bottom = float(end_row)
                    except (TypeError, ValueError):
                        bottom = top
                    _it["bBox"] = {
                        "l": 0.0,
                        "t": top,
                        "r": 1000.0,
                        "b": bottom,
                        "coord_origin": "TOPLEFT",
                    }

        try:
            items = _sort_items_by_bbox_reading_order(items)
        except Exception:
            pass

        # 后处理步骤1：合并短且语义不完整的 element（在分配 element_id 之前）
        if items:
            try:
                items = merge_short_elements(
                    items,
                    min_length=8,
                    merge_types={"text", "list"},
                )
            except Exception as e:
                logger.warning(f"后处理步骤1 - 短 element 合并失败: {e}")

        # 为每个元素分配稳定的 element_id，并注入 file_id / folder_id（与列表顺序一致，从 1 开始）
        try:
            if isinstance(items, list):
                for _idx, _it in enumerate(items):
                    if not isinstance(_it, dict):
                        continue
                    _it["element_id"] = _idx + 1
                    _it["file_id"] = resolved_file_id
                    _it["folder_id"] = folder_id
        except Exception:
            pass

        return items

    @staticmethod
    def _inject_table_cell_source_page_no_sync(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """给 table.rows 中每个单元格补充 source_page_no=所属表格 page。"""
        if not isinstance(items, list):
            return items

        for it in items:
            if not isinstance(it, dict) or it.get("type") != "table":
                continue

            source_page_no = _safe_int(it.get("page"), 0)
            rows = it.get("rows")
            if not isinstance(rows, list):
                continue

            for row in rows:
                if not isinstance(row, list):
                    continue
                for cell in row:
                    if not isinstance(cell, dict):
                        continue
                    # 与 text 并列写入，方便下游按 cell 直接读取页码。
                    cell["source_page_no"] = source_page_no

        return items

    @staticmethod
    def _extract_pdf_info_from_mineru_result_sync(
        mineru_data: Any,
        result_key: Optional[str] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        return extract_pdf_info_from_mineru_result(mineru_data, file_id=result_key)

    @staticmethod
    def _rename_table_images_by_img_path_sync(
        images_dir: Path,
        items: List[Dict[str, Any]],
    ) -> Optional[List[str]]:
        """根据 items 中 table 的 img_path，将 images_dir 下对应图片重命名为 {file_id}_{element_id:04d}.{ext}。

        返回重命名后的有序绝对路径列表（与 items 中 table 出现顺序一致）；
        个别 table 的 img_path 无效（缺失、目录路径、文件不存在）时跳过该表格，
        不影响其余表格。全部 table 均失败时返回 None。
        """
        tables = [
            it for it in items
            if isinstance(it, dict) and it.get("type") == "table"
        ]
        if not tables:
            return None

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        ordered: List[str] = []
        for tbl in tables:
            img_path_raw = tbl.get("img_path")
            if not img_path_raw or not isinstance(img_path_raw, str):
                logger.warning(f"table element_id={tbl.get('element_id')} 缺少 img_path，跳过该表格重命名")
                continue

            basename = Path(img_path_raw).name
            # 跳过目录路径（如 "images/"）等无效 img_path
            if Path(basename).suffix.lower() not in image_extensions:
                logger.warning(f"table element_id={tbl.get('element_id')} img_path 非有效图片文件: {img_path_raw}，跳过该表格重命名")
                continue

            src = images_dir / basename
            if not src.exists():
                logger.warning(f"img_path 指向的文件不存在: {src}，跳过该表格重命名")
                continue

            fid = tbl.get("file_id", "unknown")
            eid = _safe_int(tbl.get("element_id"), 0)
            dst = images_dir / f"{fid}_{eid}{src.suffix.lower()}"
            shutil.move(str(src), str(dst))
            tbl["img_path"] = dst.name
            ordered.append(str(dst))
        return ordered if ordered else None

    @staticmethod
    def _build_cross_page_chains_sync(
        items: List[Dict[str, Any]],
        pdf_info_raw: Optional[List[Dict[str, Any]]],
    ) -> List[set[Any]]:
        """基于 merge_table 的续表打标构建跨页链（element_id 集合）。"""
        if not isinstance(items, list) or not isinstance(pdf_info_raw, list) or not pdf_info_raw:
            return []

        pdf_info_for_chain = deepcopy(pdf_info_raw)
        try:
            merge_pdf_info_tables(pdf_info_for_chain)
        except Exception as e:
            logger.warning(f"识别跨页链失败，merge_table 执行异常: {e}")
            return []

        continuation_pages: set[int] = set()
        for page_idx, page_obj in enumerate(pdf_info_for_chain):
            if not isinstance(page_obj, dict):
                continue
            para_blocks = page_obj.get("para_blocks") or []
            if not isinstance(para_blocks, list) or not para_blocks:
                continue
            first_block = para_blocks[0]
            if not isinstance(first_block, dict) or first_block.get("type") != "table":
                continue
            # 首表被标记为 deleted 表示“当前页首表属于续表”，应与上一页末表成链。
            if _mineru_table_block_is_deleted(first_block):
                continuation_pages.add(page_idx + 1)

        page_to_tables: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for it in items:
            if not isinstance(it, dict) or it.get("type") != "table":
                continue
            page_no = it.get("page")
            if isinstance(page_no, int) and page_no >= 1:
                page_to_tables[page_no].append(it)

        parent: Dict[Any, Any] = {}
        rank: Dict[Any, int] = {}

        def _find(x: Any) -> Any:
            if x not in parent:
                parent[x] = x
                rank[x] = 0
            # 并查集路径压缩，降低后续查询成本。
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a: Any, b: Any) -> None:
            ra, rb = _find(a), _find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1

        for page_no in sorted(continuation_pages):
            prev_tables = page_to_tables.get(page_no - 1) or []
            cur_tables = page_to_tables.get(page_no) or []
            if not prev_tables or not cur_tables:
                continue
            prev_eid = prev_tables[-1].get("element_id")
            cur_eid = cur_tables[0].get("element_id")
            if prev_eid is None or cur_eid is None:
                continue
            _union(prev_eid, cur_eid)

        groups_map: Dict[Any, set[Any]] = defaultdict(set)
        for eid in list(parent.keys()):
            groups_map[_find(eid)].add(eid)

        chains = list(groups_map.values())
        if chains:
            logger.info(f"识别到跨页链 {len(chains)} 条")
        return chains

    @staticmethod
    def _collect_type_meta_from_items_sync(items: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
        eid_to_meta: Dict[Any, Dict[str, Any]] = {}
        if not isinstance(items, list):
            return eid_to_meta
        for el in items:
            if not isinstance(el, dict) or el.get("type") not in {"table", "excel_table"}:
                continue
            eid = el.get("element_id")
            ttype = el.get("table_type")
            if eid is None or not isinstance(ttype, str) or not ttype:
                continue
            eid_to_meta[eid] = {
                "table_type": ttype,
                "table_type_is_full_statement": bool(el.get("table_type_is_full_statement", False)),
            }
        return eid_to_meta

    @staticmethod
    def _expand_type_meta_by_chains_sync(
        eid_to_meta: Dict[Any, Dict[str, Any]],
        chains: List[set[Any]],
    ) -> None:
        """将链上已识别的主表类型扩散到续页片段，避免续页 element 丢失标签。"""
        if not eid_to_meta or not chains:
            return
        for chain_eids in chains:
            chain_meta: Optional[Dict[str, Any]] = None
            for eid in chain_eids:
                meta = eid_to_meta.get(eid)
                if not meta:
                    continue
                if meta.get("table_type") in STATEMENT_TABLE_TYPE_LABELS_SET:
                    chain_meta = meta
                    break
                if chain_meta is None:
                    chain_meta = meta
            if not chain_meta:
                continue
            for eid in chain_eids:
                eid_to_meta[eid] = dict(chain_meta)

    @staticmethod
    def _apply_type_meta_to_items_sync(
        items: List[Dict[str, Any]],
        eid_to_meta: Dict[Any, Dict[str, Any]],
    ) -> int:
        if not isinstance(items, list) or not eid_to_meta:
            return 0
        applied = 0
        for el in items:
            if not isinstance(el, dict) or el.get("type") not in {"table", "excel_table"}:
                continue
            eid = el.get("element_id")
            meta = eid_to_meta.get(eid)
            if not meta:
                continue
            table_type = meta.get("table_type")
            if not isinstance(table_type, str) or not table_type:
                continue
            el["table_type"] = table_type
            el["table_type_is_full_statement"] = bool(meta.get("table_type_is_full_statement", False))
            el.pop("table_type_evidence", None)
            applied += 1
        return applied

    @staticmethod
    def _select_statement_trace_images_sync(
        items: List[Dict[str, Any]],
        ordered_imgs: List[str],
        images_dir: Path,
    ) -> List[str]:
        """按 table_type 过滤三大表图片，输出顺序与 ordered_imgs 保持一致。"""
        if not isinstance(items, list) or not ordered_imgs:
            return []

        ordered_set = set(ordered_imgs)
        selected: List[str] = []
        seen: set[str] = set()

        for tbl in items:
            if not isinstance(tbl, dict) or tbl.get("type") != "table":
                continue
            if tbl.get("table_type") not in STATEMENT_TABLE_TYPE_LABELS_SET:
                continue

            img_path_val = tbl.get("img_path")
            if not isinstance(img_path_val, str) or not img_path_val.strip():
                continue

            path_obj = Path(img_path_val)
            candidate = str(path_obj) if path_obj.is_absolute() else str(images_dir / path_obj.name)

            if candidate not in ordered_set:
                # 兜底：某些场景 img_path 目录前缀不一致，按文件名回查一次。
                img_name = path_obj.name.lower()
                candidate = next((p for p in ordered_imgs if Path(p).name.lower() == img_name), "")
                if not candidate:
                    continue

            if candidate not in seen:
                seen.add(candidate)
                selected.append(candidate)

        return selected

    @staticmethod
    def _should_split_pdf(file_path: str, original_filename: Optional[str]) -> bool:
        """判断是否应对当前文件启用 PDF 切分。

        条件：
          1. 环境变量 ``ENABLE_SPLIT`` 严格为真（``true/1/yes/on``）；
          2. 文件后缀为 ``.pdf``（优先用 original_filename，回退到 file_path）。
        """
        enable_split = os.environ.get("ENABLE_SPLIT")
        if not MinerUParsingService._is_true(enable_split, default=False):
            return False

        name_to_check = original_filename or file_path or ""
        try:
            ext = Path(name_to_check).suffix.lower()
        except Exception:
            return False
        return ext == ".pdf"

    @staticmethod
    async def _upload_part_bytes_to_minio_temp(
        part_bytes: bytes,
        part_filename: str,
        content_type: str = "application/pdf",
    ) -> Tuple[str, str, str]:
        """将切分后的 PDF 字节流持久化到 MinIO 并落本地临时文件。

        复刻 ``upload_file_to_minio_temp`` 的逻辑，但输入是 bytes（service 层不依赖 fastapi）。
        返回 ``(file_id, local_path, filename)``。
        """
        file_id = generate_file_id()
        ext = Path(part_filename).suffix

        await upload_original_file(file_id, part_filename, part_bytes, content_type)

        fd, local_path = tempfile.mkstemp(prefix=f"{file_id}_", suffix=ext)
        try:
            with os.fdopen(fd, "wb") as buffer:
                buffer.write(part_bytes)
        except Exception:
            os.close(fd)
            raise

        return file_id, local_path, part_filename

    async def _split_and_parse_async(
        self,
        file_path: str,
        original_file_id: Optional[str],
        original_filename: str,
        folder_id: Optional[str],
        locale: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """拆分 PDF 并并发解析各 part，最后合并结果到 ``original_file_id`` 位置。

        返回 ``None`` 表示未执行拆分或拆分失败（调用方应回退到原文件解析）；
        返回 dict 表示拆分+合并成功，结构与 :meth:`parse_to_items_and_save_async` 一致。
        """
        # 1. 读取文件字节并检测页数
        try:
            file_bytes = await asyncio.to_thread(Path(file_path).read_bytes)
        except Exception:
            logger.warning("PDF 切分读取文件失败 | path={}", file_path)
            return None

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            page_count = doc.page_count
            doc.close()
        except Exception:
            logger.exception("PDF 页数检测失败，跳过拆分 | filename={}", original_filename)
            return None

        if page_count <= 0:
            return None

        # 2. 解析 pages_per_part 参数
        pages_per_part = _safe_int(os.environ.get("PAGES_PER_PART"), DEFAULT_PAGES_PER_PART)
        if pages_per_part < 1:
            logger.warning(
                "pages_per_part 非法，重置为默认值 | input={}, default={}",
                pages_per_part, DEFAULT_PAGES_PER_PART,
            )
            pages_per_part = DEFAULT_PAGES_PER_PART

        if pages_per_part >= page_count:
            logger.info(
                "pages_per_part >= 总页数，不拆分 | pages_per_part={}, page_count={}",
                pages_per_part, page_count,
            )
            return None

        # 3. 拆分 PDF
        try:
            parts = split_pdf_bytes(file_bytes, pages_per_part=pages_per_part)
        except Exception:
            logger.exception("PDF 拆分失败，仅处理原文件 | filename={}", original_filename)
            return None

        if not parts:
            return None

        # 4. 上传各 part 到 MinIO 并收集 (file_id, local_path, filename, part_info)
        stem = Path(original_filename).stem or "document"
        split_uploads: List[Tuple[str, str, str, Dict[str, Any]]] = []
        try:
            for idx, (part_bytes, part_info) in enumerate(parts, start=1):
                part_filename = f"{stem}_part{idx}.pdf"
                part_id, part_path, part_name = await self._upload_part_bytes_to_minio_temp(
                    part_bytes,
                    part_filename,
                    "application/pdf",
                )
                split_uploads.append((part_id, part_path, part_name, part_info))
        except Exception:
            logger.exception("Split part 上传失败，回退到原文件处理 | filename={}", original_filename)
            for sid, spath, _, _ in split_uploads:
                try:
                    Path(spath).unlink(missing_ok=True)
                except Exception:
                    pass
            return None

        split_ids = [t[0] for t in split_uploads]
        logger.info(
            "PDF 已拆分，拼接结果将写入原文件位置 | original_file_id={}, original_filename={}, split_file_ids={}",
            original_file_id, original_filename, split_ids,
        )

        # 5. 并发解析各 part（任一失败则整体回退）
        # 外层用信号量限制同时进行的 split 数，避免 split 数 × 内层 LLM 并发数 爆炸
        # 打爆 MinerU / LLM 网关。默认 3，可通过环境变量 SPLIT_MAX_CONCURRENCY 调整。
        split_max_concurrency = _safe_int(
            os.environ.get("SPLIT_MAX_CONCURRENCY"), DEFAULT_SPLIT_MAX_CONCURRENCY
        )
        if split_max_concurrency < 1:
            split_max_concurrency = DEFAULT_SPLIT_MAX_CONCURRENCY
        split_semaphore = asyncio.Semaphore(split_max_concurrency)
        logger.info(
            "Split 并发解析开始 | original_file_id={}, split_count={}, max_concurrency={}",
            original_file_id, len(split_uploads), split_max_concurrency,
        )

        async def _parse_one_split(split_id: str, split_path: str, split_name: str) -> bool:
            async with split_semaphore:
                try:
                    await self.parse_to_items_and_save_async(
                        split_path,
                        file_id=split_id,
                        classify_tables=True,
                        original_filename=split_name,
                        folder_id=folder_id,
                        locale=locale,
                        _skip_split=True,
                        skip_page_cluster=True,
                    )
                    return True
                except Exception:
                    logger.warning("Split 解析失败 | split_id={}", split_id)
                    return False
                finally:
                    try:
                        Path(split_path).unlink(missing_ok=True)
                    except Exception:
                        logger.warning("Split 临时文件清理失败 | path={}", split_path)

        results = await asyncio.gather(*[
            _parse_one_split(sid, spath, sname)
            for sid, spath, sname, _ in split_uploads
        ])

        if not all(results):
            failed = [split_uploads[i][0] for i, r in enumerate(results) if not r]
            logger.warning(
                "存在 split 解析失败，回退到原文件处理 | file_id={}, failed_splits={}",
                original_file_id, failed,
            )
            return None

        # 6. 合并结果到 original_file_id
        try:
            await splice_split_results_async(
                original_file_id=original_file_id or "",
                split_file_ids=split_ids,
                split_part_infos=[t[3] for t in split_uploads],
            )
        except Exception:
            logger.warning("Split 结果拼接失败，回退到原文件处理 | file_id={}", original_file_id)
            return None

        # 7. 加载合并后的结果并返回（保持返回结构与原文件解析一致）
        try:
            merged_items = await load_mineru_items(original_file_id)
            merged_result = await load_mineru_result(original_file_id)
        except Exception:
            logger.exception("加载合并后的 split 结果失败 | file_id={}", original_file_id)
            return None

        # 8. 合并后对全文档执行页码主区间后处理（分片时各 part 已跳过此步骤）
        table_count = sum(
            1 for el in merged_items
            if isinstance(el, dict) and el.get("type") in {"table", "excel_table"}
        )
        if table_count >= _LARGE_TABLE_SET_THRESHOLD:
            try:
                adjusted = _reclassify_statement_outliers_by_page_cluster(merged_items)
                if adjusted:
                    await upload_mineru_items(original_file_id, merged_items)
                    logger.info(
                        "Split 合并后页码主区间后处理: file_id={}, 表格数={}, 区间外改为其他表={}",
                        original_file_id, table_count, adjusted,
                    )
            except Exception:
                logger.warning("Split 合并后页码主区间后处理失败 | file_id={}", original_file_id)

        logger.info(
            "Split 结果拼接完成 | original_file_id={}, split_file_ids={}, total_items={}",
            original_file_id, split_ids, len(merged_items),
        )

        return {
            "items": merged_items,
            "mineru_result": merged_result,
        }

    async def parse_to_items_and_save_async(
        self,
        file_path: str,
        output_dir: Optional[str] = None,
        file_id: Any = None,
        classify_tables: bool = False,
        force_classify: bool = False,
        max_concurrency: int = 5,
        original_filename: Optional[str] = None,
        folder_id: Optional[str] = None,
        locale: Optional[str] = None,
        _skip_split: bool = False,
        skip_page_cluster: bool = False,
    ) -> Dict[str, Any]:
        # PDF 切分：满足条件时拆分原文件并发解析，任一环节失败均回退到下方原文件解析。
        # _skip_split=True 时跳过（用于递归处理 split part，避免二次切分）。
        if not _skip_split and self._should_split_pdf(file_path, original_filename):
            try:
                split_result = await self._split_and_parse_async(
                    file_path=file_path,
                    original_file_id=str(file_id) if file_id is not None else None,
                    original_filename=original_filename or "",
                    folder_id=folder_id,
                    locale=locale,
                )
                if split_result is not None:
                    return split_result
            except Exception:
                logger.exception("PDF 切分处理失败，回退到原文件处理 | file_id={}", file_id)

        sw = StopWatch(f"recognize_{file_path}")
        
        in_path = Path(file_path)
        ext_lower = in_path.suffix.lower()
        is_local_file = ext_lower in {".xlsx", ".csv", ".md", ".markdown", ".txt", ".json"}

        _ = output_dir

        result_key = str(file_id or in_path.stem)
        resolved_file_id = file_id if file_id is not None else in_path.stem
        image_tmp = tempfile.TemporaryDirectory(prefix=f"mineru_images_{in_path.stem}_")
        image_output_dir = image_tmp.name
        images_dir = Path(image_output_dir).resolve()

        if True:
            if is_local_file:
                with sw.task("parse_local_file"):
                    items, mineru_result = await asyncio.to_thread(
                        self._parse_local_file_to_items_sync,
                        file_path,
                        ext_lower,
                        None,
                        result_key,
                    )
            else:
                is_financial_report = self._should_enable_heading(original_filename)
                if is_financial_report:
                    with sw.task("parse_financial_report"):
                        items, raw_mineru_result = await self.parse_to_items_async(
                            file_path,
                            file_id=file_id,
                            image_output_dir=image_output_dir,
                            use_llm_heading=False,
                        )
                    mineru_result = await asyncio.to_thread(
                        self._normalize_mineru_result_payload_sync,
                        raw_mineru_result,
                        items,
                        result_key,
                    )
                    try:
                        if assign_heading_levels is not None:
                            items = await asyncio.to_thread(assign_heading_levels, items)
                    except Exception as e:
                        logger.warning(f"规则方法标题分级失败: {e}")
                else:
                    with sw.task("parse_non_financial_report"):
                        items, raw_mineru_result = await self.parse_to_items_async(
                            file_path,
                            file_id=file_id,
                            image_output_dir=image_output_dir,
                            use_llm_heading=True,
                        )
                    mineru_result = await asyncio.to_thread(
                        self._normalize_mineru_result_payload_sync,
                        raw_mineru_result,
                        items,
                        result_key,
                    )

            items = await asyncio.to_thread(
                self._postprocess_items_sync,
                items,
                resolved_file_id,
                folder_id,
            )

        # 基础后处理后，补充 table.rows 中每个单元格的 source_page_no。
        # 特殊处理，单元格记录文件的物理页码，解决表格跨页合并时会丢失物理页码的问题，为了 cell 溯源用。
        items = await asyncio.to_thread(self._inject_table_cell_source_page_no_sync, items)
        
        pdf_info_raw: Optional[List[Dict[str, Any]]] = None
        header_done: bool = False

        # 执行单元格溯源功能（直接调用 pipeline）
        try:
            table_count = sum(1 for item in items if isinstance(item, dict) and item.get('type') == 'table')
            if table_count > 0:
                if not images_dir.exists():
                    logger.warning("图片目录不存在或为空，跳过单元格溯源")
                else:
                    try:
                        # 重命名图片
                        ordered_imgs = await asyncio.to_thread(
                            self._rename_table_images_by_img_path_sync, images_dir, items,
                        )
                        if not ordered_imgs:
                            logger.warning("img_path 重命名失败，跳过单元格溯源")
                        else:
                            trace_imgs = ordered_imgs

                            if self.ENABLE_STATEMENTS_CELL_LOCATION:
                                try:
                                    if pdf_info_raw is None:
                                        pdf_info_raw = await asyncio.to_thread(
                                            self._extract_pdf_info_from_mineru_result_sync,
                                            mineru_result,
                                            result_key,
                                        )
                                    if pdf_info_raw:
                                        # 第一步：先识别跨页链（不改正式 items）。
                                        chains = await asyncio.to_thread(
                                            self._build_cross_page_chains_sync,
                                            items,
                                            pdf_info_raw,
                                        )

                                        # 第二步：在副本上做“跨页合并 + 分类”，拿到更稳定的三大表判定。
                                        items_for_classify = deepcopy(items)
                                        pdf_info_for_classify = deepcopy(pdf_info_raw)
                                        items_for_classify = await asyncio.to_thread(
                                            apply_cross_page_table_merge_from_pdf_info,
                                            items_for_classify,
                                            pdf_info_for_classify,
                                        )
                                        with sw.task("classify_tables"):
                                            await classify_tables_async(
                                                items_for_classify,
                                                force_reclassify=False,
                                                max_concurrency=max_concurrency,
                                                locale=locale,
                                                skip_page_cluster=skip_page_cluster,
                                            )

                                        # 第三步：收集分类结果并沿跨页链扩散到续页片段。
                                        eid_to_meta = await asyncio.to_thread(
                                            self._collect_type_meta_from_items_sync,
                                            items_for_classify,
                                        )
                                        await asyncio.to_thread(
                                            self._expand_type_meta_by_chains_sync,
                                            eid_to_meta,
                                            chains,
                                        )
                                        await asyncio.to_thread(
                                            self._apply_type_meta_to_items_sync,
                                            items,
                                            eid_to_meta,
                                        )

                                        trace_imgs = await asyncio.to_thread(
                                            self._select_statement_trace_images_sync,
                                            items,
                                            ordered_imgs,
                                            images_dir,
                                        )
                                        if not trace_imgs:
                                            logger.warning("三大表结果为空，回退全量溯源")
                                            trace_imgs = ordered_imgs
                                    else:
                                        logger.warning("未获取到 pdf_info，回退全量溯源")
                                except Exception as e:
                                        logger.warning(f"三大表识别异常，回退全量溯源: {e}")
                                        trace_imgs = ordered_imgs

                            trace_target_table_count = len(trace_imgs) if isinstance(trace_imgs, list) else 0

                            # 并行执行：cell_traceability（写 bBox/cellTraceabilitySource）
                            # 与 detect_headers_for_items_concurrent（写 standard2D/hasHeader/headerCells）
                            async def _traceability_task():
                                with sw.task("cell_traceability") as t:
                                    t.detail(f"table: {trace_target_table_count}")
                                    return await cell_traceability(images=trace_imgs, input_json=items)

                            async def _pre_header_task():
                                with sw.task("detect_headers_for_items_concurrent") as t:
                                    total, detected = await adetect_headers_for_items(
                                        items,
                                        model_id="model_for_table_header_aided",
                                        max_concurrency=10,
                                        batch_mode=TABLE_HEADER_BATCH_MODE,
                                    )
                                    t.detail(f"total: {total}, detected: {detected}")
                                return (total, detected)

                            trace_result, header_stats = await asyncio.gather(
                                _traceability_task(),
                                _pre_header_task(),
                                return_exceptions=False,
                            )
                            result = trace_result
                            header_done = True
                            if result and result.get("content"):
                                items = result["content"]
                            elif result and result.get("tables_processed", 0) == 0:
                                logger.warning("单元格溯源未匹配到任何表格")
                    except Exception as e:
                        logger.error(f"单元格溯源异常: {e}", exc_info=True)
            else:
                logger.info("未发现表格，跳过单元格溯源")
        except Exception as e:
            logger.error(f"单元格溯源功能执行失败: {e}", exc_info=True)

        # 跨页表格合并
        try:
            # 直接使用内存中的 items（已包含单元格溯源结果，若有）
            # 正式合并阶段使用新的 pdf_info 副本，避免复用被 merge_table 污染的对象。
            pdf_info = deepcopy(pdf_info_raw) if isinstance(pdf_info_raw, list) else None
            if pdf_info is None:
                try:
                    pdf_info = await asyncio.to_thread(
                        self._extract_pdf_info_from_mineru_result_sync,
                        mineru_result,
                        result_key,
                    )
                except Exception as e:
                    logger.warning(f"解析 middle_json 失败: {e}")
            if items and pdf_info:
                items = await asyncio.to_thread(apply_cross_page_table_merge_from_pdf_info, items, pdf_info)
            elif not pdf_info:
                logger.info("未找到 pdf_info，跳过跨页表合并")

            # 跨页合并后：基于最终 rows 构造 headerCells
            if items and header_done:
                try:
                    with sw.task("build_header_cells_from_pending") as t:
                        built = await asyncio.to_thread(build_header_cells_from_pending, items)
                        t.detail(f"built: {built}")
                except Exception as e:
                    logger.warning(f"构造 headerCells 失败: {e}")

            # 跨页表合并后：并行执行表格分类 + 表头检测（互不依赖，可以同时进行）
            if items:
                tasks = []

                # 任务1：表格分类
                if classify_tables:
                    async def _classify_task():
                        try:
                            with sw.task("classify_tables_async") as t:
                                _total, _done = await classify_tables_async(
                                    items,
                                    force_reclassify=force_classify,
                                    max_concurrency=max_concurrency,
                                    locale=locale,
                                    skip_page_cluster=skip_page_cluster,
                                )
                                t.detail(f"total: {_total}, done: {_done}")
                            try:
                                await asyncio.to_thread(build_table_2d_for_items, items)
                            except Exception:
                                pass
                            return ("classify", _total, _done)
                        except Exception as e:
                            logger.warning(f"表格分类失败: {e}")
                            return ("classify", 0, 0)
                    tasks.append(_classify_task())

                # 任务2：表头检测（仅当未在 cell_traceability 并行阶段执行时才运行）
                if not header_done:
                    async def _header_task():
                        try:
                            with sw.task("detect_headers_for_items") as t:
                                total, detected = await adetect_headers_for_items(
                                    items,
                                    model_id="model_for_table_header_aided",
                                    max_concurrency=10,
                                    batch_mode=TABLE_HEADER_BATCH_MODE,
                                )
                                t.detail(f"total: {total}, detected: {detected}")
                            # 非并行分支：LLM 推断完成后立即基于当前 rows 构造 headerCells
                            try:
                                built = await asyncio.to_thread(build_header_cells_from_pending, items)
                                logger.debug(f"build_header_cells_from_pending built={built}")
                            except Exception as be:
                                logger.warning(f"构造 headerCells(非并行分支) 失败: {be}")
                            return ("header", total, detected)
                        except Exception as e:
                            logger.warning(f"表头识别失败: {e}")
                            return ("header", 0, 0)
                    tasks.append(_header_task())

                await asyncio.gather(*tasks)

                
        except Exception as e:
            logger.warning(f"跨页表合并处理失败: {e}")

        sw.print()

        # 收集待上传的 MinerU 图片路径
        img_paths_set: set[str] = set()
        for item in items:
            if isinstance(item, dict):
                img = item.get("img_path")
                if img and isinstance(img, str) and img.strip():
                    img_paths_set.add(img.strip())

        # 预校验所有本地图片，避免部分上传后失败留下孤儿对象
        img_uploads: list[tuple[str, bytes, str]] = []  # (img_path, data, content_type)
        if img_paths_set:
            for img_path in img_paths_set:
                img_filename = Path(img_path).name
                local_img_path = images_dir / img_filename
                try:
                    with open(local_img_path, "rb") as f:
                        img_data = f.read()
                except FileNotFoundError:
                    logger.warning("本地 MinerU 图片不存在，跳过上传 | img_path={}, local={}", img_path, local_img_path)
                    continue
                content_type = mimetypes.guess_type(img_path)[0] or "application/octet-stream"
                img_uploads.append((img_path, img_data, content_type))

        # 上传到 MinIO：先上传 converted.json（作为完整性信号），再上传图片和 mineru.json
        await upload_mineru_items(str(resolved_file_id), items)
        for img_path, img_data, content_type in img_uploads:
            await upload_mineru_image(str(resolved_file_id), img_path, img_data, content_type)
        await upload_mineru_result(str(resolved_file_id), mineru_result)
        image_tmp.cleanup()

        return {
            "items": items,
            "mineru_result": mineru_result,
        }


async def splice_split_results_async(
    original_file_id: str,
    split_file_ids: List[str],
    split_part_infos: List[Dict[str, Any]],
) -> None:
    """等待所有 split 解析完成，合并结果后写入 ``original_file_id`` 位置。

    适用场景：调用方先把原 PDF 拆成 N 份，N 份分别走 ``parse_to_items_and_save_async`` 落盘到
    各自的 ``{split_id}`` 目录，调用本方法把多份产物按拆分顺序合并为一份，写到
    ``{original_file_id}`` 目录下。

    合并策略（与未拆分处理保持完全一致）：
      - ``items``：按 split 顺序拼接，``page`` 字段重写为原 PDF 的绝对页码，
        ``element_id`` 重新从 1 顺序编号，``file_id`` 改为 ``original_file_id``。
        对于 ``type == "table"`` 的元素，``img_path`` 重命名为
        ``{original_file_id}_{new_eid}{ext}``，与未拆分处理时
        ``_rename_table_images_by_img_path_sync`` 的命名规则保持一致；
        其他元素 ``img_path`` 保持原样。
      - ``mineru.json``：``md_content`` 用分隔符拼接；``middle_json.pdf_info``、
        ``content_list``、``model_output`` 解析后合并再序列化。
      - ``images/``：各 split 的图片按 ``item["img_path"]`` 实际相对路径复制到
        ``{original_file_id}/{item["img_path"]}``；table 元素在原 split 下的
        路径不带 ``images/`` 前缀（由 ``_rename_table_images_by_img_path_sync``
        重命名为 ``{split_id}_{eid}{ext}``），其它元素保留 MinerU 原始的
        ``images/...`` 前缀——拷贝时严格按单 split 解析产出的实际 MinIO key
        定位，不做"硬加 images/"的推断。

    注意：图片复制阶段（download / upload 任一）任一失败都会**直接抛出异常**，
    以便调用方回退到"只处理原文件"路径。

    Args:
        original_file_id: 原文件 file_id（写入目标）。
        split_file_ids: split file_id 列表，**按拆分顺序**传入。
        split_part_infos: 每个 split 对应的 part_info 字典，至少需要 ``start_page`` 字段
            （1-based，包含），用于把 split 内相对页码映射回原 PDF 绝对页码。

    Raises:
        ValueError: 入参长度不一致或为空。
        FileNotFoundError: split 缺少 ``converted.json``。
        Exception: 图片下载或上传失败。
    """
    if not split_file_ids:
        raise ValueError("split_file_ids 不能为空")
    if len(split_file_ids) != len(split_part_infos):
        raise ValueError(
            f"split_file_ids 与 split_part_infos 长度不一致: {len(split_file_ids)} vs {len(split_part_infos)}"
        )

    # 调用方（_split_and_parse_async）已通过 asyncio.gather 等待所有 split 解析完成，
    # 每个 split 的最后一步 upload_mineru_items 已写入 converted.json，
    # 故此处无需再次轮询 MinIO；直接加载合并即可。
    logger.info(
        "Split 结果拼接开始 | original_file_id={}, split_file_ids={}, split_count={}",
        original_file_id, split_file_ids, len(split_file_ids),
    )

    # 2. 加载每个 split 的 items 与 mineru_result
    split_items_list: List[List[Dict[str, Any]]] = []
    split_results_list: List[Dict[str, Any]] = []
    for sid in split_file_ids:
        try:
            items = await load_mineru_items(sid)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"split converted.json 缺失 | split_id={sid}") from exc
        try:
            results = await load_mineru_result(sid)
        except FileNotFoundError:
            results = {"results": {}}
        split_items_list.append(items)
        split_results_list.append(results)

    # 3. 拼接 items：调整 page / element_id / file_id / img_path。
    #    为实现与未拆分处理完全一致：type=="table" 的元素 img_path 重命名为
    #    {original_file_id}_{new_eid}{ext}，与 _rename_table_images_by_img_path_sync
    #    的命名规则保持一致；其他元素 img_path 保持原样。
    #    同时构建 (old_obj, new_obj) 映射，供 Step 5 精确复制图片。
    merged_items: List[Dict[str, Any]] = []
    img_path_mapping: List[tuple[str, str]] = []
    next_eid = 1
    for i, items in enumerate(split_items_list):
        part_info = split_part_infos[i] or {}
        local_start_raw = part_info.get("start_page", 1)
        try:
            local_start = int(local_start_raw)
        except Exception:
            local_start = 1
        split_id = split_file_ids[i]

        for item in items:
            if not isinstance(item, dict):
                continue
            old_page = item.get("page")
            if isinstance(old_page, int):
                item["page"] = old_page - 1 + local_start
            item["element_id"] = next_eid
            item["file_id"] = original_file_id

            # split 内 source_page_no 是相对页码，需同步偏移为原 PDF 绝对页码，
            # 与 item["page"] 保持一致，避免下游 _get_end_page 取到错误页码。
            if item.get("type") == "table":
                rows = item.get("rows")
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, list):
                            continue
                        for cell in row:
                            if not isinstance(cell, dict):
                                continue
                            cell_pno = cell.get("source_page_no")
                            if isinstance(cell_pno, int):
                                cell["source_page_no"] = cell_pno - 1 + local_start

            img_path = item.get("img_path")
            if isinstance(img_path, str) and img_path.strip():
                img_path_stripped = img_path.strip()
                ext = Path(img_path_stripped).suffix.lower()

                # 旧对象名：直接用 item 当前的 img_path 作为相对路径，
                # 避免硬编码 "images/" 前缀。
                # 注：table 元素的 img_path 在单 split 解析时已被
                # _rename_table_images_by_img_path_sync 重命名为
                # "{split_id}_{eid}{ext}"（basename），实际位于
                # "{split_id}/" 根目录下，不在 images/ 子目录中；
                # 非 table 元素的 img_path 仍带 "images/" 前缀。
                old_obj_name = f"{split_id}/{img_path_stripped}"

                if item.get("type") == "table":
                    # 与未拆分处理一致：table img_path 重命名为 {file_id}_{eid}{ext}
                    new_img_path = f"{original_file_id}_{next_eid}{ext}"
                    item["img_path"] = new_img_path
                else:
                    # 非 table 元素保持原 img_path 不变
                    new_img_path = img_path_stripped

                # 新对象名与写入 item 的 img_path 字段完全一致，
                # 下游可直接通过 build_mineru_image_object_name(file_id, item["img_path"]) 取图。
                new_obj_name = f"{original_file_id}/{new_img_path}"
                img_path_mapping.append((old_obj_name, new_obj_name))

            next_eid += 1
            merged_items.append(item)

    # 4. 拼接 mineru.json 兼容结构
    merged_md_parts: List[str] = []
    merged_pdf_info: List[Dict[str, Any]] = []
    merged_content_list: List[Any] = []
    merged_model_output: List[Any] = []
    first_middle_template: Optional[Dict[str, Any]] = None

    for i, result in enumerate(split_results_list):
        if not isinstance(result, dict):
            continue
        results_obj = result.get("results", {})
        if not isinstance(results_obj, dict) or not results_obj:
            continue
        first_key = next(iter(results_obj.values()))
        if not isinstance(first_key, dict):
            continue

        md = first_key.get("md_content")
        if isinstance(md, str) and md:
            merged_md_parts.append(md)

        middle_str = first_key.get("middle_json")
        if isinstance(middle_str, str) and middle_str:
            if i == 0:
                try:
                    parsed = json_utils.loads(middle_str)
                    if isinstance(parsed, dict):
                        first_middle_template = parsed
                except Exception:
                    first_middle_template = None
            try:
                parsed = json_utils.loads(middle_str)
                if isinstance(parsed, dict):
                    pdf_info = parsed.get("pdf_info")
                    if isinstance(pdf_info, list):
                        merged_pdf_info.extend(pdf_info)
            except Exception:
                pass

        cl = first_key.get("content_list")
        if cl:
            try:
                cl_data = json_utils.loads(cl) if isinstance(cl, str) else cl
                if isinstance(cl_data, list):
                    merged_content_list.extend(cl_data)
            except Exception:
                pass

        mo = first_key.get("model_output")
        if mo:
            try:
                mo_data = json_utils.loads(mo) if isinstance(mo, str) else mo
                if isinstance(mo_data, list):
                    merged_model_output.extend(mo_data)
            except Exception:
                pass

    new_middle: Dict[str, Any] = dict(first_middle_template) if isinstance(first_middle_template, dict) else {}
    if merged_pdf_info:
        new_middle["pdf_info"] = merged_pdf_info

    new_mineru_result: Dict[str, Any] = {
        "results": {
            original_file_id: {
                "md_content": "\n\n---\n\n".join(merged_md_parts),
            }
        }
    }
    if new_middle:
        new_mineru_result["results"][original_file_id]["middle_json"] = json_utils.dumps(new_middle, ensure_ascii=False)
    if merged_content_list:
        new_mineru_result["results"][original_file_id]["content_list"] = json_utils.dumps(merged_content_list, ensure_ascii=False)
    if merged_model_output:
        new_mineru_result["results"][original_file_id]["model_output"] = json_utils.dumps(merged_model_output, ensure_ascii=False)

    # 5. 将各 split 的图片复制到 original_file_id/images/ 下的最终位置。
    #    复用 Step 3 构建的精确 (old_obj, new_obj) 映射，保证与 items.img_path 完全一致。
    #    任一步失败都直接抛出，调用方据此回退到原文件处理。
    for old_obj_name, new_obj_name in img_path_mapping:
        try:
            data = await storage_service.download(MINERU_OUTPUT_BUCKET, old_obj_name)
        except Exception as e:
            logger.warning("下载 split 图片失败 | obj={}, err={}", old_obj_name, e)
            raise
        rel_path = new_obj_name[len(original_file_id) + 1:]  # "images/xxx.ext"
        content_type = mimetypes.guess_type(rel_path)[0] or "application/octet-stream"
        try:
            await storage_service.upload(
                MINERU_OUTPUT_BUCKET,
                new_obj_name,
                data,
                content_type,
            )
        except Exception as e:
            logger.warning("上传 split 图片失败 | obj={}, err={}", new_obj_name, e)
            raise

    # 6. 上传合并后的结果到 original_file_id 位置
    await upload_mineru_items(original_file_id, merged_items)
    await upload_mineru_result(original_file_id, new_mineru_result)

    logger.info(
        "Split 结果拼接完成 | original_file_id={}, split_file_ids={}, total_items={}",
        original_file_id, split_file_ids, len(merged_items),
    )


__all__ = ["MinerUParsingService", "splice_split_results_async"]
