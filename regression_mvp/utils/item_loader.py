from __future__ import annotations

from typing import Any

from regression_mvp.utils.api_clients.file_read_client import FileReadApiClient, UploadResult


def resolve_actual_items(client: FileReadApiClient, upload_result: UploadResult) -> list[Any]:
    if upload_result.items is not None:
        return upload_result.items
    if not upload_result.file_id:
        raise ValueError("upload response did not include convert_json or file_id")
    return client.wait_result_items(upload_result.file_id)
