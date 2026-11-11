from __future__ import annotations

import tempfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from regression_mvp.storage import minio_sync


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        case_dir = root / "case"
        case_dir.mkdir()
        (case_dir / "case.yaml").write_text('case_name: "case_001"\n', encoding="utf-8")
        (case_dir / "expected.json").write_text('{"a": 1}\n', encoding="utf-8")

        records = minio_sync._scan_local_files(case_dir)
        assert sorted(records) == ["case.yaml", "expected.json"]
        assert records["expected.json"].content_type == "application/json"

        base = minio_sync._manifest_from_records("case", "file_read/case_001", 1, list(records.values()), None)

        (case_dir / "expected.json").write_text('{"a": 2}\n', encoding="utf-8")
        local = minio_sync._manifest_from_records(
            "case",
            "file_read/case_001",
            0,
            list(minio_sync._scan_local_files(case_dir).values()),
            base,
        )

        remote_files = []
        for record in base.files:
            if record.path == "expected.json":
                remote_files.append(
                    minio_sync.FileRecord(
                        path=record.path,
                        size=9,
                        sha256="remote-different",
                        content_type=record.content_type,
                        object_name="remote/object",
                    )
                )
            else:
                remote_files.append(record)
        remote = minio_sync._manifest_from_records("case", "file_read/case_001", 2, remote_files, base)
        report = minio_sync._publish_conflicts("case", "file_read/case_001", base, local, remote)
        assert report.entries
        assert report.entries[0].conflict_type == "both_modified_different"

        try:
            minio_sync._safe_relative_path(Path("../bad"))
        except minio_sync.PathSafetyError:
            pass
        else:
            raise AssertionError("unsafe path was accepted")

    print("minio_sync_check passed")


if __name__ == "__main__":
    main()
