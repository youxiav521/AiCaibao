"""FileParserService 单元测试"""

import pytest
import tempfile
import os
from src.applications.file.app import FileParserService
from src.applications.policy_review.app import FileParseException


class TestFileParserService:
    """FileParserService 测试类"""

    @pytest.fixture
    def file_parser_service(self):
        """FileParserService 实例fixture"""
        return FileParserService()

    @pytest.mark.asyncio
    async def test_parse_file(self, file_parser_service: FileParserService):
        """测试文件解析功能"""
        file_name = "金富科技股份有限公司-尽调报告.docx"
        test_file = Path(
            f"/Users/dave/Downloads/测试数据/{file_name}"
        )

        # 检查文件是否存在
        if not test_file.exists():
            pytest.skip(f"测试文件不存在: {test_file}")

        content = await file_parser_service.parse_file(test_file)
        with open(f"data/loan/{file_name}.md", "w", encoding="utf-8") as f:
            f.write(content)
        assert content is not None, "解析内容不应为空"


if __name__ == "__main__":
    # 运行测试的示例代码
    pytest.main([__file__, "-v"])
