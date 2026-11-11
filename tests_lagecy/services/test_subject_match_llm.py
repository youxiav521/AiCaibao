"""财报科目匹配LLM语义匹配测试

测试覆盖：
1. LLMSemanticMatcher 的 select_best_match 方法（短ID映射、置信度过滤、异常处理）
2. SubjectMatchService 的规则匹配（精确、同义词、加总关系）
3. SubjectMatchService 的 LLM 回退匹配
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any, List

from src.applications.parsing.service.subject_match_service import SubjectMatchService
from src.applications.parsing.schemas.subject_match_schema import (
    SubjectMatchRequest,
    Report,
    Subject,
    Table,
    Cell,
    BBox,
)
from src.applications.parsing.utils.llm_semantic_matcher import LLMSemanticMatcher


class TestLLMSemanticMatcher:
    """LLM语义匹配器测试类（选择题模式 + 短ID映射）"""
    
    @pytest.fixture
    def sample_bbox(self) -> BBox:
        return BBox(b=100.0, coordOrigin="TOPLEFT", l=50.0, r=200.0, t=50.0)
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_llm_matcher_initialization(self):
        """测试：LLM匹配器初始化"""
        with patch('src.applications.parsing.utils.llm_semantic_matcher.ModelFactory') as mock_factory:
            mock_model = MagicMock()
            mock_factory.get_model.return_value = mock_model
            
            matcher = LLMSemanticMatcher(model_id="model_for_table_classify")
            assert matcher.model is not None
            assert matcher.model_id == "model_for_table_classify"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_select_best_match_success(self):
        """测试：select_best_match 成功匹配（使用短ID映射）"""
        with patch('src.applications.parsing.utils.llm_semantic_matcher.ModelFactory') as mock_factory:
            mock_response = MagicMock()
            mock_response.content = '{"id": 1, "confidence": 0.95}'
            
            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=mock_response)
            
            mock_factory.get_model.return_value = mock_model
            
            matcher = LLMSemanticMatcher()
            candidates = [
                {"id": "subj_001", "name": "流动资产合计"},
                {"id": "subj_002", "name": "固定资产"},
            ]
            result = await matcher.select_best_match("流动资产总计", candidates)
            
            assert result == "subj_001"
            mock_model.ainvoke.assert_called_once()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_select_best_match_second_candidate(self):
        """测试：select_best_match 匹配到第二个候选"""
        with patch('src.applications.parsing.utils.llm_semantic_matcher.ModelFactory') as mock_factory:
        mock_response = MagicMock()
        mock_response.content = '{"id": 2, "confidence": 0.88}'
        
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=mock_response)
        
        mock_factory.get_model.return_value = mock_model
            
            matcher = LLMSemanticMatcher()
            candidates = [
                {"id": "subj_001", "name": "流动资产合计"},
                {"id": "subj_002", "name": "货币资金"},
            ]
            result = await matcher.select_best_match("现金及银行存款", candidates)
            
            assert result == "subj_002"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_select_best_match_no_match(self):
        """测试：select_best_match 无匹配"""
        with patch('src.applications.parsing.utils.llm_semantic_matcher.ModelFactory') as mock_factory:
            mock_response = MagicMock()
            mock_response.content = '{"id": null, "confidence": 0.0}'
            
            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=mock_response)
            
            mock_factory.get_model.return_value = mock_model
            
            matcher = LLMSemanticMatcher()
            candidates = [
                {"id": "subj_001", "name": "流动资产合计"},
            ]
            result = await matcher.select_best_match("完全不相关的名称", candidates)
            
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_select_best_match_low_confidence(self):
        """测试：select_best_match 置信度低于阈值（0.6）"""
        with patch('src.applications.parsing.utils.llm_semantic_matcher.ModelFactory') as mock_factory:
            mock_response = MagicMock()
            mock_response.content = '{"id": 1, "confidence": 0.4}'
            
            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=mock_response)
            
            mock_factory.get_model.return_value = mock_model
            
            matcher = LLMSemanticMatcher()
            candidates = [
                {"id": "subj_001", "name": "流动资产合计"},
            ]
            result = await matcher.select_best_match("某个科目", candidates)
            
            assert result is None  # confidence 0.4 < 0.6 threshold
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_select_best_match_empty_candidates(self):
        """测试：select_best_match 空候选列表"""
        with patch('src.applications.parsing.utils.llm_semantic_matcher.ModelFactory') as mock_factory:
            mock_model = AsyncMock()
            mock_factory.get_model.return_value = mock_model
            
            matcher = LLMSemanticMatcher()
            result = await matcher.select_best_match("流动资产合计", [])
            
            assert result is None
            mock_model.ainvoke.assert_not_called()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_select_best_match_json_with_markdown(self):
        """测试：解析包含markdown代码块的JSON响应"""
        with patch('src.applications.parsing.utils.llm_semantic_matcher.ModelFactory') as mock_factory:
            mock_response = MagicMock()
            mock_response.content = '```json\n{"id": 2, "confidence": 0.9}\n```'
            
            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=mock_response)
            
            mock_factory.get_model.return_value = mock_model
            
            matcher = LLMSemanticMatcher()
            candidates = [
                {"id": "subj_001", "name": "流动资产合计"},
                {"id": "subj_002", "name": "货币资金"},
            ]
            result = await matcher.select_best_match("货币资金", candidates)
            
            assert result == "subj_002"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_select_best_match_invalid_json(self):
        """测试：JSON解析错误处理"""
        with patch('src.applications.parsing.utils.llm_semantic_matcher.ModelFactory') as mock_factory:
            mock_response = MagicMock()
            mock_response.content = '这不是有效的JSON'
            
            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=mock_response)
            
            mock_factory.get_model.return_value = mock_model
            
            matcher = LLMSemanticMatcher()
            candidates = [{"id": "subj_001", "name": "流动资产合计"}]
            result = await matcher.select_best_match("流动资产总计", candidates)
            
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_select_best_match_exception_handling(self):
        """测试：LLM调用异常处理"""
        with patch('src.applications.parsing.utils.llm_semantic_matcher.ModelFactory') as mock_factory:
            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(side_effect=Exception("LLM API调用失败"))
            
            mock_factory.get_model.return_value = mock_model
            
            matcher = LLMSemanticMatcher()
            candidates = [{"id": "subj_001", "name": "流动资产合计"}]
            result = await matcher.select_best_match("流动资产总计", candidates)
            
            assert result is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_select_best_match_with_source_tags(self):
        """测试：候选列表中包含同义词和加总关系标签"""
        with patch('src.applications.parsing.utils.llm_semantic_matcher.ModelFactory') as mock_factory:
            mock_response = MagicMock()
            mock_response.content = '{"id": 2, "confidence": 0.88}'
            
            mock_model = AsyncMock()
            mock_model.ainvoke = AsyncMock(return_value=mock_response)
            
            mock_factory.get_model.return_value = mock_model
            
            matcher = LLMSemanticMatcher()
            candidates = [
                {"id": "subj_001", "name": "流动资产合计"},
                {"id": "subj_001", "name": "流动资产总计", "source": "synonym"},
                {"id": "subj_001", "name": "货币资金", "source": "summary"},
            ]
            result = await matcher.select_best_match("流动资产总额", candidates)
            
            assert result == "subj_001"
            # 验证prompt中包含标签
            call_args = mock_model.ainvoke.call_args[0][0]
            assert "[同义词]" in call_args
            assert "[加总关系]" in call_args


class TestSubjectMatchServiceWithLLM:
    """科目匹配服务测试类（规则匹配 + LLM回退）"""
    
    @pytest.fixture
    def service(self):
        """创建服务实例（启用LLM）"""
        return SubjectMatchService(use_llm=True, llm_max_concurrency=10)
    
    @pytest.fixture
    def sample_bbox(self) -> BBox:
        return BBox(b=100.0, coordOrigin="TOPLEFT", l=50.0, r=200.0, t=50.0)
    
    @pytest.fixture
    def sample_subjects(self) -> List[Subject]:
        return [
            Subject(
                subjectId="subj_001",
                subjectName="流动资产合计",
                synonymSubject=["流动资产总计", "合计流动资产"],
                summaryRelation=["货币资金", "交易性金融资产", "应收票据"]
            ),
            Subject(
                subjectId="subj_002",
                subjectName="货币资金",
                synonymSubject=[],
                summaryRelation=[]
            ),
        ]
    
    @pytest.fixture
    def sample_reports(self, sample_subjects) -> List[Report]:
        return [
            Report(
                reportId="report_001",
                reportName="资产负债表",
                reportCaliber=2,
                subjectList=sample_subjects
            ),
        ]
    
    @pytest.fixture
    def mock_llm_matcher(self):
        """创建mock LLM匹配器（注入到 service._llm_matcher 使用）"""
        matcher = MagicMock(spec=LLMSemanticMatcher)
        matcher.select_best_match = AsyncMock(return_value=None)
        return matcher
    
    # =========================================================
    # 规则匹配测试（不需要LLM）
    # =========================================================
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rule_match_exact(self, service, sample_bbox, sample_reports):
        """测试：完全一致的规则匹配"""
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[[
                    Cell(cellId="cell_001", subjectId=None, text="流动资产合计",
                         rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox),
                ]],
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_exact", reportList=sample_reports, tables=tables
        )
        response = await service.match_subjects(request)
        assert response.tables[0].rows[0][0].subjectId == "subj_001"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rule_match_synonym(self, service, sample_bbox, sample_reports):
        """测试：同义词规则匹配"""
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[[
                    Cell(cellId="cell_001", subjectId=None, text="合计流动资产",
                         rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox),
                ]],
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_synonym", reportList=sample_reports, tables=tables
        )
        response = await service.match_subjects(request)
        assert response.tables[0].rows[0][0].subjectId == "subj_001"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rule_match_summary_relation(self, service, sample_bbox):
        """测试：加总关系规则匹配（新增功能）"""
        subjects = [
            Subject(
                subjectId="subj_001",
                subjectName="流动资产合计",
                synonymSubject=[],
                summaryRelation=["货币资金", "交易性金融资产", "应收票据"]
            ),
        ]
        reports = [
            Report(
                reportId="report_001", reportName="资产负债表",
                reportCaliber=2, subjectList=subjects
            ),
        ]
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[
                    [Cell(cellId="cell_001", subjectId=None, text="货币资金",
                          rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox)],
                    [Cell(cellId="cell_002", subjectId=None, text="交易性金融资产",
                          rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox)],
                    [Cell(cellId="cell_003", subjectId=None, text="应收票据",
                          rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox)],
                ],
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_summary", reportList=reports, tables=tables
        )
        response = await service.match_subjects(request)
        
        # 所有加总关系项都应匹配到父科目
        assert response.tables[0].rows[0][0].subjectId == "subj_001"
        assert response.tables[0].rows[1][0].subjectId == "subj_001"
        assert response.tables[0].rows[2][0].subjectId == "subj_001"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rule_match_only_first_column(self, service, sample_bbox, sample_reports):
        """测试：只对第一列（科目名称列）赋值subjectId，不影响数值列"""
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[[
                    Cell(cellId="cell_001", subjectId=None, text="流动资产合计",
                         rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox),
                    Cell(cellId="cell_002", subjectId=None, text="1000000",
                         rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox),
                    Cell(cellId="cell_003", subjectId=None, text="2000000",
                         rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox),
                ]],
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_first_col", reportList=sample_reports, tables=tables
        )
        response = await service.match_subjects(request)
        
        row = response.tables[0].rows[0]
        assert row[0].subjectId == "subj_001"  # 第一列匹配成功
        assert row[1].subjectId is None  # 第二列不应被赋值
        assert row[2].subjectId is None  # 第三列不应被赋值
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_rule_match_priority_over_llm(self, service, sample_bbox, sample_reports):
        """测试：规则匹配优先，规则成功时不调用LLM"""
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[[
                    Cell(cellId="cell_001", subjectId=None, text="流动资产合计",
                         rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox),
                ]],
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_priority", reportList=sample_reports, tables=tables
        )
        response = await service.match_subjects(request)
        assert response.tables[0].rows[0][0].subjectId == "subj_001"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_disabled_service(self, sample_bbox, sample_reports):
        """测试：禁用LLM的服务（只使用规则匹配）"""
        service = SubjectMatchService(use_llm=False)
        
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[
                    [Cell(cellId="cell_001", subjectId=None, text="流动资产合计",
                          rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox)],
                    [Cell(cellId="cell_002", subjectId=None, text="需要LLM匹配的科目",
                          rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox)],
                ],
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_no_llm", reportList=sample_reports, tables=tables
        )
        response = await service.match_subjects(request)
        
        assert response.tables[0].rows[0][0].subjectId == "subj_001"  # 规则匹配成功
        assert response.tables[0].rows[1][0].subjectId is None  # 规则失败且LLM禁用
    
    # =========================================================
    # LLM 回退匹配测试（需要 mock LLM 匹配器）
    # =========================================================
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_fallback_after_rule_failure(self, service, sample_bbox, mock_llm_matcher):
        """测试：规则匹配失败后使用LLM匹配"""
        mock_llm_matcher.select_best_match.return_value = "subj_001"
        service._llm_matcher = mock_llm_matcher
        
        subjects = [
            Subject(
                subjectId="subj_001", subjectName="流动资产合计",
                synonymSubject=[], summaryRelation=[]
            ),
        ]
        reports = [
            Report(
                reportId="report_001", reportName="资产负债表",
                reportCaliber=2, subjectList=subjects
            ),
        ]
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[[
                    Cell(cellId="cell_001", subjectId=None, text="流动资产总额",
                         rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox),
                ]],
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_llm_fallback", reportList=reports, tables=tables
        )
        response = await service.match_subjects(request)
        
        mock_llm_matcher.select_best_match.assert_called_once()
        assert response.tables[0].rows[0][0].subjectId == "subj_001"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_match_failure_handling(self, service, sample_bbox, sample_reports, mock_llm_matcher):
        """测试：LLM匹配失败时的处理（返回None）"""
        mock_llm_matcher.select_best_match.return_value = None
        service._llm_matcher = mock_llm_matcher
        
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[[
                    Cell(cellId="cell_001", subjectId=None, text="完全不相关的科目名称",
                         rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox),
                ]],
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_llm_fail", reportList=sample_reports, tables=tables
        )
        response = await service.match_subjects(request)
        
        mock_llm_matcher.select_best_match.assert_called_once()
        assert response.tables[0].rows[0][0].subjectId is None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_match_exception_handling(self, service, sample_bbox, sample_reports, mock_llm_matcher):
        """测试：LLM调用异常时不影响整体服务（异常被捕获）"""
        mock_llm_matcher.select_best_match.side_effect = Exception("LLM API调用失败")
        service._llm_matcher = mock_llm_matcher
        
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[[
                    Cell(cellId="cell_001", subjectId=None, text="需要LLM匹配的科目",
                         rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox),
                ]],
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_llm_exception", reportList=sample_reports, tables=tables
        )
        
        # 不应该抛出异常，异常在 process_single_text 中被捕获
        response = await service.match_subjects(request)
        assert response.tables[0].rows[0][0].subjectId is None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_llm_match_complex_scenario(self, service, sample_bbox, mock_llm_matcher):
        """测试：复杂场景 - 多个报表、多个科目、混合匹配（规则+LLM）"""
        # LLM 根据候选列表智能返回匹配结果
        async def smart_select(subject_text, candidates, context=None):
            for c in candidates:
                if "货币" in c['name'] and "现金" in subject_text:
                    return c['id']
                if "应收" in c['name'] and "应收" in subject_text:
                    return c['id']
            return None
        
        mock_llm_matcher.select_best_match.side_effect = smart_select
        service._llm_matcher = mock_llm_matcher
        
        balance_sheet_subjects = [
            Subject(subjectId="bs_001", subjectName="流动资产合计",
                    synonymSubject=["流动资产总计"], summaryRelation=[]),
            Subject(subjectId="bs_002", subjectName="货币资金",
                    synonymSubject=[], summaryRelation=[]),
            Subject(subjectId="bs_003", subjectName="应收账款",
                    synonymSubject=[], summaryRelation=[]),
        ]
        income_subjects = [
            Subject(subjectId="is_001", subjectName="营业收入",
                    synonymSubject=[], summaryRelation=[]),
        ]
        
        reports = [
            Report(reportId="report_bs", reportName="资产负债表",
                   reportCaliber=2, subjectList=balance_sheet_subjects),
            Report(reportId="report_is", reportName="利润表",
                   reportCaliber=2, subjectList=income_subjects),
        ]
        tables = [
            Table(
                reportId="report_bs",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[
                    [Cell(cellId="c1", subjectId=None, text="流动资产总计",
                          rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox)],
                    [Cell(cellId="c2", subjectId=None, text="现金及银行存款",
                          rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox)],
                    [Cell(cellId="c3", subjectId=None, text="应收款项",
                          rowSpan=1, colSpan=1, pageAt=1, bBox=sample_bbox)],
                ],
            ),
            Table(
                reportId="report_is",
                reportName="合并利润表",
                pageAt=2,
                bBox=sample_bbox,
                rows=[[
                    Cell(cellId="c4", subjectId=None, text="营业收入",
                         rowSpan=1, colSpan=1, pageAt=2, bBox=sample_bbox),
                ]],
            ),
        ]
        
        request = SubjectMatchRequest(
            fileId="test_complex", reportList=reports, tables=tables
        )
        response = await service.match_subjects(request)
        
        assert len(response.tables) == 2
        # 资产负债表
        assert response.tables[0].rows[0][0].subjectId == "bs_001"  # 同义词规则匹配
        assert response.tables[0].rows[1][0].subjectId == "bs_002"  # LLM匹配
        assert response.tables[0].rows[2][0].subjectId == "bs_003"  # LLM匹配
        # 利润表
        assert response.tables[1].rows[0][0].subjectId == "is_001"  # 精确规则匹配
