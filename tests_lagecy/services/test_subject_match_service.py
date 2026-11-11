"""财报科目匹配服务测试"""

import pytest
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


class TestSubjectMatchService:
    """财报科目匹配服务测试类"""
    
    @pytest.fixture
    def service(self):
        """创建服务实例（测试使用规则匹配，不调用LLM）"""
        return SubjectMatchService(use_llm=False)
    
    @pytest.fixture
    def sample_bbox(self) -> BBox:
        """创建示例边界框"""
        return BBox(
            b=100.0,
            coordOrigin="TOPLEFT",
            l=50.0,
            r=200.0,
            t=50.0
        )
    
    @pytest.fixture
    def sample_subjects(self) -> List[Subject]:
        """创建示例科目列表"""
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
            Subject(
                subjectId="subj_003",
                subjectName="存货",
                synonymSubject=["库存商品", "存货资产"],
                summaryRelation=[]
            ),
            Subject(
                subjectId="subj_004",
                subjectName="持有待售资产",
                synonymSubject=[],
                summaryRelation=[]
            ),
            Subject(
                subjectId="subj_005",
                subjectName="应收账款",
                synonymSubject=["应收款项"],
                summaryRelation=[]
            ),
        ]
    
    @pytest.fixture
    def sample_reports(self, sample_subjects) -> List[Report]:
        """创建示例报表列表"""
        return [
            Report(
                reportId="report_001",
                reportName="资产负债表",
                reportCaliber=2,  # 合并
                subjectList=sample_subjects
            ),
            Report(
                reportId="report_002",
                reportName="利润表",
                reportCaliber=1,  # 单体
                subjectList=[
                    Subject(
                        subjectId="subj_101",
                        subjectName="营业收入",
                        synonymSubject=["主营业务收入"],
                        summaryRelation=[]
                    ),
                    Subject(
                        subjectId="subj_102",
                        subjectName="营业成本",
                        synonymSubject=["主营业务成本"],
                        summaryRelation=[]
                    ),
                ]
            ),
        ]
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_exact_match_by_subject_name(self, service, sample_bbox, sample_reports):
        """测试：通过科目名称完全匹配"""
        # 构造表格数据
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[
                    [
                        Cell(
                            cellId="cell_001",
                            subjectId=None,
                            text="流动资产合计",
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                        Cell(
                            cellId="cell_002",
                            subjectId=None,
                            text="1000000",
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                    ],
                ]
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_file_001",
            reportList=sample_reports,
            tables=tables
        )
        
        response = await service.match_subjects(request)
        
        # 验证匹配结果
        assert response.fileId == "test_file_001"
        assert len(response.tables) == 1
        assert response.tables[0].rows[0][0].subjectId == "subj_001"
        assert response.tables[0].rows[0][1].subjectId == "subj_001"  # 整行都应该填充
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_match_by_synonym(self, service, sample_bbox, sample_reports):
        """测试：通过同义词匹配"""
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[
                    [
                        Cell(
                            cellId="cell_001",
                            subjectId=None,
                            text="流动资产总计",  # 使用同义词
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                        Cell(
                            cellId="cell_002",
                            subjectId=None,
                            text="2000000",
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                    ],
                ]
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_file_002",
            reportList=sample_reports,
            tables=tables
        )
        
        response = await service.match_subjects(request)
        
        # 验证通过同义词匹配成功
        assert response.tables[0].rows[0][0].subjectId == "subj_001"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_match_by_summary_relation(self, service, sample_bbox):
        """测试：通过汇总关系匹配（单个项）"""
        # 创建一个科目，包含汇总关系项，但不包含该汇总关系项作为独立科目
        subjects = [
            Subject(
                subjectId="subj_001",
                subjectName="流动资产合计",
                synonymSubject=[],
                summaryRelation=["货币资金", "交易性金融资产", "应收票据"]  # 货币资金是汇总关系项
            ),
            # 注意：不包含"货币资金"作为独立科目，这样第二步不会匹配，会进入第四步
        ]
        
        reports = [
            Report(
                reportId="report_001",
                reportName="资产负债表",
                reportCaliber=2,
                subjectList=subjects
            ),
        ]
        
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[
                    [
                        Cell(
                            cellId="cell_001",
                            subjectId=None,
                            text="货币资金",  # 汇总关系中的子科目，不在subjectList中
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                        Cell(
                            cellId="cell_002",
                            subjectId=None,
                            text="500000",
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                    ],
                ]
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_file_003",
            reportList=reports,
            tables=tables
        )
        
        response = await service.match_subjects(request)
        
        # 验证通过汇总关系匹配成功（应该匹配到"流动资产合计"）
        assert response.tables[0].rows[0][0].subjectId == "subj_001"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_match_by_multiple_summary_relations(self, service, sample_bbox):
        """测试：通过多个汇总关系项匹配多行"""
        # 创建一个科目，包含多个汇总关系项
        subjects = [
            Subject(
                subjectId="subj_001",
                subjectName="流动资产合计",
                synonymSubject=[],
                summaryRelation=["货币资金", "交易性金融资产", "应收票据"]  # 多个汇总关系项
            ),
        ]
        
        reports = [
            Report(
                reportId="report_001",
                reportName="资产负债表",
                reportCaliber=2,
                subjectList=subjects
            ),
        ]
        
        # 创建表格，包含多个汇总关系项对应的行
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[
                    [
                        Cell(
                            cellId="cell_001",
                            subjectId=None,
                            text="货币资金",  # 第一个汇总关系项
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                    ],
                    [
                        Cell(
                            cellId="cell_002",
                            subjectId=None,
                            text="交易性金融资产",  # 第二个汇总关系项
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                    ],
                    [
                        Cell(
                            cellId="cell_003",
                            subjectId=None,
                            text="应收票据",  # 第三个汇总关系项
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                    ],
                ]
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_file_summary_multi",
            reportList=reports,
            tables=tables
        )
        
        response = await service.match_subjects(request)
        
        # 验证所有汇总关系项都匹配成功，都填充了相同的subjectId
        assert response.tables[0].rows[0][0].subjectId == "subj_001"
        assert response.tables[0].rows[1][0].subjectId == "subj_001"
        assert response.tables[0].rows[2][0].subjectId == "subj_001"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_report_name_matching_with_caliber(self, service, sample_bbox, sample_reports):
        """测试：报表名称匹配（考虑口径）"""
        # 测试合并报表匹配
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[
                    [
                        Cell(
                            cellId="cell_001",
                            subjectId=None,
                            text="流动资产合计",
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                    ],
                ]
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_file_004",
            reportList=sample_reports,
            tables=tables
        )
        
        response = await service.match_subjects(request)
        
        # 验证报表名称匹配成功
        assert response.tables[0].rows[0][0].subjectId == "subj_001"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_no_match_scenario(self, service, sample_bbox, sample_reports):
        """测试：未匹配场景"""
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[
                    [
                        Cell(
                            cellId="cell_001",
                            subjectId=None,
                            text="未知科目名称",  # 不存在的科目
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                    ],
                ]
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_file_005",
            reportList=sample_reports,
            tables=tables
        )
        
        response = await service.match_subjects(request)
        
        # 验证未匹配时 subjectId 仍为 None
        assert response.tables[0].rows[0][0].subjectId is None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_multiple_tables_and_reports(self, service, sample_bbox, sample_reports):
        """测试：多个表格和报表的匹配"""
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[
                    [
                        Cell(
                            cellId="cell_001",
                            subjectId=None,
                            text="流动资产合计",
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                    ],
                ]
            ),
            Table(
                reportId="report_002",
                reportName="单体利润表",
                pageAt=2,
                bBox=sample_bbox,
                rows=[
                    [
                        Cell(
                            cellId="cell_101",
                            subjectId=None,
                            text="营业收入",
                            rowSpan=1,
                            colSpan=1,
                            pageAt=2,
                            bBox=sample_bbox
                        ),
                    ],
                ]
            ),
        ]
        
        request = SubjectMatchRequest(
            fileId="test_file_006",
            reportList=sample_reports,
            tables=tables
        )
        
        response = await service.match_subjects(request)
        
        # 验证两个表格都匹配成功
        assert len(response.tables) == 2
        assert response.tables[0].rows[0][0].subjectId == "subj_001"
        assert response.tables[1].rows[0][0].subjectId == "subj_101"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_report_id_precedence_prevents_caliber_mixing(self, service, sample_bbox):
        """测试：当单体/合并同名时，优先用 reportId 避免混匹。"""
        consolidated_subjects = [
            Subject(
                subjectId="cons_001",
                subjectName="长期股权投资",
                synonymSubject=[],
                summaryRelation=[]
            )
        ]
        parent_subjects = [
            Subject(
                subjectId="parent_001",
                subjectName="长期股权投资",
                synonymSubject=[],
                summaryRelation=[]
            )
        ]

        reports = [
            Report(
                reportId="report_cons_bs",
                reportName="资产负债表",
                reportCaliber=2,
                subjectList=consolidated_subjects,
            ),
            Report(
                reportId="report_parent_bs",
                reportName="资产负债表",
                reportCaliber=1,
                subjectList=parent_subjects,
            ),
        ]

        tables = [
            Table(
                reportId="report_cons_bs",
                reportName="资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[[Cell(
                    cellId="cell_cons_001",
                    subjectId=None,
                    text="长期股权投资",
                    rowSpan=1,
                    colSpan=1,
                    pageAt=1,
                    bBox=sample_bbox,
                )]],
            ),
            Table(
                reportId="report_parent_bs",
                reportName="资产负债表",
                pageAt=2,
                bBox=sample_bbox,
                rows=[[Cell(
                    cellId="cell_parent_001",
                    subjectId=None,
                    text="长期股权投资",
                    rowSpan=1,
                    colSpan=1,
                    pageAt=2,
                    bBox=sample_bbox,
                )]],
            ),
        ]

        request = SubjectMatchRequest(
            fileId="test_file_report_id_precedence",
            reportList=reports,
            tables=tables,
        )

        response = await service.match_subjects(request)

        assert response.tables[0].rows[0][0].subjectId == "cons_001"
        assert response.tables[1].rows[0][0].subjectId == "parent_001"

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_empty_tables(self, service, sample_bbox, sample_reports):
        """测试：空表格场景"""
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[]  # 空行
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_file_007",
            reportList=sample_reports,
            tables=tables
        )
        
        response = await service.match_subjects(request)
        
        # 验证空表格不会报错
        assert len(response.tables) == 1
        assert len(response.tables[0].rows) == 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_partial_match_in_row(self, service, sample_bbox, sample_reports):
        """测试：行中部分单元格已有 subjectId 的情况"""
        tables = [
            Table(
                reportId="report_001",
                reportName="合并资产负债表",
                pageAt=1,
                bBox=sample_bbox,
                rows=[
                    [
                        Cell(
                            cellId="cell_001",
                            subjectId=None,
                            text="流动资产合计",
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                        Cell(
                            cellId="cell_002",
                            subjectId="already_matched",  # 已有 subjectId
                            text="1000000",
                            rowSpan=1,
                            colSpan=1,
                            pageAt=1,
                            bBox=sample_bbox
                        ),
                    ],
                ]
            )
        ]
        
        request = SubjectMatchRequest(
            fileId="test_file_008",
            reportList=sample_reports,
            tables=tables
        )
        
        response = await service.match_subjects(request)
        
        # 验证只填充 subjectId 为 None 的单元格
        assert response.tables[0].rows[0][0].subjectId == "subj_001"
        assert response.tables[0].rows[0][1].subjectId == "already_matched"  # 保持不变
