"""知识图谱类型定义 —— 单一数据源。

所有 ENTITY_TYPE_LIST、ENTITY_TYPE_MAP、ENTITY_TYPES_TABLE、
RELATION_TYPE_LIST、RELATION_TYPE_MAP、RELATION_TYPES_TABLE
均由 SchemaBuilder 从此文件派生。
"""

from dataclasses import dataclass, field


@dataclass
class EntityTypeDef:
    id: int
    name: str
    name_en: str
    attributes: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class RelationTypeDef:
    id: int
    name: str
    name_en: str
    source_types: list[str]
    target_types: list[str]
    attributes: list[str] = field(default_factory=list)
    description: str = ""
    is_bidirectional: bool = False


ENTITY_TYPES: list[EntityTypeDef] = [
    EntityTypeDef(
        id=1, name="企业主体", name_en="ENTERPRISE",
        attributes=[
            "曾用名", "简称", "统一社会信用代码", "企业类型", "生产规模", "经济性质", "股票代码",
            "上市状态", "所属行业", "注册地", "外部信用评级", "成立日期",
            "营业期限", "注册状态", "登记状态", "注册资本", "实收资本", "经营状态",
        ],
        description="企业名称，即依法设立的公司、合伙企业、个人独资企业。",
    ),
    EntityTypeDef(
        id=2, name="自然人", name_en="PERSON",
        attributes=[
            "曾用名", "年龄", "性别", "身份证号", "身份证地址",
            "出生日期", "联系方式", "职位", "任职时间", "信用等级", "婚姻状态",
        ],
        description="人名",
    ),
    EntityTypeDef(
        id=3, name="抵押物", name_en="MORTGAGE_ASSET",
        attributes=[
            "权证编号", "有效期", "权属人", "权属状态", "抵押价值", "法定优先受偿款价值",
        ],
        description="抵押人提供担保的财产",
    ),
    EntityTypeDef(
        id=4, name="质押物", name_en="PLEDGE_ASSET",
        attributes=[
            "权证编号", "有效期", "权属人", "权属状态", "市值", "是否已被重复质押",
        ],
        description="出质给债权人占有的动产财产",
    ),
    EntityTypeDef(
        id=5, name="债权人", name_en="CREDITOR",
        attributes=["类型"],
        description="金融机构或银行机构名称，用于归集企业存量负债、对外授信数据。",
    ),
    EntityTypeDef(
        id=6, name="风险舆情", name_en="RISK_EVENT",
        attributes=[
            "事件类型（涉诉/失信被执行/限高/行政处罚/被列入经营异常等）",
            "案由", "金额", "舆情状态", "主体在案件中的角色",
        ],
        description="信贷负面信息，舆论事件，包括信贷催收、违约、诉讼、失信、处罚等，尽量简写事件。",
    ),
]


RELATION_TYPES: list[RelationTypeDef] = [
    # === 2.1 项目关联关系 ===
    RelationTypeDef(
        id=1, name="主借款人", name_en="IS_BORROWER",
        source_types=["企业主体", "自然人"], target_types=["项目"],
        description="源实体作为本次授信业务申请人，即借款主体，承担还款履约责任",
    ),
    RelationTypeDef(
        id=2, name="保证人", name_en="IS_GUARANTOR",
        source_types=["企业主体", "自然人"], target_types=["项目"],
        description="原文明示源实体是本次授信业务的保证人",
    ),
    RelationTypeDef(
        id=3, name="出质人", name_en="IS_PLEDGOR",
        source_types=["企业主体", "自然人"], target_types=["项目"],
        description="原文明示源实体是本次授信业务的出质人，即源实体为本次授信业务提供质押物。",
    ),
    RelationTypeDef(
        id=4, name="抵押人", name_en="IS_MORTGAGOR",
        source_types=["企业主体", "自然人"], target_types=["项目"],
        description="原文明示源实体是本次授信业务的抵押人，即源实体为本次授信业务提供抵押物。",
    ),
    RelationTypeDef(
        id=5, name="担保人", name_en="IS_WARRANTOR",
        source_types=["企业主体", "自然人"], target_types=["项目"],
        description="原文必须明示源实体为本次授信业务作担保",
    ),
    RelationTypeDef(
        id=6, name="押品担保", name_en="COLLATERAL_FOR",
        source_types=["抵押物", "质押物"], target_types=["项目"],
        attributes=["担保方式", "抵押顺位", "担保金额"],
        description="该押品为本次授信提供财产担保。",
    ),

    # === 2.2 押品关联关系 ===
    RelationTypeDef(
        id=7, name="押品权属", name_en="COLLATERAL_OWNED_BY",
        source_types=["抵押物", "质押物"], target_types=["企业主体", "自然人"],
        description="押品的实际权属人",
    ),

    # === 2.3 自然人之间关系 ===
    RelationTypeDef(
        id=8, name="配偶关系", name_en="SPOUSE_OF",
        source_types=["自然人"], target_types=["自然人"],
        description="双向关系，源实体和目标实体是夫妻关系",
        is_bidirectional=True,
    ),
    RelationTypeDef(
        id=9, name="亲属关系", name_en="KINSHIP_OF",
        source_types=["自然人"], target_types=["自然人"],
        description="双向关系，源实体和目标实体是亲属关系",
        is_bidirectional=True,
    ),

    # === 2.4 自然人 → 企业主体关系 ===
    RelationTypeDef(
        id=10, name="法定代表人", name_en="LEGAL_REPRESENTATIVE",
        source_types=["自然人"], target_types=["企业主体"],
        attributes=["是否历史法人"],
        description="原文明示源实体是目标实体的法定代表人。",
    ),
    RelationTypeDef(
        id=11, name="高管任职", name_en="EXECUTIVE_OF",
        source_types=["自然人"], target_types=["企业主体"],
        attributes=["是否历史任职"],
        description="源实体在目标实体担任董事长、董事、监事、总经理、副总经理、财务负责人、董事会秘书等公司治理或管理职位。",
    ),
    #高管任职可以是：**董事、执行董事、董事长、副董事长、经理、副总经理、职工董事、总经理、负责人、理事、监事、财务负责人。**

    RelationTypeDef(
        id=12, name="实际控制人", name_en="CONTROLLER_OF",
        source_types=["自然人", "企业主体"], target_types=["企业主体"],
        description="原文明示源实体实际控制目标实体。",
    ),
    RelationTypeDef(
        id=13, name="股东", name_en="SHAREHOLDER_OF",
        source_types=["自然人", "企业主体"], target_types=["企业主体"],
        attributes=["是否历史股东", "持股比例", "是否代持"],
        description="源实体投资目标实体，即源实体持有目标实体股权。",
    ),

    # === 2.5 企业主体 → 企业主体关系 ===
    RelationTypeDef(
        id=14, name="供应商关系", name_en="SUPPLIER_OF",
        source_types=["企业主体"], target_types=["企业主体"],
        description="源实体是目标实体的供应商，包括供应场地/服务/商品等",
    ),
    RelationTypeDef(
        id=15, name="对外投资", name_en="INVESTS_IN",
        source_types=["企业主体"], target_types=["企业主体"],
        attributes=["是否历史对外投资"],
        description="源实体公司对目标实体企业的投资，但不持有股权，而是通过买债券、发放股东借款、委托贷款、购买债权的方式投资目标实体企业。",
    ),
    RelationTypeDef(
        id=16, name="分支机构", name_en="BRANCH_OF",
        source_types=["企业主体"], target_types=["企业主体"],
        description="源实体是目标实体的分支机构，比如分公司。",
    ),
    RelationTypeDef(
        id=17, name="借款", name_en="LOAN_TO",
        source_types=["企业主体"], target_types=["企业主体"],
        attributes=["是否历史借款"],
        description="企业间的借款、资金占用关系",
    ),

    # === 2.6 风险与负债关系 ===
    RelationTypeDef(
        id=18, name="风险事件", name_en="RISK_INVOLVES",
        source_types=["企业主体", "自然人"], target_types=["风险舆情"],
        description="原文明示源实体是目标实体的责任主体（当事人/行为主体）",
    ),
    RelationTypeDef(
        id=19, name="存量授信", name_en="CREDIT_FROM",
        source_types=["企业主体", "自然人"], target_types=["债权人"],
        attributes=["余额", "期限", "是否还清", "是否逾期"],
        description="源实体在目标实体的存量借贷情况",
    ),
]
