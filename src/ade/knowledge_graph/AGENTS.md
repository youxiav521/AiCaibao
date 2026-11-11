## 代码索引

```
router/                    # API入口层
|  kg_router.py             # 5个REST接口：任务创建/列表/详情、图谱获取、问答
|
service/                   # 业务逻辑编排层
|  knowledge_graph_service.py  # 核心编排：切片->LLM提取->去重->合并->向量化->保存
|  query_service.py            # 图谱查询：向量/关键词检索->LLM生成回答
|  extractor_service.py        # LLM结果解析，实体/关系与文件元素的匹配
|  splitting_service.py        # 文档切片，按10万字符上限切分Piece
|  embed_service.py            # 向量化服务，调用Embedding API并Redis缓存去重
|  graph_storage_service.py    # MinIO持久化：图谱保存/读取/删除/更新
|
client/                    # LLM调用层
|  kg_client.py                # 调用大模型从文本提取实体和关系
|  prompt.py                   # 提示词模板、分隔符常量、类型映射表
|
schema/                    # 数据模型层
|  kg_schema.py                # 全部Pydantic模型：实体、关系、任务、请求/响应
|
tasks/                     # 异步任务层
|  kg_tasks.py                 # ARQ后台任务，图谱生成支持3次断点重试
|
utils/                     # 工具层
   preprocess_file.py          # converted.json -> FileInput格式预处理
```

> 模型搜索代码时优先查看目录。

