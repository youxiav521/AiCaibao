"""ADE 财报（Financial）模块。

对外暴露两个基于 file_id 的财报结构化接口：
- GET /openapi/ade/finacial/metadata/{file_id}          提取财报元数据 + 财务三大表
- GET /openapi/ade/finacial/three-statements/{file_id}  仅提取前端结构化的财务三大表
"""
