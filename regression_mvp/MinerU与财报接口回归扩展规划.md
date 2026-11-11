# MinerU 与财报接口回归扩展规划

本文档用于约束后续自动化回归测试的接口扩展顺序。当前阶段优先覆盖 MinerU 解析和财报相关接口，暂不扩展 RAG、聊天、政策、知识图谱等链路。

## 1. 当前已覆盖接口

| 接口 | 当前状态 | 说明 |
|---|---|---|
| `POST /api/file/read` | 已覆盖 MVP | 综合链路：上传文件 -> 保存 MinIO -> MinerU 解析 -> 生成 `convert_json` -> 保存解析产物 -> RAG 入库 -> 返回前端展示结构 |

`/api/file/read` 适合作为主冒烟链路，但它包含 RAG 入库等后置动作。后续若失败，需要结合更细粒度的 MinerU 接口判断问题阶段。

## 2. 第一批：MinerU 解析主链路

目标：先把“文件解析 -> converted items 落库 -> 读取解析产物”打牢，隔离 `/api/file/read` 中 RAG 入库等额外影响。

| 优先级 | 接口 | 用途 | 回归重点 |
|---|---|---|---|
| P0 | `POST /api/parsing/mineru/to-json` | 同步上传并调用 MinerU，保存解析产物，不做 RAG 入库 | HTTP 成功、`success=true`、`file_id`、`indexing_status=disabled` |
| P0 | `GET /api/parsing/mineru/result/{file_id}` | 公共结果读取接口，根据 `file_id` 读取 MinIO 中的 `converted.json` | `success=true`、`json` 可反序列化、items 数量和关键字段稳定 |
| P1 | `POST /api/parsing/mineru/upload` | 异步触发 MinerU 解析，立即返回 `file_id` | 返回 `file_id`、后台解析能完成 |
| P1 | `POST /api/parsing/mineru/upload` -> 轮询 `GET /api/parsing/mineru/result/{file_id}` | 异步解析完整链路 | waiter、超时分类、最终解析产物对比 |

推荐第一条新增链路：

```text
POST /api/parsing/mineru/to-json
-> GET /api/parsing/mineru/result/{file_id}
-> 保存 actual_items.json
-> 生成 preview.html / overlay.html
-> compare expected_items.json
```

`GET /api/parsing/mineru/result/{file_id}` 本身不是同步接口，也不是异步接口。它是同步解析链路和异步解析链路共同复用的结果读取接口：

```text
同步解析链路：
POST /api/parsing/mineru/to-json
-> GET /api/parsing/mineru/result/{file_id}

异步解析链路：
POST /api/parsing/mineru/upload
-> wait/轮询 GET /api/parsing/mineru/result/{file_id}
```

## 3. 第二批：财报表格与三大表链路

目标：基于 MinerU 已解析出来的 `file_id`，验证财报表格识别、原始表格拆分、三大表抽取和元数据抽取。

| 优先级 | 接口 | 用途 | 回归重点 |
|---|---|---|---|
| P0 | `GET /api/parsing/extract/org-tables/{file_id}` | 获取所有原始表格块 | 表格数量、`id`、`mergeIds`、`primaryElementId`、bbox、拆分后缀 |
| P0 | `GET /api/parsing/extract/three-statements/{file_id}` | 提取前端结构化财务三大表 | 报表分类、表格数量、行列结构、金额、单位、bbox、`mergeIds` |
| P1 | `GET /api/parsing/extract/financial-metadata/{file_id}?save=false` | 提取财报元数据和三大表工具结果 | 企业名称、证券代码、证券简称、货币类型、审计信息、三大表摘要 |
| P2 | `POST /api/parsing/extract/merge-tables` | 手动合并表格 | 指定 `sourceIds` 后合并结果是否稳定 |

推荐链路：

```text
POST /api/parsing/mineru/to-json
-> GET /api/parsing/mineru/result/{file_id}
-> GET /api/parsing/extract/org-tables/{file_id}
-> GET /api/parsing/extract/three-statements/{file_id}
-> GET /api/parsing/extract/financial-metadata/{file_id}?save=false
```

说明：

- `org-tables` 是三大表失败排查的重要辅助证据。
- `three-statements` 是财报回归的核心业务结果。
- `financial-metadata` 依赖财报字段规则，建议在三大表链路稳定后纳入。
- `merge-tables` 需要人工准备明确的 `sourceIds`，适合作为专项 case。

## 4. 第三批：财报结构化提取接口

目标：覆盖 LLM/VLM 财报结构化提取能力。该类接口依赖模型输出，波动通常比规则解析链路更大，因此放在 MinerU 和三大表稳定之后。

| 优先级 | 接口 | 用途 | 回归重点 |
|---|---|---|---|
| P2 | `POST /api/doc/fin_extract` | 固定财报规则抽取，支持 `llm` / `vlm` | 财报基础信息字段、证据来源、空值处理 |
| P2 | `POST /api/doc/extract` | 通用结构化抽取，输入 `docContent` + `extract` 规则 | 自定义规则抽取结果、`objectSource` |
| P2 | `POST /api/doc/extract_v2` | 通用结构化抽取 V2，支持 `llm` / `vlm` | LLM/VLM 模式差异、错误分类 |

优先建议：

```text
POST /api/doc/fin_extract
```

先只覆盖 `mode=llm`，输入使用前面 MinerU 链路生成的 `docContent`，待稳定后再考虑 `mode=vlm`。

## 5. 第四批：ADE OpenAPI 兼容链路

目标：如果需要覆盖外部系统接入或 OpenAPI 兼容性，再纳入 ADE 接口。否则该批次不抢占 MinerU 和财报核心链路资源。

### 5.1 ADE Parse 前置链路

| 优先级 | 接口 | 用途 |
|---|---|---|
| P2 | `POST /openapi/ade/parse` | ADE 同步解析 |
| P2 | `POST /openapi/ade/upload-file` | ADE 只上传文件 |
| P2 | `POST /openapi/ade/submit-parse-job` | ADE 异步解析任务 |
| P2 | `GET /openapi/ade/parse/jobs/{job_id}` | 查询异步解析结果 |

### 5.2 ADE 财报接口

| 优先级 | 接口 | 用途 |
|---|---|---|
| P2 | `GET /openapi/ade/finacial/metadata/{file_id}` | ADE file_id 版财报元数据 |
| P2 | `GET /openapi/ade/finacial/three-statements/{file_id}` | ADE file_id 版财务三大表 |
| P3 | `GET /openapi/ade/finacial/metadata/by-job/{job_id}` | ADE job_id 版财报元数据 |
| P3 | `GET /openapi/ade/finacial/three-statements/by-job/{job_id}` | ADE job_id 版财务三大表 |

注意：当前项目路由中路径拼写是 `finacial`，不是 `financial`。

## 6. 总体推进顺序

| 阶段 | 接口/链路 | 目标 |
|---|---|---|
| 1 | `POST /api/parsing/mineru/to-json` | 建立纯 MinerU 同步解析回归 |
| 2 | `GET /api/parsing/mineru/result/{file_id}` | 验证 MinIO 解析产物读取 |
| 3 | `GET /api/parsing/extract/org-tables/{file_id}` | 验证财报原始表格识别和拆分 |
| 4 | `GET /api/parsing/extract/three-statements/{file_id}` | 验证财报核心三大表结果 |
| 5 | `GET /api/parsing/extract/financial-metadata/{file_id}?save=false` | 验证财报元数据 |
| 6 | `POST /api/parsing/mineru/upload` + waiter | 覆盖 MinerU 异步解析 |
| 7 | `POST /api/parsing/extract/merge-tables` | 覆盖手动合并表格专项能力 |
| 8 | `POST /api/doc/fin_extract` | 覆盖财报结构化提取 |
| 9 | `POST /api/doc/extract_v2` | 覆盖通用结构化提取 V2 |
| 10 | ADE `/openapi/ade/*` | 覆盖外部 OpenAPI 兼容链路 |

## 7. 测试用例管理要求

新增接口测试时继续沿用候选用例流程：

```text
generate candidate
-> 生成 actual / expected candidate
-> 生成 metrics
-> 生成 preview.html
-> 如果结果包含 bBox/page，生成 overlay.html
-> 人工确认
-> promote 为正式 case
```

正式 case 必须包含：

- `case.yaml`
- 输入文件或上游依赖说明
- expected baseline
- `input_sha256`
- 接口类型或链路类型
- 人工确认状态

对于链路型 case，例如 MinerU -> 财报三大表，建议保存每一步的 actual artifact，避免只看到最终 diff 而无法定位问题阶段。
