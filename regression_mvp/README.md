# 自动化回归测试框架说明

`regression_mvp/` 是当前项目内的本地接口回归测试框架，重点覆盖 MinerU 解析、财报结构化、科目匹配等链路。

框架目标是把真实 PDF / JSON 输入沉淀成可重复运行、可对比、可人工复核、可通过 Web UI 交付测试使用。

## 当前已接入接口

| 模块 | 测试入口 | 覆盖链路 | 基线文件 |
| --- | --- | --- | --- |
| 解析给前端展示 | `tests/test_file_read.py` | `POST /api/file/read` | `expected_items.json` |
| MinerU 同步解析 | `tests/test_mineru_to_json.py` | `POST /api/parsing/mineru/to-json` -> `GET /api/parsing/mineru/result/{file_id}` | `expected_items.json` |
| MinerU 异步解析 | `tests/test_mineru_upload_async.py` | `POST /api/parsing/mineru/upload` -> 轮询 `GET /api/parsing/mineru/result/{file_id}` | `expected_items.json` |
| ADE 异步解析 | `tests/test_ade_jobs_async.py` | `POST /openapi/ade/jobs/submit` -> 轮询 `GET /openapi/ade/jobs/{job_id}` -> `GET /openapi/ade/jobs/{job_id}/data` | `expected_items.json` |
| 财报元数据提取 | `tests/test_financial_metadata.py` | MinerU 解析 -> `GET /api/parsing/extract/financial-metadata/{file_id}` | `expected_response.json` |
| 财务三大表提取 | `tests/test_financial_three_statements.py` | MinerU 解析 -> `GET /api/parsing/extract/three-statements/{file_id}` | `expected_response.json` |
| 财报原始表格块 | `tests/test_financial_org_tables.py` | MinerU 解析 -> `GET /api/parsing/extract/org-tables/{file_id}` | `expected_response.json` |
| 财报手动合并表格 | `tests/test_financial_merge_tables.py` | org-tables -> `POST /api/parsing/extract/merge-tables` | `expected_response.json` |
| 财报科目匹配 | `tests/test_financial_subject_match.py` | JSON 直测或 PDF -> MinerU -> org-tables -> 组装请求 -> `POST /api/parsing/subject-match/match` | `expected_response.json` |

说明：

- `GET /api/parsing/mineru/result/{file_id}` 是公共结果读取接口。同步和异步 MinerU 链路都会用它确认保存后的解析结果。
- 财报类接口依赖 MinerU 产出的 `file_id`，所以测试会先完成 MinerU 解析，再调用财报抽取接口。
- 科目匹配支持直接上传 JSON，也支持上传 PDF 后自动走 MinerU 和表格提取链路，便于测试使用。
- PDF 科目匹配会根据抽取表格文本自动选择本地模板：中文使用公司平台模板（3 张报表、311 个科目）；英文使用 IFRS Taxonomy v68.24 模板，并先由大模型分别判断 BS、IS、CF 的列报方式，再只发送对应列报树的科目。完整英文模板包含 2773 个科目，运行时不会把不同列报树混在一起。模板路径分别由 `subject_match_template_file` 和 `subject_match_english_template_file` 配置，列报判断模型和超时由 `subject_match_presentation_model_id`、`subject_match_presentation_timeout_seconds` 配置。模型返回模板外值、缺少决定或调用失败时会终止本次匹配并明确报错，不会退回全量英文模板。如需改回业务接口模板，将 `subject_match_template_source` 设为 `api`。

## 目录结构

```text
regression_mvp/
  config.json                     # 本地测试配置
  run_all_tests.py                # PyCharm 友好的总测试入口
  case_tool.py                    # 命令行候选用例工具
  sync_subject_templates.py       # 同步标准科目模板到本地 fallback 文件
  tests/                          # pytest 接口测试入口
  cases/                          # 正式回归用例
  case_candidates/                # 候选用例，人工确认后再固化
  actual/                         # 最近实际输出
  reports/                        # 测试报告和历史数据
  uploads/                        # Web 上传临时文件
  web_app/                        # 本地 Web 管理页
  utils/
    api_clients/                  # 各接口 HTTP client
    runners/                      # 各接口测试链路编排
    case_loader.py                # 用例加载
    case_manager.py               # 候选用例生成和固化
    comparator.py                 # 规范化、智能匹配、diff 分级
    metric_display.py             # 报告指标中文化展示
    reporter.py                   # Markdown / HTML / JSON 报告生成
    report_store.py               # SQLite 报告索引
```

新增接口通常按这个顺序扩展：

1. 在 `utils/api_clients/` 新增或复用接口 client。
2. 在 `utils/runners/` 新增链路 runner。
3. 在 `tests/` 新增 pytest 入口。
4. 在 `cases/{endpoint_type}/` 固化正式用例。
5. 如果要支持 Web 生成候选用例，在 `case_manager.py` 和 Web 页面里补充 endpoint 类型。

## 启动被测服务

Windows PowerShell 下启动 API：

```powershell
Copy-Item -Force docker/local/.env.local .env
.venv\Scripts\python.exe -m uvicorn src.main:app --port 6300
```

如果链路依赖异步任务，另开窗口启动 Worker：

```powershell
.venv\Scripts\python.exe -m arq src.worker.WorkerSettings
```

打开 Swagger 确认 API 可用：

```text
http://localhost:6300/swagger#/
```

默认测试配置位于 `regression_mvp/config.json`。其中 `trust_env=false` 表示测试脚本自己的 HTTP 请求不读取系统代理环境变量，避免本机代理影响接口测试。

## 运行测试

运行全部接口测试：

```powershell
.venv\Scripts\python.exe -m pytest regression_mvp\tests
```

运行单个接口：

```powershell
.venv\Scripts\python.exe -m pytest regression_mvp\tests\test_mineru_to_json.py
```

PyCharm 中可以直接运行：

```text
regression_mvp/run_all_tests.py
```

`run_all_tests.py` 顶部可配置本次要跑的目标。Web UI 也支持在页面上选择一个或多个接口运行。

## 用例管理

正式用例目录：

```text
regression_mvp/cases/{endpoint_type}/{case_name}/
```

典型结构：

```text
cases/mineru_to_json/case_001/
  input.pdf
  expected_items.json
  case.yaml
```

`case.yaml` 记录用例名、接口类型、说明、启停状态、输入文件、预期文件和输入文件哈希。`input_sha256` 用来防止输入文件被替换后仍然使用旧预期答案。

候选用例目录：

```text
regression_mvp/case_candidates/{run_id}_{case_name}/
```

候选用例用于人工确认。确认后通过 Web 或 `case_tool.py promote` 固化为正式用例。

Web UI 当前支持：

- 批量上传 PDF 生成候选用例。
- 在候选详情中修改用例名称、说明。
- 编辑标准答案 JSON。
- 对已支持的接口进行可视化编辑标准答案。
- `financial_merge_tables` 候选可在详情页调整合并来源，并重新生成合并结果。
- 固化候选用例。
- 在正式用例页启用、停用、逻辑归档删除、恢复、彻底删除、更新答案、编辑名称说明。
- 在正式用例页查看单个用例的历史运行汇总和每次运行产物。
- 在 MinIO 管理页查看本地/远端用例数量、集中同步用例、查看冲突报告。

## MinIO 用例同步

正式用例和候选用例已经接入团队共享 MinIO。默认配置如下：

```text
REGRESSION_MVP_MINIO_BUCKET=regression-mvp-dev
REGRESSION_MVP_MINIO_PREFIX=regression_mvp
REGRESSION_MVP_MINIO_ENABLED=true
```

MinIO 连接地址、账号、密钥复用项目 `.env` 中的 `MINIO__ENDPOINT`、`MINIO__ACCESS_KEY`、`MINIO__SECRET_KEY`、`MINIO__SECURE`。当前开发环境通常是 `10.192.72.11:9100`。

远端对象结构：

```text
regression_mvp/
  cases/{endpoint_type}/{case_name}/
    current.json
    versions/{version}/manifest.json
    versions/{version}/files/...
  candidates/{candidate_dir}/
    current.json
    versions/{version}/manifest.json
    versions/{version}/files/...
```

Web UI 中：
- `上传本地到 MinIO`：把当前本机 `cases/` 和 `case_candidates/` 批量上传到 `regression-mvp-dev`，适合第一次初始化或本地修复后推送。
- `从 MinIO 刷新`：从远端下载正式用例和候选用例到本机，适合其他同事开始使用前同步最新内容。
- `MinIO` 页面：集中查看同步配置、本地/远端对象统计、启动扫描状态、最近同步记录、远端历史版本清理、本地归档清理和冲突报告；冲突详情页支持采用远端、强制采用本地、标记已处理。
- 后续生成候选、编辑候选、固化候选、启停用例、编辑答案、更新答案、改名、逻辑归档删除、恢复和彻底删除会尽力自动同步到 MinIO。

每个远端用例通过 manifest 记录文件清单、大小、SHA-256、Content-Type 和版本。`current.json` 指向当前可见版本，上传会先写入新版本文件和 manifest，最后再更新 `current.json`，避免测试人员读到半成品。

本地会在 `regression_mvp/.sync_state/` 保存上次拉取或发布时的共同基线。如果本地和远端都修改了同一个文件，会在 `regression_mvp/.sync_state/conflicts/` 生成冲突报告，避免静默覆盖。

自动同步失败不会阻断本地操作，失败原因会写入 Web 进程日志，并记录到 MinIO 页的最近同步记录。需要确认远端状态时，可以手动点击 `上传本地到 MinIO` 或 `从 MinIO 刷新`。

Web UI 启动时会自动扫描一次本地和远端状态，但不会自动上传、下载或覆盖文件。扫描结果保存在 `regression_mvp/.sync_state/sync_status.json`，用于提示本地未发布、远端未拉取、冲突风险和缺少基线等情况。

远端历史版本清理会保留 current 指向版本和最近 N 个版本。由于新版本可能复用旧版本对象，清理时只删除没有被保留版本 manifest 引用的旧对象。启动扫描会根据 `REGRESSION_MVP_MINIO_WARN_HISTORY_VERSIONS` 和 `REGRESSION_MVP_MINIO_WARN_HISTORY_MB` 提醒历史缓存是否偏大，但不会自动删除。本地归档清理只作用于 `case_history/`、`case_archive/`、`reports/_archive/` 和已处理冲突目录，不会删除活跃用例或 MinIO 对象。

正式用例的 `归档删除` 是逻辑删除：系统会在 `case.yaml` 中写入 `lifecycle_status: deleted`，并将 `enabled` 置为 `false`。这类用例默认不在用例页展示，也不会被 pytest 收集；在状态筛选中选择 `deleted` 后可以查看，并可执行 `恢复` 或 `彻底删除`。`彻底删除` 会清理 MinIO 远端对象，页面中无法恢复。

## 预期答案更新

正式用例页的“更新答案”会默认复用该用例原有 `input.pdf` 重新调用接口。如果用户选择新文件，则使用新文件重新生成候选答案。

更新流程是：

1. 调用对应接口链路生成新结果。
2. 进入确认页面查看新结果、预览和差异。
3. 用户确认后覆盖该用例的预期答案文件。

该流程避免自动把一次不稳定结果直接写成标准答案。

## 对比规则

当前 diff 不是简单 JSON 全等比较，已经加入以下规则：

- 忽略动态字段，例如 `file_id`、`task_id`、临时 URL、路径、时间戳等。
- 规范化文本空白、不可见字符和常见 Unicode 差异。
- `bbox` 坐标允许小范围误差，默认容忍 4px。
- `score`、`confidence` 等浮点字段允许小精度误差。
- 对解析 item 做智能匹配，优先按稳定 key、页面、类型、坐标和文本相似度匹配，减少数组错位导致的大面积误报。
- 对同页相邻的文本 item 支持有限的 1 对多、多对 1 匹配，识别为拆分或合并时仅作为结构性提示。
- 对表格行做匹配，尽量把“新增/缺失行”和“真实字段变化”区分开。
- diff 样本带严重程度：`blocker`、`critical`、`major`、`minor`、`info`。

对 MinerU 同步/异步解析接口，结论额外分为三层：契约失败（直接失败）、内容疑似回归（直接失败，需要优先人工复核）和结构性抖动（warning，不阻断）。内容层比较每页文本覆盖率、表格内容覆盖率、数字覆盖率与新增文本比例；原始 item diff 只作为高级排查材料。其他接口仍按原有基线 diff 规则判定。

MinerU 的正常抖动阈值不建议凭感觉放宽。可在 `mineru_stability_profile.py` 顶部填写 `CASE_NAME`，连续运行同一个用例 3 次；工具会在 `reports/stability_profiles/` 输出每次结果、观察到的抖动范围和建议阈值，不会自动改写 `config.json`。

## 指标和报告

每次运行会生成报告目录：

```text
regression_mvp/reports/{endpoint_type}/runs/{run_id}/
```

主要报告：

- `report.html`：Web UI 默认展示的可读报告。
- `report.md`：原始 Markdown 报告，便于本地查看或归档。
- `result.json`：结构化运行结果，供 Web UI 和后续统计使用。

Web UI 普通入口只展示 HTML 报告，Markdown 和 JSON 仍保留为原始产物。

HTML 报告包含：

- Case Overview：用例结果、产物链接、预览链接。
- 关键指标：按接口类型中文化展示，例如页数、元素数、表格数、匹配率、三大表数量等。
- 完整指标明细：折叠展示原始 metrics。
- 差异摘要：按严重程度、差异类型、高频路径汇总。
- MinerU Assessment：优先展示契约、内容覆盖率、数字覆盖率、结构性抖动和重点复核页。
- Advanced Raw Diff Samples：折叠展示按严重程度排序的字段级差异。
- 解析预览、坐标预览、diff overlay、表格预览等可视化产物。

报告页支持删除无意义报告。删除不是硬删除，而是移动到：

```text
regression_mvp/reports/_archive/
```

删除后 Web 报告索引会刷新，避免页面继续展示已不存在的报告链接。

## Web 管理页

启动 Web：

```powershell
.venv\Scripts\python.exe -m uvicorn regression_mvp.web_app.app:app --host 127.0.0.1 --port 6400
```

或者在 PyCharm 中运行：

```text
regression_mvp/web_app/run_web.py
```

访问：

```text
http://127.0.0.1:6400
```

Web 使用说明见：

```text
regression_mvp/WEB_USAGE.md
```

## Git 同步建议

适合提交：

- `tests/`
- `utils/`
- `web_app/`
- `config.json`
- 文档和脚本

不适合提交：

- `cases/`
- `actual/`
- `reports/`
- `case_candidates/`
- `case_history/`
- `case_archive/`
- `uploads/`
- `.sync_state/`
- SQLite 数据库
- 日志、缓存、`__pycache__`
- 大体积或敏感 PDF

用例文件现在由团队共享 MinIO 管理，不再通过 Git 同步。新同事拉取代码后，先启动 Web UI，再进入 `MinIO` 页面点击 `从 MinIO 刷新全部`，或在 `用例` 页面点击 `从 MinIO 刷新`，即可把正式用例和候选用例拉到本地工作区。

如果历史上已经把 `cases/` 或 `case_history/` 提交进 Git，需要执行一次：

```powershell
git rm -r --cached regression_mvp/cases regression_mvp/case_history
```

该命令只取消 Git 跟踪，不删除本地文件。提交这次变更后，后续新增或修改的用例文件会被 `.gitignore` 忽略。

## 当前边界

- 这是本地优先的回归测试框架，尚未接入 CI。
- Web UI 暂无权限系统，只适合本地或内网可信环境试用。
- 结构性断言目前未作为强规则加入，优先以基线 diff 和可读报告定位回归。
- 字段忽略和比较策略暂由代码配置维护，暂不开放给普通用户在 Web 页面自由配置，避免误配导致测试失效。
