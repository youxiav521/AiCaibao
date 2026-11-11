# Policy Retractor Java 对接说明

## 接口概览

政策提取任务提交接口：

- `POST /api/policy/rule-extract/jobs`

Java 端回调接收接口：

- `/api/policy/doc/modelCallback`

说明：

- Java 端提交任务后，服务端异步执行提取。
- 服务端会通过回调接口，把开始、增量 requirement、完成、异常等事件推送给 Java 端。

## 推荐调用方式

推荐参数：

- `enable_llm=true`
- `model_id=model_qwen35_35b`
- `max_section_concurrency=8`

当前默认模型已经是：

- `model_qwen35_35b`

## 提交接口

### 请求地址

```text
POST /api/policy/rule-extract/jobs
Content-Type: application/json
```

### 请求参数

支持两种方式：

1. 直接传 `docContent`
2. 传 `input_path`

推荐使用第 1 种。

### 方式一：直接传 docContent

```json
{
  "fileId": "2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8",
  "callbackUrl": "http://your-java-host/api/policy/doc/modelCallback",
  "docContent": [
    {
      "page": 1,
      "type": "text",
      "value": "xxx",
      "md": "xxx",
      "file_id": "2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8",
      "element_id": 1
    }
  ],
  "enable_llm": true,
  "model_id": "model_qwen35_35b",
  "max_section_concurrency": 8
}
```

### 方式二：传 input_path

```json
{
  "input_path": "storage/mineru_output/converted_output/xxx_mineru_converted.json",
  "callbackUrl": "http://your-java-host/api/policy/doc/modelCallback",
  "enable_llm": true,
  "model_id": "model_qwen35_35b",
  "max_section_concurrency": 8
}
```

### 字段说明

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `fileId` | string | 否 | 文件 ID |
| `docContent` | array | 否 | Java 直接传入的解析内容，推荐 |
| `input_path` | string | 否 | 本地文件路径，与 `docContent` 二选一 |
| `callbackUrl` | string | 否 | 回调完整地址，推荐显式传入 |
| `enable_llm` | boolean | 否 | 是否启用 LLM，默认 `true` |
| `model_id` | string | 否 | 模型 ID，默认 `model_qwen35_35b` |
| `max_section_concurrency` | int | 否 | section 并发数，建议 `8` |

说明：

- `docContent` 和 `input_path` 至少传一个。
- 若同时传，优先使用 `docContent`。

## 提交接口返回

提交成功后，接口立即返回 `start` 事件：

```json
{
  "jobId": "0d9b7f12-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
  "data": {
    "frameworkId": "FW-2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8",
    "fileId": "2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8",
    "event": "start",
    "data": {}
  }
}
```

## 回调接口

### 回调地址

Java 端回调接口：

```text
/api/policy/doc/modelCallback
```

说明：

- 如果请求中传了 `callbackUrl`，服务端直接回调这个完整地址。
- 如果没有传 `callbackUrl`，服务端会按默认配置拼接到 `/api/policy/doc/modelCallback`。

## 回调事件类型

服务端会回调以下 4 类事件：

| event | 含义 | 说明 |
| --- | --- | --- |
| `start` | 开始 | 任务已开始执行 |
| `message` | 处理中 | 每产出 1 条 requirement，就回调 1 次 |
| `complete` | 完成 | 整个文件处理结束，不会再有新的 requirement |
| `error` | 异常 | 整个任务失败 |

Java 端建议处理方式：

- `start`：更新任务开始状态
- `message`：按 requirement 增量落库
- `complete`：更新任务完成状态
- `error`：更新任务失败状态并记录错误

## 回调返回结构

统一返回结构如下：

```json
{
  "data": {
    "frameworkId": "String",
    "fileId": "String",
    "event": "message|complete|error|start",
    "data": {
      "frameworkId": "string",
      "fileId": "string",
      "name": "string",
      "status": "PENDING_REVIEW|ERROR|APPROVED|REJECTED",
      "description": "string",
      "requirements": [
        {
          "requirementId": "string",
          "frameworkId": "string",
          "title": "string",
          "description": "string",
          "sourceText": "string",
          "status": "PENDING_REVIEW|ERROR|APPROVED|REJECTED",
          "sourceLocation": {
            "fileId": "string",
            "elementIds": "string",
            "pageNo": 1
          },
          "controls": [
            {
              "controlId": "string",
              "requirementId": "string",
              "status": "PENDING_REVIEW|ERROR|APPROVED|REJECTED",
              "content": "string",
              "controlType": "qualification_check|document_check|quantitative_check|fund_use_check|collateral_check|prohibited_item_check|monitoring_check",
              "risk": "HIGH|MEDIUM|LOW",
              "sourceText": "string",
              "logicDescription": "string",
              "sourceLocation": {
                "fileId": "string",
                "elementIds": "string",
                "pageNo": 1
              }
            }
          ]
        }
      ]
    }
  }
}
```

## requirement 级回调规则

### 1. `message` 事件

`message` 事件用于增量回调。

特点：

- 每产出 1 条 requirement，回调 1 次
- `data.data.requirements` 中只会有 1 条 requirement
- requirement 下仍然包含其对应的全部 controls

示例：

```json
{
  "data": {
    "frameworkId": "FW-2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8",
    "fileId": "2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8",
    "event": "message",
    "data": {
      "frameworkId": "FW-2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8",
      "fileId": "2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8",
      "name": "",
      "status": "PENDING_REVIEW",
      "description": "",
      "requirements": [
        {
          "requirementId": "REQ-f68334ff710a4a7c9c1a91a57f9918cf",
          "frameworkId": "FW-2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8",
          "title": "示例 requirement",
          "description": "示例描述",
          "sourceText": "原文",
          "status": "PENDING_REVIEW",
          "sourceLocation": {
            "fileId": "2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8",
            "elementIds": "12,13",
            "pageNo": 3
          },
          "controls": [
            {
              "controlId": "RULE-3f0ce25cbbe048d7aa0d9de7e5e78ad1",
              "requirementId": "REQ-f68334ff710a4a7c9c1a91a57f9918cf",
              "status": "PENDING_REVIEW",
              "content": "示例 control",
              "controlType": "document_check",
              "risk": "MEDIUM",
              "sourceText": "原文",
              "logicDescription": "逻辑说明",
              "sourceLocation": {
                "fileId": "2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8",
                "elementIds": "12",
                "pageNo": 3
              }
            }
          ]
        }
      ]
    }
  }
}
```

### 2. `complete` 事件

`complete` 事件表示整份文件已经处理完成。

特点：

- 不会再有新的 requirement 回调
- `data.data.requirements` 中包含整份文件的全部 requirements

Java 端可用 `complete` 事件作为整份文件处理完成标志。

### 3. `error` 事件

任务失败时返回：

```json
{
  "data": {
    "frameworkId": "FW-xxx",
    "fileId": "xxx",
    "event": "error",
    "data": {
      "message": "错误信息"
    }
  }
}
```

## ID 规则

### frameworkId

格式：

```text
FW-<fileId>
```

示例：

```text
FW-2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8
```

### requirementId

格式：

```text
REQ-<32位uuidhex>
```

示例：

```text
REQ-f68334ff710a4a7c9c1a91a57f9918cf
```

### controlId

格式：

```text
RULE-<32位uuidhex>
```

示例：

```text
RULE-3f0ce25cbbe048d7aa0d9de7e5e78ad1
```

## sourceLocation 字段说明

### fileId

- 来源文件 ID

### elementIds

- 字符串类型
- 多个元素 ID 用英文逗号拼接

### pageNo

- number/null 类型
- 表示溯源命中的页码（1-based 页码）
- 若无法确定页码，返回 `null`

示例：

```json
{
  "fileId": "2b0747435e7c4656a3a706f689e0a387gZTyfUcwq8",
  "elementIds": "12,13,14",
  "pageNo": 5
}
```

说明：

- 返回给 Java 的 `elementIds` 是字符串，不是数组。
- Java 端如需拆分，可按 `,` 分割。

## Java 端落库建议

建议唯一键：

- framework：`frameworkId`
- requirement：`requirementId`
- control：`controlId`

建议处理流程：

1. 收到 `start`
   - 记录任务开始状态

2. 收到 `message`
   - upsert framework 基础信息
   - upsert 当前 requirement
   - upsert 当前 requirement 下的 controls

3. 收到 `complete`
   - 更新任务完成状态
   - 可做最终核对

4. 收到 `error`
   - 更新任务失败状态
   - 保存错误信息

## 顺序说明

由于 section 是并发处理：

- `message` 事件到达顺序不一定严格等于文档自然顺序
- 但每一条 requirement 都会单独回调
- 最终一定会收到一个 `complete` 事件作为结束标志

因此 Java 端不要依赖回调顺序，而应依赖：

- `requirementId`
- `controlId`
- `event=complete`

## 本地实测建议

针对本地文件测试结果，推荐：

- 模型：`model_qwen35_35b`
- section 并发：`8`
