# 配置文件说明

## llm.json

大模型配置文件，配置模型的提供商、模型等信息，用于管理不同提供商的模型。

### 配置结构

```json
{
    // 模型提供商列表，可配置多个
    "providers": [
        {
            // 提供商名称标识
            "name": "provider_name",
            // API 地址
            "api_base": "http://api-endpoint/v1",
            // API 密钥
            "api_key": "your_api_key",
            // 额外请求参数，可用于控制模型行为
            "extra_body": {
                "enable_thinking": false
            }
        }
    ],
    // 默认使用的提供商
    "default_provider": "provider_name",
    // 默认使用的模型
    "default_model": "model_name",
    // 各功能模块的自定义模型配置
    "custom_models": {
        "function_name": {
            "provider": "provider_name",
            "model": "model_name"
        }
    }
}
```

### 字段说明

| 字段 | 说明 | 示例 |
|------|------|------|
| `providers` | 模型提供商列表，可配置多个 | - |
| `providers[].name` | 提供商名称标识 | `bailian` |
| `providers[].api_base` | API 地址 | `http://172.16.40.16:20001/compatible-mode/v1` |
| `providers[].api_key` | API 密钥 | `sk-xxx` |
| `providers[].extra_body` | 额外请求参数，可用于控制模型行为 | `{"enable_thinking": false}` |
| `default_provider` | 默认使用的提供商 | `bailian` |
| `default_model` | 默认使用的模型 | `qwen3.5-35b-a3b` |
| `custom_models` | 各功能模块的自定义模型配置 | - |

### 示例

#### 单提供商配置
```json
{
    "providers": [
        {
            "name": "bailian",
            "api_base": "http://api.base.com/compatible-mode/v1",
            "api_key": "api_key",
            "extra_body": {
                "enable_thinking": false
            }
        }
    ],
    "default_provider": "bailian",
    "default_model": "qwen3.5-35b-a3b",
    "custom_models": {
        //报告问题回答使用模型配置
        "content_generation_agent": "qwen3.5-35b-a3b",
        //问题回答代码生成使用模型配置
        "data_analyst_agent": "qwen3.6-35b-a3b"
    }
}
```

#### 多提供商配置
```json
{
    "providers": [
        {
            "name": "bailian",
            "api_base": "http://api.base.com/compatible-mode/v1",
            "api_key": "api_key"
        },
        {
            "name": "openai",
            "api_base": "https://api.openai.com/v1",
            "api_key": "sk-xxx"
        }
    ],
    "default_provider": "bailian",
    "default_model": "qwen3.5-35b-a3b",
    "custom_models": {
        //报告问题回答使用模型配置
        "content_generation": {
            "provider": "openai",
            "model": "gpt-4o"
        },
        //问题回答代码生成使用模型配置
        "data_analyst_agent": {
            "provider": "bailian",
            "model": "qwen3.6-35b-a3b"
        }
    }
}
```
