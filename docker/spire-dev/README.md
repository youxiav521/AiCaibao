## 开发环境部署流程
Jenkins会通过手动执行`Jenkinsfile`对当前应用进行构建和部署.
1. STEP 1 : 使用`docker/Dockerfile`进行镜像构建
2. STEP 2 : 将`docker/dev/`文件夹下所有文件传送到服务器的`/data/app/ynsp/yusys_ai_dev`
3. STEP 3 : 通过`docker-compose`启动`docker-compose.yml`文件, 这个文件中包含了项目所需要的配置文件(`.env`, `llm.json`)和环境`redis`, `minio`, `elastic vector`.

服务依然有一些外部依赖, 包括大模型推理, OCR模型调用. 这些是不同服务之间共享

## 大模型限流配置 (`llm.json`)

对通过src/utils/model_factory.py中方法创建的langchain的BaseModel进行了额外包装，包装后的模型会根据配置进行限流.
限流仅在 provider 级别配置，无全局默认。在 provider 中配置即启用，配置示例：

```json
"providers": [{
  "name": "bailian",
  "concurrency": 10,      // 并发控制，正数启用，-1或不配置关闭
  "rps": 50,              // 每秒请求限制，正数启用，-1或不配置关闭
  "rpm": 2000,             // 每分钟请求限制，正数启用，-1或不配置关闭
  "max_retries": 5,        // 最大重试次数，默认 5
  "retry_delay": 20.0,    // 重试初始延迟（秒），默认 20
  "max_wait": 300.0          // 获取资源最大等待时间（秒），默认 300
}]
```

**约定:** 不配置 或 `-1` = 关闭该限制，正数 = 启用该限制