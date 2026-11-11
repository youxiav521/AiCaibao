# 本地运行环境启动流程
有两种方式初始化本地环境:
1. 本地环境通过`vscode` / `trae` / `cursor`的`运行和调试`功能启动`FastAPI`服务的时候, 每次会自动检查和启动本地服务
2. 手动在根路径下运行
   - `cd docker/local`
   - `./setup_local.sh` 或 windows下执行`./setup_local.ps1`

这里几个文件主要包括:
1. `.env_local`: 会被复制到根路径的`.env`下, 作为项目启动时候的基本配置
2. `llm.json`: 会被复制到 `config/llm.json`下, 作为大模型配置
3. `docker-compose-env.yml`: 启动所需要的中间件依赖

## NebulaGraph

本地环境使用单节点 NebulaGraph 3.8.0：

- 常驻服务：`nebula-metad`、`nebula-storaged`、`nebula-graphd`
- 一次性服务：`nebula-init`，负责注册 Storage、创建 `knowledge_graph` Space 和 Schema，成功后退出
- 宿主机连接地址：`127.0.0.1:9669`
- 持久化数据：Docker named volumes

**NebulaGraph 属于可选组件，归入 `nebula` profile。** `setup_local.sh` 和裸 `docker compose up -d` 都不会启动这几个服务，也不会拉取对应镜像。需要使用时，显式带上 `--profile nebula`：

```bash
docker compose -f docker/local/docker-compose-env.yml --profile nebula up -d
docker compose -f docker/local/docker-compose-env.yml --profile nebula wait nebula-init
```

同理，`stop` / `down` 也要带 `--profile nebula`，否则 NebulaGraph 容器不会被停止。

镜像源使用 `docker.m.daocloud.io`（vesoft 官方镜像的国内加速，manifest 与 Docker Hub 完全一致），避免直连 Docker Hub 的网络问题。

检查集群和 Schema（启动临时 Console 后依次执行，`run` 会自动启用所属 profile，无需额外加 `--profile`）：

```bash
docker compose -f docker/local/docker-compose-env.yml run --rm \
  --entrypoint nebula-console nebula-init \
  -addr nebula-graphd -port 9669 -u root -p nebula
```

```ngql
SHOW HOSTS STORAGE;
SHOW SPACES;
USE knowledge_graph;
SHOW TAGS;
SHOW EDGES;
```

退出 Console 后，临时容器会自动删除。

`nebula-init` 中的 DDL 是幂等的，日常重启不会清空图数据。普通 `docker compose down` 会保留数据卷；`down -v` 会同时删除本 Compose 中 Elasticsearch 和 NebulaGraph 的全部本地数据，不应作为日常停止命令。

若初始化检测到旧版通用 `entity`/`relation` Schema，会拒绝继续执行。应先确认旧 Space 中的顶点和边均可由 MinIO 恢复，再单独重建 `knowledge_graph` Space；初始化脚本不会自动删除数据。


TODO:
1. 添加`docker-compose-env-optional.yml`: 将ocr相关模型本地启动并开发, 避免依赖某些情况下必须依赖服务器的问题


## 脚本说明

### setup_local.sh / setup_local.ps1
两个脚本除了启动 Docker 环境外，还会在**项目根目录**创建 `.coder_venv` 虚拟环境，用于代码执行器。

- 使用 `pyproject.toml` 中的 `[dependency-groups] coder` 分组管理依赖版本（当前包含 pandas、numpy）
- 通过 `uv pip install --python` 安装依赖，版本由 `uv.lock` 锁定确保一致性
- 检查到 `.coder_venv/pyvenv.cfg` 已存在时会跳过创建，避免重复安装

**注意**：`setup_local.ps1` 脚本逻辑与 `.sh` 保持一致，但未在实际 Windows 环境中测试验证。

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
