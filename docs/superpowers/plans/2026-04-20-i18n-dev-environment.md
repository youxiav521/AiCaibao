# i18n-dev Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full `docker/i18n-dev` deployment environment that mirrors `docker/dev`, can be deployed by Jenkins, and can run alongside `docker/dev` on the same host.

**Architecture:** Keep the existing `docker/dev` deployment structure intact and create a sibling `docker/i18n-dev` directory with the same file set. Change only the environment identity, exposed host ports, and requested `.env_app` values so internal Compose networking stays consistent while host-level resources do not collide.

**Tech Stack:** Docker Compose, Jenkins Pipeline, repository-managed environment files

---

### Task 1: Scaffold the `docker/i18n-dev` deployment directory

**Files:**
- Create: `docker/i18n-dev/docker-compose.yml`
- Create: `docker/i18n-dev/.env_app`
- Create: `docker/i18n-dev/README.md`
- Create: `docker/i18n-dev/Jenkinsfile`
- Create: `docker/i18n-dev/llm.json`
- Source reference: `docker/dev/docker-compose.yml`
- Source reference: `docker/dev/.env_app`
- Source reference: `docker/dev/README.md`
- Source reference: `docker/dev/Jenkinsfile`
- Source reference: `docker/dev/llm.json`

- [ ] **Step 1: Copy the baseline files from `docker/dev`**

```bash
mkdir -p docker/i18n-dev
cp docker/dev/docker-compose.yml docker/i18n-dev/docker-compose.yml
cp docker/dev/.env_app docker/i18n-dev/.env_app
cp docker/dev/README.md docker/i18n-dev/README.md
cp docker/dev/Jenkinsfile docker/i18n-dev/Jenkinsfile
cp docker/dev/llm.json docker/i18n-dev/llm.json
```

- [ ] **Step 2: Update host port mappings in `docker/i18n-dev/docker-compose.yml`**

```yaml
ports:
  - "36300:6300"
```

```yaml
ports:
  - "36479:6379"
```

```yaml
ports:
  - "29200:9200"
```

```yaml
ports:
  - "39100:9000"
  - "39101:9001"
```

- [ ] **Step 3: Update the runtime environment file**

```dotenv
CONTENT_GENERATE_CALLBACK_URL=http://10.192.72.11:22223
TABLE_SEMANTIC_ENRICHMENT_ENABLED=false
TABLE_SEMANTIC_ENRICHMENT_LLM_CAPTION_ENABLED=false
```

- [ ] **Step 4: Update Jenkins deployment identity for `i18n-dev`**

```groovy
DEPLOY_ENV="i18n-dev"
APP_NAME="yusys_ai_i18n_dev"
SOURCE_DIR="docker/i18n-dev"
DEPLOY_DIR = "/data/app/ynsp/yusys_ai_i18n_dev"
```

Also update the SCP source from `docker/dev/* docker/dev/.env_app` to `docker/i18n-dev/* docker/i18n-dev/.env_app`.

- [ ] **Step 5: Update the copied deployment README**

```markdown
将`docker/i18n-dev/`文件夹下所有文件传送到服务器的`/data/app/ynsp/yusys_ai_i18n_dev`
```

### Task 2: Expose the new environment in repository docs

**Files:**
- Modify: `README.MD`

- [ ] **Step 1: Add an `i18n-dev` documentation entry near the existing `docker/dev` link**

```markdown
- [开发环境(i18n): ./docker/i18n-dev/README.md](./docker/i18n-dev/README.md)
```

### Task 3: Verify the new environment wiring

**Files:**
- Verify: `docker/i18n-dev/docker-compose.yml`
- Verify: `docker/i18n-dev/.env_app`
- Verify: `docker/i18n-dev/Jenkinsfile`
- Verify: `docker/i18n-dev/README.md`
- Verify: `README.MD`

- [ ] **Step 1: Check for stale `docker/dev` deployment references inside `docker/i18n-dev`**

Run: `rg -n "docker/dev|yusys_ai_dev|DEPLOY_ENV=\"dev\"|19200:9200|6479:6379|9100:9000|9101:9001|6300:6300" docker/i18n-dev -S`
Expected: only intentional baseline text remains absent from the new directory

- [ ] **Step 2: Render the Compose configuration**

Run: `docker compose -f docker/i18n-dev/docker-compose.yml config`
Expected: exit code `0` and rendered services include the updated host port mappings

- [ ] **Step 3: Inspect repository status**

Run: `git status --short`
Expected: only the new `docker/i18n-dev` files, updated `README.MD`, and plan document are present as intended
