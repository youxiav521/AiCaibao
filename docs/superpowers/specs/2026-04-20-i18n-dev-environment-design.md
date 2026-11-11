# i18n-dev Environment Design

## Background

The repository already contains a full Docker-based deployment environment under `docker/dev/`, including:

- `docker-compose.yml` for the app and supporting services
- `.env_app` for runtime configuration
- `llm.json` for model configuration
- `README.md` for deployment notes
- `Jenkinsfile` for image build and remote deployment

The new requirement is to create an `i18n-dev` environment that can be deployed in the same way as `dev` and can run on the same host at the same time. The new environment must:

- expose Elasticsearch on host port `29200`
- expose the Yusys API on host port `36300`
- add the following environment values to `.env_app`
  - `CONTENT_GENERATE_CALLBACK_URL=http://10.192.72.11:22223`
  - `TABLE_SEMANTIC_ENRICHMENT_ENABLED=false`
  - `TABLE_SEMANTIC_ENRICHMENT_LLM_CAPTION_ENABLED=false`

## Goals

- Add a complete `docker/i18n-dev/` deployment directory modeled after `docker/dev/`
- Preserve the existing `docker/dev/` behavior unchanged
- Ensure `dev` and `i18n-dev` can run concurrently on the same machine without host port conflicts
- Keep the Jenkins deployment flow consistent with the existing `dev` pipeline

## Non-Goals

- Refactoring `docker/dev/` into a shared template system
- Changing container-internal ports or service names used inside Compose networking
- Changing application logic beyond environment configuration

## Chosen Approach

Create a new `docker/i18n-dev/` directory by mirroring `docker/dev/` and updating only the deployment identifiers, host port mappings, and environment settings required for concurrent deployment.

This is the lowest-risk option because it keeps the existing deployment contract intact and minimizes behavioral differences between `dev` and `i18n-dev`. It also avoids introducing a wider refactor to the deployment layout.

## File-Level Design

### `docker/i18n-dev/docker-compose.yml`

Copy the existing `docker/dev/docker-compose.yml` structure and keep:

- service names unchanged inside the Compose network
- internal container ports unchanged
- mounted files and data layout aligned with `docker/dev`
- health checks, restart policy, and startup commands aligned with `docker/dev`

Update only the host-facing configuration needed for coexistence:

- API: `36300:6300`
- Elasticsearch: `29200:9200`
- Redis: move host port away from `6479` to a non-conflicting dedicated port
- MinIO API: move host port away from `9100` to a non-conflicting dedicated port
- MinIO Console: move host port away from `9101` to a non-conflicting dedicated port

Recommended host port set for `i18n-dev`:

- Redis: `36479:6379`
- MinIO API: `39100:9000`
- MinIO Console: `39101:9001`

These ports preserve the existing service exposure pattern while avoiding conflicts with `dev`.

### `docker/i18n-dev/.env_app`

Copy `docker/dev/.env_app` and keep the existing environment defaults unless this environment specifically requires a divergence.

Append or update the following keys:

- `CONTENT_GENERATE_CALLBACK_URL=http://10.192.72.11:22223`
- `TABLE_SEMANTIC_ENRICHMENT_ENABLED=false`
- `TABLE_SEMANTIC_ENRICHMENT_LLM_CAPTION_ENABLED=false`

No additional application code changes are required because the repository already reads these values from environment variables.

### `docker/i18n-dev/Jenkinsfile`

Copy `docker/dev/Jenkinsfile` and update deployment identity values so the new environment deploys independently:

- `DEPLOY_ENV="i18n-dev"`
- `APP_NAME="yusys_ai_i18n_dev"`
- `SOURCE_DIR="docker/i18n-dev"`
- `DEPLOY_DIR="/data/app/ynsp/yusys_ai_i18n_dev"`

Keep the same image build process and remote deployment pattern, but make the SCP source directory point to `docker/i18n-dev`.

This keeps the Jenkins usage pattern familiar while isolating the deployment target path and runtime naming from `dev`.

### `docker/i18n-dev/README.md`

Copy the `docker/dev` deployment instructions and update the referenced source directory and remote deployment directory to the `i18n-dev` equivalents.

### `docker/i18n-dev/llm.json`

Copy from `docker/dev/llm.json` without modification unless the copied file references environment-specific values. If no such references exist, keep it byte-for-byte aligned.

### Repository Root Documentation

Update the root `README.MD` so the new environment is discoverable alongside the existing `dev` environment entry.

## Data and Runtime Isolation

This design intentionally keeps isolation at the deployment directory and host-port layer:

- separate remote deploy directory
- separate Compose project identity via `DEPLOY_ENV`
- separate host ports for externally reachable services

Container-to-container communication remains unchanged because it continues to use Compose service names such as `yusys-elastic-search`, `yusys-redis`, and `yusys-minio`.

## Risks and Mitigations

### Host port conflicts

Risk:
If `i18n-dev` reuses any host ports currently occupied by `dev`, concurrent startup will fail.

Mitigation:
Assign a full non-conflicting host port set for all exposed services, not only API and Elasticsearch.

### Jenkins path drift

Risk:
If the new pipeline still references `docker/dev` or the `yusys_ai_dev` deploy path, deployments will overwrite the wrong environment.

Mitigation:
Update all path, app-name, and deploy-environment identifiers in the copied Jenkinsfile and README.

### Hidden documentation gaps

Risk:
The new environment exists but is hard to discover or use.

Mitigation:
Add a root README entry for `docker/i18n-dev`.

## Verification Plan

After implementation, verify with:

1. A diff review confirming `docker/i18n-dev` mirrors `docker/dev` except for the intended environment-specific changes
2. A text search confirming no stale `docker/dev` or `yusys_ai_dev` references remain inside `docker/i18n-dev`
3. A Compose config render for `docker/i18n-dev/docker-compose.yml`
4. A repository status check confirming only the intended files changed

## Acceptance Criteria

The work is complete when:

- `docker/i18n-dev/` exists with `docker-compose.yml`, `.env_app`, `README.md`, `Jenkinsfile`, and `llm.json`
- the API is exposed on host port `36300`
- Elasticsearch is exposed on host port `29200`
- the new environment can coexist with `docker/dev` on the same machine
- `.env_app` contains the three requested configuration values
- Jenkins can deploy `i18n-dev` into its own remote directory without colliding with `dev`
