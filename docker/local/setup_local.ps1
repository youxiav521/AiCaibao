# setup_local.ps1 - Windows local environment bootstrap script
# 1. Copy .env.local to project .env
# 2. Start docker-compose-env.yml

# Configurable paths
$MinioDrive = "D:"                          # MinIO drive letter
$MinioRootPath = Join-Path $MinioDrive "minio"  # MinIO root directory
$MinioDataPath = Join-Path $MinioRootPath "data" # MinIO data directory
$MinioExePath = Join-Path $MinioRootPath "minio.exe" # MinIO executable path
$MinioLogPath = Join-Path $MinioRootPath "minio.log" # MinIO stdout log
$MinioErrorLogPath = Join-Path $MinioRootPath "minio.error.log" # MinIO stderr log

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptDir) {
    $ScriptDir = $PWD.Path
}

Write-Host "[setup_local] Start local environment setup..."

# 1. Copy environment file
Write-Host "[setup_local] Copy docker/local/.env.local -> .env"
if (Test-Path "$ScriptDir\.env.local") {
    Copy-Item -Force "$ScriptDir\.env.local" "$ScriptDir\..\..\.env"
    Write-Host "[setup_local] .env updated"
} else {
    Write-Host "[setup_local] Warning: .env.local not found, skip"
}

# 2. Copy llm.json config
Write-Host "[setup_local] Copy docker/local/llm.json -> config/llm.json"
if (Test-Path "$ScriptDir\llm.json") {
    if (-not (Test-Path "$ScriptDir\..\..\config")) {
        New-Item -ItemType Directory -Path "$ScriptDir\..\..\config" -Force | Out-Null
    }
    Copy-Item -Force "$ScriptDir\llm.json" "$ScriptDir\..\..\config\llm.json"
    Write-Host "[setup_local] config/llm.json updated"
} else {
    Write-Host "[setup_local] Warning: docker/local/llm.json not found, skip"
}

# 3. Install MinIO
if (Test-Path $MinioExePath) {
    Write-Host "[setup_local] MinIO already installed, skip install"
} else {
    Write-Host "[setup_local] Installing MinIO..."
    # Create MinIO root directory
    New-Item -ItemType Directory -Path $MinioRootPath -Force | Out-Null
    # Download MinIO executable
    Invoke-WebRequest -Uri "https://dl.min.io/server/minio/release/windows-amd64/minio.exe" -OutFile $MinioExePath
    # Set MinIO credentials
    setx MINIO_ROOT_USER minioadmin
    setx MINIO_ROOT_PASSWORD minioadmin
    # Create data directory
    New-Item -ItemType Directory -Path $MinioDataPath -Force | Out-Null
    Write-Host "[setup_local] MinIO install completed"
}

# 4. Start MinIO
$minioRunning = $false
try {
    $listener = [System.Net.Sockets.TcpListener]9000
    $listener.Start()
    $listener.Stop()
} catch {
    $minioRunning = $true
}

if ($minioRunning) {
    Write-Host "[setup_local] MinIO already running, skip start"
} else {
    Write-Host "[setup_local] Starting MinIO..."
    # Start MinIO service
    Start-Process -FilePath $MinioExePath -ArgumentList "server", $MinioDataPath -WindowStyle Hidden -RedirectStandardOutput $MinioLogPath -RedirectStandardError $MinioErrorLogPath

    # Check startup status
    Write-Host "[setup_local] Checking MinIO status..."
    Start-Sleep -Seconds 3
    $minioStarted = $false
    try {
        $listener = [System.Net.Sockets.TcpListener]9000
        $listener.Start()
        $listener.Stop()
    } catch {
        $minioStarted = $true
    }

    if ($minioStarted) {
        Write-Host "[setup_local] MinIO started"
    } else {
        Write-Host "[setup_local] Warning: MinIO may have failed to start, check $MinioLogPath"
    }
}

# 5. Start Docker
Write-Host "[setup_local] Starting Docker environment..."
docker compose -f "$ScriptDir\docker-compose-env.yml" up -d
Write-Host "[setup_local] Docker environment started"

# ========== Build local .coder_venv (project root) ==========
Write-Host "[setup_local] Building .coder_venv in project root..."

# Resolve .coder_venv path
$coderVenvDir = Join-Path $ScriptDir "..\..\.coder_venv"
$coderVenvDir = [System.IO.Path]::GetFullPath($coderVenvDir)
$projectRoot = Join-Path $ScriptDir "..\.."
$projectRoot = [System.IO.Path]::GetFullPath($projectRoot)

# Check venv existence by pyvenv.cfg
$pyvenvCfg = Join-Path $coderVenvDir "pyvenv.cfg"
if (Test-Path $pyvenvCfg) {
    Write-Host "[setup_local] .coder_venv already exists, skip create"
} else {
    Write-Host "[setup_local] Creating .coder_venv..."
    python -m venv --copies $coderVenvDir
    Write-Host "[setup_local] .coder_venv created"
}

# Resolve python path in .coder_venv
$coderPython = Join-Path $coderVenvDir "Scripts\python.exe"

# Install coder dependency group from root pyproject.toml
Write-Host "[setup_local] Installing coder dependency group into .coder_venv..."
if (-not (Test-Path $coderPython)) {
    throw "[setup_local] Error: python executable not found in .coder_venv: $coderPython"
}

Push-Location $projectRoot
try {
    uv pip install --python $coderPython --group coder -i https://pypi.tuna.tsinghua.edu.cn/simple
} finally {
    Pop-Location
}
Write-Host "[setup_local] .coder_venv dependency install completed"

Write-Host "[setup_local] Local environment setup completed"
