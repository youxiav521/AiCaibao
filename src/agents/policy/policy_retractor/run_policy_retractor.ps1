param(
    [Parameter(Mandatory = $true)]
    [string]$InputPath,

[string]$ModelId = "model_qwen35_35b",

    [string]$Output = "",

    [switch]$EnableLlm,

    [switch]$AnalyzeOutput,

    [int]$MaxSectionConcurrency = 16,

    [bool]$AutoSectionConcurrency = $true,

    [int]$StableMaxSectionConcurrency = 16,

    [int]$MaxCharsPerSection = 12000,

    [int]$SectionOverlapChunks = 1
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $scriptDir "..\\..\\..\\..")).Path
$pythonExe = Join-Path $repoRoot ".venv\\Scripts\\python.exe"
$agentScript = Join-Path $repoRoot "src\\agents\\policy\\policy_retractor\\agent.py"
$analyzeScript = Join-Path $repoRoot "src\\agents\\policy\\policy_retractor\\analyze_output.py"
$envFile = Join-Path $repoRoot ".env"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

if (-not (Test-Path $agentScript)) {
    throw "Agent script not found: $agentScript"
}

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
            $name = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"').Trim("'")
            if ($name) {
                Set-Item -Path "Env:$name" -Value $value
            }
        }
    }
}

$env:PYTHONPATH = $repoRoot

$argsList = @(
    $agentScript,
    "--input", $InputPath,
    "--model-id", $ModelId,
    "--max-section-concurrency", $MaxSectionConcurrency,
    "--stable-max-section-concurrency", $StableMaxSectionConcurrency,
    "--max-chars-per-section", $MaxCharsPerSection,
    "--section-overlap-chunks", $SectionOverlapChunks
)

if ($Output -and $Output.Trim().Length -gt 0) {
    $argsList += @("--output", $Output)
}

if ($EnableLlm) {
    $argsList += "--enable-llm"
    if (-not $env:LLM_API_BASE -or -not $env:LLM_API_KEY) {
        throw "EnableLlm is set, but LLM_API_BASE or LLM_API_KEY is missing. Check .env or current environment."
    }
}

if ($AutoSectionConcurrency) {
    $argsList += "--auto-section-concurrency"
}
else {
    $argsList += "--no-auto-section-concurrency"
}

Write-Host "Running policy_retractor..."
Write-Host "RepoRoot: $repoRoot"
Write-Host "Input: $InputPath"
Write-Host "ModelId: $ModelId"
Write-Host "EnableLlm: $($EnableLlm.IsPresent)"
Write-Host "MaxSectionConcurrency: $MaxSectionConcurrency"
Write-Host "AutoSectionConcurrency: $AutoSectionConcurrency"
Write-Host "StableMaxSectionConcurrency: $StableMaxSectionConcurrency"
Write-Host "MaxCharsPerSection: $MaxCharsPerSection"
Write-Host "SectionOverlapChunks: $SectionOverlapChunks"
if ($Output) { Write-Host "Output: $Output" }

& $pythonExe @argsList

if ($AnalyzeOutput) {
    if (-not (Test-Path $analyzeScript)) {
        throw "Analyze script not found: $analyzeScript"
    }

    $resolvedOutput = if ($Output -and $Output.Trim().Length -gt 0) {
        $Output
    }
    else {
        ""
    }

    if (-not $resolvedOutput) {
        throw "AnalyzeOutput requires -Output so the generated framework json path is known."
    }

    Write-Host "Running analyze_output..."
    & $pythonExe $analyzeScript "--input" $resolvedOutput
}
