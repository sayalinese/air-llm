<#
    run.ps1 -- Chain LSTM (official Flight_chain.py protocol) full pipeline.

    Pipeline: 0_ build chain tensors -> 2_ XGB baseline (15 static feats, flattened) -> 3_ chain LSTM.
    Chain = flights sharing (OP_CARRIER, OP_CARRIER_FL_NUM, FL_DATE), sorted by scheduled departure,
    padded/truncated to max_len. Only 15 static features; actual delays are labels only (no leakage).
    Unidirectional RNN: position t uses only flights 1..t.

    Usage:
        .\run.ps1                         # build + baseline + lstm, eval on test
        .\run.ps1 -SkipBuild              # reuse cached chain tensors
        .\run.ps1 -Rnn GRU -Epochs 30
        .\run.ps1 -MaxRows 200000         # quick smoke: cap rows read per split
#>
param(
    [int]$Epochs = 20,
    [string]$Rnn = "LSTM",
    [int]$MaxTrainChains = 30000,
    [int]$MaxEvalChains = 15000,
    [int]$MaxRows = 0,
    [string]$Target = "DEP",
    [string]$Experiment = "chain_lstm",
    [switch]$SkipBuild,
    [switch]$SkipBaseline,
    [string]$Python = "python"
)

# tqdm 等库会向 stderr 写进度条; 用 Continue 避免 PowerShell 把 native stderr 当致命错误。
# 失败与否一律看 $LASTEXITCODE。
$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"

$env:CHAIN_EXPERIMENT_NAME = "$Experiment"
$env:CHAIN_TARGET          = "$Target"
$env:CHAIN_RNN             = "$Rnn"
$env:CHAIN_EPOCHS          = "$Epochs"
$env:CHAIN_MAX_TRAIN       = "$MaxTrainChains"
$env:CHAIN_MAX_VAL         = "$MaxEvalChains"
$env:CHAIN_MAX_TEST        = "$MaxEvalChains"
if ($MaxRows -gt 0) { $env:CHAIN_MAX_ROWS = "$MaxRows" } else { $env:CHAIN_MAX_ROWS = "none" }

# Resolve numbered scripts by glob so the Chinese filenames need not be embedded literally.
$build    = (Get-ChildItem -LiteralPath $PSScriptRoot -Filter '0_*.py' | Select-Object -First 1).Name
$baseline = (Get-ChildItem -LiteralPath $PSScriptRoot -Filter '2_*.py' | Select-Object -First 1).Name
$trainpy  = (Get-ChildItem -LiteralPath $PSScriptRoot -Filter '3_*.py' | Select-Object -First 1).Name

Write-Host "==================== run (Chain LSTM, official protocol) ====================" -ForegroundColor Cyan
Write-Host " experiment       = $Experiment"
Write-Host " target           = $Target (DEP_DELAY>15)"
Write-Host " rnn / epochs     = $Rnn / $Epochs"
Write-Host " chains train/eval= $MaxTrainChains / $MaxEvalChains"
Write-Host " max rows/split   = $(if ($MaxRows -gt 0) { $MaxRows } else { 'full' })"
Write-Host "=============================================================================" -ForegroundColor Cyan

if (-not $SkipBuild) {
    Write-Host "[0] build chain tensors..." -ForegroundColor Yellow
    & $Python $build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

if (-not $SkipBaseline) {
    Write-Host "[2] XGB baseline (15 static features)..." -ForegroundColor Yellow
    & $Python $baseline
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[3] chain LSTM..." -ForegroundColor Yellow
& $Python $trainpy --test
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

exit $LASTEXITCODE
