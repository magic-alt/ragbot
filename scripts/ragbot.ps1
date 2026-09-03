param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"
$scriptPath = Join-Path $PSScriptRoot "ragbot.py"
$python = Get-Command python -ErrorAction SilentlyContinue

if (-not $python) {
    Write-Error "Python was not found in PATH. Ragbot requires Python 3.10+."
    exit 1
}

& $python.Source $scriptPath @RemainingArgs
exit $LASTEXITCODE
