param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$EvoArguments
)

$ErrorActionPreference = "Stop"
$ProfileRoot = [Environment]::GetEnvironmentVariable("USERPROFILE")
$BundledPython = if ($ProfileRoot) {
    Join-Path $ProfileRoot ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
} else {
    ""
}

if ($BundledPython -and (Test-Path -LiteralPath $BundledPython)) {
    $Python = $BundledPython
} else {
    $Python = (Get-Command python -ErrorAction Stop).Source
}

Push-Location -LiteralPath $PSScriptRoot
try {
    & $Python -m evoagent @EvoArguments
    exit $LASTEXITCODE
} finally {
    Pop-Location
}

