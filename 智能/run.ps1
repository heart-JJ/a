param(
    [int]$Port = 8787,
    [string]$Database = "data/evoagent.db"
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
    & $Python -m evoagent --db $Database serve --host 127.0.0.1 --port $Port
} finally {
    Pop-Location
}
