[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))

& (Join-Path $PSScriptRoot "build-python-sidecar.ps1")
if ($LASTEXITCODE -ne 0) {
    throw "Python sidecar packaging failed"
}

Push-Location $repositoryRoot
try {
    & yarn.cmd tauri build --config src-tauri/tauri.release.conf.json
    if ($LASTEXITCODE -ne 0) {
        throw "Tauri Windows release build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
