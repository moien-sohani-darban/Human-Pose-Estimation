[CmdletBinding()]
param(
    [switch]$SkipSmokeTest
)

$ErrorActionPreference = "Stop"
$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$engineRoot = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot "python-engine"))
$python = Join-Path $engineRoot ".venv\Scripts\python.exe"
$spec = Join-Path $engineRoot "hpe-python-sidecar.spec"
$workRoot = [System.IO.Path]::GetFullPath((Join-Path $engineRoot "build\pyinstaller"))
$distRoot = [System.IO.Path]::GetFullPath((Join-Path $engineRoot "dist"))
$artifact = Join-Path $distRoot "hpe-python-sidecar\hpe-python-sidecar.exe"

if (-not [System.IO.File]::Exists($python)) {
    throw "Task 13 requires the project Python environment: $python"
}
$pythonVersion = (& $python -c "import platform; print(platform.python_version())").Trim()
if ($pythonVersion -ne "3.13.5") {
    throw "Task 13 requires Python 3.13.5; found $pythonVersion"
}
$pyInstallerVersion = (& $python -m PyInstaller --version).Trim()
if ($pyInstallerVersion -ne "6.22.2") {
    throw "Install the pinned package build dependencies first. Expected PyInstaller 6.22.2; found '$pyInstallerVersion'."
}
if (-not [System.IO.File]::Exists($spec)) {
    throw "PyInstaller specification is missing: $spec"
}

foreach ($generatedRoot in @($workRoot, $distRoot)) {
    if (-not $generatedRoot.StartsWith($engineRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to clean generated path outside python-engine: $generatedRoot"
    }
    if ([System.IO.Directory]::Exists($generatedRoot)) {
        Remove-Item -LiteralPath $generatedRoot -Recurse -Force
    }
}
[System.IO.Directory]::CreateDirectory($workRoot) | Out-Null

$previousAutoInstall = $env:YOLO_AUTOINSTALL
$previousPipNoIndex = $env:PIP_NO_INDEX
$previousMatplotlibConfig = $env:MPLCONFIGDIR
try {
    # Packaging analysis imports Ultralytics modules. Disable its optional
    # dependency auto-installer and pip network access for a reproducible build.
    $env:YOLO_AUTOINSTALL = "false"
    $env:PIP_NO_INDEX = "1"
    $env:MPLCONFIGDIR = Join-Path $workRoot "matplotlib"
    Push-Location $engineRoot
    try {
        & $python -m PyInstaller --noconfirm --clean --distpath $distRoot --workpath $workRoot $spec
        if ($LASTEXITCODE -ne 0) {
            throw "PyInstaller failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
} finally {
    $env:YOLO_AUTOINSTALL = $previousAutoInstall
    $env:PIP_NO_INDEX = $previousPipNoIndex
    $env:MPLCONFIGDIR = $previousMatplotlibConfig
}

if (-not [System.IO.File]::Exists($artifact)) {
    throw "Expected packaged sidecar was not produced: $artifact"
}
if (-not $SkipSmokeTest) {
    & (Join-Path $PSScriptRoot "test-packaged-sidecar.ps1") -Executable $artifact
    if ($LASTEXITCODE -ne 0) {
        throw "Packaged sidecar smoke test failed"
    }
}

$size = (Get-ChildItem (Split-Path $artifact) -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host "Python sidecar package: $artifact"
Write-Host ("Packaged runtime size: {0:N0} bytes" -f $size)
