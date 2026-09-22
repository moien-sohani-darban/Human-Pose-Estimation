[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Executable
)

$ErrorActionPreference = "Stop"
$executablePath = [System.IO.Path]::GetFullPath($Executable)
if (-not [System.IO.File]::Exists($executablePath)) {
    throw "Packaged sidecar executable is missing: $executablePath"
}

$missingModel = [System.IO.Path]::Combine(
    [System.IO.Path]::GetDirectoryName($executablePath),
    "__packaging_smoke_missing__.task"
)
if ([System.IO.File]::Exists($missingModel)) {
    throw "Reserved smoke-test path unexpectedly exists: $missingModel"
}

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$frameFixture = Join-Path $repositoryRoot "python-engine\tests\test_person.jfif"
if (-not [System.IO.File]::Exists($frameFixture)) {
    throw "Frame smoke-test fixture is missing: $frameFixture"
}
$frameBytes = [System.IO.File]::ReadAllBytes($frameFixture)
if ($frameBytes.Length -gt (4 * 1024 * 1024)) {
    throw "Frame smoke-test fixture exceeds the 4 MiB protocol limit"
}
$requests = @(
    @{ id = "1"; protocol_version = 1; type = "ping" },
    @{ id = "2"; protocol_version = 1; type = "get_backends" },
    @{
        id = "3"
        protocol_version = 1
        type = "estimate_frame"
        backend = "mediapipe"
        image_base64 = [Convert]::ToBase64String($frameBytes)
        backend_config = @{ model_path = $missingModel }
    },
    @{ id = "4"; protocol_version = 1; type = "shutdown" }
)

$startInfo = [System.Diagnostics.ProcessStartInfo]::new()
$startInfo.FileName = $executablePath
$startInfo.WorkingDirectory = [System.IO.Path]::GetDirectoryName($executablePath)
$startInfo.UseShellExecute = $false
$startInfo.CreateNoWindow = $true
$startInfo.RedirectStandardInput = $true
$startInfo.RedirectStandardOutput = $true
$startInfo.RedirectStandardError = $true
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)

if ($startInfo.PSObject.Properties.Name -contains "StandardInputEncoding") {
    $startInfo.StandardInputEncoding = $utf8NoBom
}
if ($startInfo.PSObject.Properties.Name -contains "StandardOutputEncoding") {
    $startInfo.StandardOutputEncoding = $utf8NoBom
}
if ($startInfo.PSObject.Properties.Name -contains "StandardErrorEncoding") {
    $startInfo.StandardErrorEncoding = $utf8NoBom
}
$startInfo.Environment["HPE_PACKAGED_IMPORT_CHECK"] = "1"
$startInfo.Environment["YOLO_AUTOINSTALL"] = "false"

$process = [System.Diagnostics.Process]::new()
$process.StartInfo = $startInfo
if (-not $process.Start()) {
    throw "Failed to start packaged sidecar: $executablePath"
}

$stdoutTask = $process.StandardOutput.ReadToEndAsync()
$stderrTask = $process.StandardError.ReadToEndAsync()
foreach ($request in $requests) {
    $process.StandardInput.WriteLine(($request | ConvertTo-Json -Compress -Depth 5))
}
$process.StandardInput.Close()

if (-not $process.WaitForExit(120000)) {
    $process.Kill($true)
    $process.WaitForExit()
    throw "Packaged sidecar smoke test timed out"
}
$stdout = $stdoutTask.GetAwaiter().GetResult()
$stderr = $stderrTask.GetAwaiter().GetResult()
if ($process.ExitCode -ne 0) {
    throw "Packaged sidecar exited with code $($process.ExitCode). stderr: $stderr"
}

$lines = @($stdout -split "`r?`n" | Where-Object { $_ -ne "" })
if ($lines.Count -ne $requests.Count) {
    throw "Expected $($requests.Count) protocol lines, received $($lines.Count). stdout: $stdout"
}
$responses = @($lines | ForEach-Object { $_ | ConvertFrom-Json })
if (-not $responses[0].ok -or $responses[0].id -ne "1" -or $responses[0].result.protocol_version -ne 1) {
    throw "Packaged ping response was invalid: $($lines[0])"
}
if (-not $responses[1].ok -or $responses[1].id -ne "2") {
    throw "Packaged get_backends response was invalid: $($lines[1])"
}
if (@($responses[1].result.available) -notcontains "mediapipe" -or @($responses[1].result.available) -notcontains "yolo") {
    throw "Packaged backend discovery did not advertise both supported backends"
}
if ($responses[2].ok -or $responses[2].id -ne "3" -or $responses[2].error.code -ne "model_asset_not_found") {
    throw "Packaged estimate_frame did not preserve the typed missing-model error: $($lines[2])"
}
if (-not $responses[3].ok -or $responses[3].id -ne "4" -or $responses[3].result.status -ne "shutdown") {
    throw "Packaged shutdown response was invalid: $($lines[3])"
}
if ($stderr -notmatch '\[packaged-import-check\].*python=3\.13\.5.*numpy=.*opencv=.*mediapipe=.*torch=.*torchvision=.*ultralytics=.*app=app') {
    throw "Packaged dependency import marker was missing. stderr: $stderr"
}

Write-Host "Packaged sidecar smoke test passed."
Write-Host ($stderr.Trim())
Write-Host $stdout.Trim()
