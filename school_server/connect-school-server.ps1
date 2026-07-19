param(
    [string]$HostAlias = "sslab-school",
    [string]$RemoteDir = "~/workpiece_data_studio",
    [int]$LocalPort = 7866,
    [int]$RemotePort = 7865,
    [ValidateSet(2, 3, 6)]
    [int]$RemoteGpu = 2
)

$ErrorActionPreference = "Stop"

& ssh.exe $HostAlias "cd $RemoteDir && PHYSICAL_GPU=$RemoteGpu PORT=$RemotePort bash school_server/start_ui.sh --background"
if ($LASTEXITCODE -ne 0) {
    throw "Could not start Workpiece Data Studio on the school server."
}

Write-Host ""
Write-Host "School GPU UI: http://127.0.0.1:$LocalPort"
Write-Host "Remote generation GPU: physical card $RemoteGpu"
Write-Host "Keep this window open. Press Ctrl+C to close the SSH tunnel."
Write-Host ""

& ssh.exe `
    -o ExitOnForwardFailure=yes `
    -o ServerAliveInterval=30 `
    -o ServerAliveCountMax=3 `
    -N `
    -L "${LocalPort}:127.0.0.1:${RemotePort}" `
    $HostAlias
