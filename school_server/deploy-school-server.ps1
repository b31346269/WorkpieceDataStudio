param(
    [string]$HostAlias = "sslab-school",
    [string]$RemoteDir = "~/workpiece_data_studio",
    [switch]$Bootstrap
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Archive = Join-Path $env:TEMP "workpiece-data-studio-upload.tgz"
$RemoteArchive = "~/workpiece-data-studio-upload.tgz"

if (Test-Path -LiteralPath $Archive) {
    Remove-Item -LiteralPath $Archive -Force
}

Push-Location $Root
try {
    & tar.exe -czf $Archive `
        --exclude=.venv `
        --exclude=.school-env `
        --exclude=workspace/.huggingface `
        --exclude='workspace/projects/*/imports/*.zip' `
        --exclude='workspace/projects/*/exports/*.zip' `
        --exclude='**/__pycache__' `
        --exclude='*.pyc' `
        .
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the deployment archive."
    }

    & scp.exe $Archive "${HostAlias}:$RemoteArchive"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not upload the project through SSH."
    }

    & ssh.exe $HostAlias "mkdir -p $RemoteDir && tar -xzf $RemoteArchive -C $RemoteDir && rm -f $RemoteArchive"
    if ($LASTEXITCODE -ne 0) {
        throw "Could not unpack the project on the school server."
    }

    if ($Bootstrap) {
        & ssh.exe -t $HostAlias "cd $RemoteDir && bash school_server/bootstrap.sh"
        if ($LASTEXITCODE -ne 0) {
            throw "School server environment setup failed."
        }
    }
} finally {
    Pop-Location
    Remove-Item -LiteralPath $Archive -Force -ErrorAction SilentlyContinue
}

Write-Host "Project deployed to ${HostAlias}:$RemoteDir"
if (-not $Bootstrap) {
    Write-Host "Run again with -Bootstrap to install the remote Python environment."
}
