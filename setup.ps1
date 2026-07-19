param(
    [switch]$WithML
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Venv = Join-Path $Root ".venv"

function Find-Python {
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        try {
            & $pyLauncher.Source -3.12 -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0) {
                return @($pyLauncher.Source, "-3.12")
            }
        } catch {
        }
    }

    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        return @($python.Source)
    }

    throw "Python 3.11 or 3.12 was not found. Install Python, then run this script again."
}

$PythonCommand = Find-Python
if (-not (Test-Path -LiteralPath $Venv)) {
    if ($PythonCommand.Count -eq 2) {
        & $PythonCommand[0] $PythonCommand[1] -m venv $Venv
    } else {
        & $PythonCommand[0] -m venv $Venv
    }
}

$VenvPython = Join-Path $Venv "Scripts\python.exe"
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $Root "requirements-core.txt")

if ($WithML) {
    $TorchCudaReady = $false
    try {
        & $VenvPython -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() and torch.version.cuda else 1)"
        $TorchCudaReady = $LASTEXITCODE -eq 0
    } catch {
        $TorchCudaReady = $false
    }

    if (-not $TorchCudaReady) {
        $NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
        if ($NvidiaSmi) {
            & $VenvPython -m pip install --upgrade --force-reinstall torch torchvision `
                --index-url https://download.pytorch.org/whl/cu130
        } else {
            Write-Warning "NVIDIA GPU was not detected; installing CPU PyTorch."
            & $VenvPython -m pip install --upgrade torch torchvision
        }
    }
    & $VenvPython -m pip install -r (Join-Path $Root "requirements-ml.txt")
}

Write-Host ""
Write-Host "Setup complete."
if (-not $WithML) {
    Write-Host "Core mode is installed. Run '.\setup.ps1 -WithML' before real AI generation."
}
Write-Host "Start with: .\run.ps1"
Write-Host "Open: http://127.0.0.1:7865"
