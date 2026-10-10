# =============================================================================
# Setup script: creates a virtual environment and installs Python dependencies.
#
# Usage (from the project root, in PowerShell):
#   ./setup.ps1          # venv + Python deps + CPU llama-cpp-python (fallback)
#   ./setup.ps1 -NoCpu   # venv + Python deps only (GPU-via-server users)
#
# GPU inference (recommended) does NOT use this script to install a backend.
# Instead it uses a prebuilt Vulkan release of llama.cpp — see the GPU section
# in README.md. That path needs no compiler and no SDK: just download the
# Windows Vulkan release, extract it, and point config/bench.yaml at
# llama-server.exe. This script only sets up the Python side.
# =============================================================================

param(
    [switch]$NoCpu
)

$ErrorActionPreference = "Stop"

# 1. Create the venv if it does not already exist.
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment in .venv ..." -ForegroundColor Cyan
    python -m venv .venv
}

# 2. Activate it for this session.
Write-Host "Activating .venv ..." -ForegroundColor Cyan
& ".\.venv\Scripts\Activate.ps1"

# 3. Upgrade pip tooling.
python -m pip install --upgrade pip wheel setuptools

# 4. Install the pure-Python dependencies (harness, TUI, HTTP client).
python -m pip install PyYAML tabulate rich requests pytest

# 5. Optionally install a prebuilt CPU wheel of llama-cpp-python for the
#    'python' (CPU) backend. This uses the official prebuilt wheel index, so
#    no compiler is required. GPU users can skip this with -NoCpu.
if (-not $NoCpu) {
    Write-Host "Installing prebuilt CPU llama-cpp-python (fallback backend) ..." -ForegroundColor Cyan
    python -m pip install llama-cpp-python `
        --only-binary=:all: `
        --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
}

Write-Host ""
Write-Host "Python setup complete." -ForegroundColor Green
Write-Host "Activate later with:  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "For GPU (AMD RX 9070 / Vulkan):" -ForegroundColor Yellow
Write-Host "  1. Download the llama.cpp Windows Vulkan release (llama-*-bin-win-vulkan-x64.zip)"
Write-Host "     from https://github.com/ggml-org/llama.cpp/releases"
Write-Host "  2. Extract it into the project folder."
Write-Host "  3. Set llama.server_bin in config/bench.yaml to the extracted llama-server.exe"
Write-Host "  4. Ensure backend: server in config/bench.yaml (this is the default)."
