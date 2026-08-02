# Phase 0 gate: does this Windows box have a working CUDA authoring toolchain?
#
# Run on the GPU host (locally or over SSH from the Mac):
#   powershell -ExecutionPolicy Bypass -File scripts\probe_windows.ps1
#
# Checks are ordered cheapest-first, but the one that actually decides the project
# is the last: whether torch.utils.cpp_extension can drive MSVC to compile a .cu.
# Everything else can be present and correct while that still fails, which is why
# it gets its own script rather than a version string.

$ErrorActionPreference = 'Continue'
$results = [ordered]@{}

function Probe($name, $block) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    try {
        $out = & $block 2>&1 | Out-String
        Write-Host $out.Trim()
        $script:results[$name] = if ($LASTEXITCODE -eq 0 -or $null -eq $LASTEXITCODE) { 'ok' } else { "exit $LASTEXITCODE" }
    } catch {
        Write-Host $_.Exception.Message -ForegroundColor Red
        $script:results[$name] = 'MISSING'
    }
}

Probe 'gpu' { nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv }
Probe 'nvcc' { nvcc --version }
Probe 'ncu' { ncu --version }

# nvcc needs a host C++ compiler; the CUDA Toolkit alone is not sufficient on Windows.
Probe 'msvc' {
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) { throw 'vswhere.exe not found — VS Build Tools not installed' }
    & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
}

Probe 'python' { python --version }
Probe 'torch' {
    python -c "import torch; print('torch', torch.__version__); print('cuda', torch.version.cuda); print('available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'); print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'n/a')"
}

Write-Host "`n=== summary ===" -ForegroundColor Cyan
$results.GetEnumerator() | ForEach-Object { "{0,-8} {1}" -f $_.Key, $_.Value }
Write-Host "`nNow run the real gate:  python scripts\probe_cuda_build.py" -ForegroundColor Yellow
