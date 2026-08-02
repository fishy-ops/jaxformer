# Assemble a standard CUDA_HOME from the 7-Zip-extracted installer payloads.
#
# The monolithic installer refused to run (it force-bundles the GeForce driver,
# which is older than the installed 596.36 and aborts). But the payloads it ships
# are just plain files: each cuda_<component>\<name>\ folder mirrors a slice of the
# toolkit tree (bin / include / lib\x64 / nvvm). Merging the ones the kernel build
# needs yields a working toolkit with no driver, no installer, no registry changes.
$ErrorActionPreference = 'Stop'
$H = 'C:\cudahome'
$src = 'C:\cuda7z'

Remove-Item $H -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $H | Out-Null

# Each of these inner folders is a toolkit-shaped subtree; robocopy merges them.
$trees = @(
  "$src\cuda_nvcc\nvcc",                       # nvcc, ptxas, cicc, nvvm, crt headers
  "$src\cuda_cudart\cudart",                   # cuda_runtime.h, cudart.lib, cudart DLL
  "$src\cuda_nvrtc\nvrtc",                      # nvrtc DLLs
  "$src\cuda_nvrtc\nvrtc_dev",                  # nvrtc headers + import lib
  "$src\cuda_nvdisasm\nvdisasm",               # nvdisasm (nvcc pulls it in)
  "$src\cuda_profiler_api\cuda_profiler_api",  # cuda_profiler_api.h
  "$src\libcublas\cublas",                     # cublas DLLs
  "$src\libcublas\cublas_dev"                  # cublas headers + import lib
)
foreach ($t in $trees) {
  if (Test-Path $t) { robocopy $t $H /E /NFL /NDL /NJH /NJS /NP | Out-Null }
  else { Write-Host "MISSING TREE: $t" }
}

# CCCL headers (thrust / cub / libcudacxx). torch/extension.h transitively includes
# them; prefer the versioned 7z copy, fall back to the pip wheel.
$cccl = @("$src\cuda_cccl\cccl\include", "C:\jfvenv\Lib\site-packages\nvidia\cuda_cccl\include") |
        Where-Object { Test-Path $_ } | Select-Object -First 1
if ($cccl) { robocopy $cccl "$H\include" /E /NFL /NDL /NJH /NJS /NP | Out-Null; Write-Host "cccl from $cccl" }
else { Write-Host "WARNING: no cccl headers found" }

Write-Host "`n=== verify CUDA_HOME layout ==="
$need = @('bin\nvcc.exe','bin\ptxas.exe','nvvm\bin\cicc.exe','nvvm\libdevice',
          'include\cuda_runtime.h','include\crt','lib\x64\cudart.lib')
$ok = $true
foreach ($f in $need) {
  $p = Join-Path $H $f
  if (Test-Path $p) { Write-Host "  ok  $f" } else { Write-Host "  MISSING  $f"; $ok = $false }
}
Write-Host "`n=== nvcc --version ==="
& "$H\bin\nvcc.exe" --version | Select-String 'release'
if ($ok) { Write-Host "CUDAHOME_OK" } else { Write-Host "CUDAHOME_INCOMPLETE" }
