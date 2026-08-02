# Windows GPU box (RTX 2070 Super) — working toolchain

Reproducible setup for the CUDA-kernel half of the project. Reached over Tailscale
SSH as `reach@100.109.40.37` (host alias `akpc`). The session runs elevated (High
Mandatory Level), which the CUDA/component steps need.

## What's installed

| Piece | Version | Notes |
|---|---|---|
| GPU | RTX 2070 SUPER, sm_75, 8 GB | driver **596.36** — newer than any CUDA bundle |
| MSVC | 19.44 (toolset 14.44.35207) | VS 2022 Build Tools |
| CUDA toolkit | 12.8.93 | **hand-assembled**, see below |
| Python | 3.13 | venv at `C:\jfvenv` |
| torch | **2.8.0+cu128** | 2.11.0 does NOT build extensions here (see gotcha) |
| ninja | 1.13 | required by cpp_extension |
| 7-Zip | 26.02 | used to crack the CUDA installer |

## The one that matters: CUDA_HOME was assembled by hand

The monolithic CUDA installer (`cuda_12.8.1_572.61_windows.exe`, both network and
local variants) **cannot be used**: it bundles the full GeForce Game Ready driver,
force-selects it regardless of the `-s <packages>` component filter, and aborts with
`-522190823` because the bundled 572.61 driver is older than the installed 596.36.
Confirmed in the installer log: it selects `Display.Driver`, `ShadowPlay`,
`FrameViewSdk`, `NvTelemetry`, ... — the whole driver stack — and never honors `-s`.

NVIDIA installers are 7-Zip self-extracting archives, so the fix is to unpack the
component payloads and merge the toolkit slices by hand — no driver, no installer,
no registry:

```
7z x cuda_12.8.1_572.61_windows.exe -oC:\cuda7z
# merge these inner trees into C:\cudahome (each mirrors bin/ include/ lib\x64/ nvvm/):
#   cuda_nvcc\nvcc, cuda_cudart\cudart, cuda_nvrtc\{nvrtc,nvrtc_dev},
#   cuda_nvdisasm\nvdisasm, cuda_profiler_api\cuda_profiler_api,
#   libcublas\{cublas,cublas_dev}
# + cccl headers from the pip wheel (nvidia-cuda-cccl-cu12)
```
`scripts/assemble_cudahome.ps1` does exactly this. Result: `C:\cudahome` with
`bin\nvcc.exe`, `nvvm\`, `include\cuda_runtime.h`, `lib\x64\cudart.lib`.

The pip `nvidia-cuda-nvcc-cu12` wheel is NOT enough on Windows — it ships only
`ptxas.exe`, no `nvcc.exe`. But its cudart wheel does provide headers + import libs.

## Gotcha: torch 2.11.0 will not build CUDA extensions here

nvcc compiling `torch/csrc/dynamo/compiled_autograd.h` fails with
`error C2872: 'std': ambiguous symbol` at `::std::string` in a `constexpr` branch —
nvcc's frontend choking on MSVC 19.44's STL. **torch 2.8.0+cu128 compiles cleanly.**
If a future torch bump reintroduces this, either pin back to 2.8.0 or add the MSVC
14.43 toolset and pass `vcvars64.bat -vcvars_ver=14.43`.

nvcc's own MSVC guard (`host_config.h`: `_MSC_VER < 1910 || _MSC_VER >= 1950`) does
pass for 19.44 (1944) — so `-allow-unsupported-compiler` is NOT needed.

## Running anything on the box

SSH sessions are non-interactive: neither the MSVC dev env nor `C:\cudahome` is on
PATH. `C:\jaxformer\jf_env.bat` sets both (sources vcvars64, exports CUDA_HOME, puts
`C:\cudahome\bin` and the venv Scripts on PATH). Always:

```
ssh akpc "cmd /c \"call C:\jaxformer\jf_env.bat && C:\jfvenv\Scripts\python.exe <script>\""
```

Gate verified with `scripts/probe_cuda_build.py` → `PASS: cpp_extension built,
launched, and returned correct results`.
