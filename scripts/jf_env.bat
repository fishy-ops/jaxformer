@echo off
REM One-stop environment for JaxFormer GPU work on this box.
REM SSH sessions are non-interactive and have neither the MSVC dev environment nor
REM the hand-assembled CUDA toolkit on PATH; cpp_extension needs both (it shells out
REM to `cl` and `nvcc`). Source this before any build/bench/train command:
REM   cmd /c "call C:\jaxformer\jf_env.bat && <python ...>"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\cudahome"
set "CUDA_PATH=C:\cudahome"
set "PATH=C:\cudahome\bin;C:\jfvenv\Scripts;%PATH%"
