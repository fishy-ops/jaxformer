@echo off
REM Profile the v1 attention kernel with Nsight Compute. Captures one steady-state
REM launch (skip warmups) and a curated metric set: compute/memory throughput,
REM achieved occupancy, register/shared-mem footprint, shared-memory bank conflicts,
REM and the dominant warp-stall reasons -- the "before" picture that motivates v2.
call C:\jaxformer\jf_env.bat
cd /d C:\jaxformer
set "NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.4.1\target\windows-desktop-win7-x64\ncu.exe"
set "M=gpu__time_duration.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,launch__registers_per_thread,launch__shared_mem_per_block_static,launch__waves_per_multiprocessor,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio,smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"
"%NCU%" --target-processes all -k "regex:fused_attn_fwd_kernel" --launch-skip 12 --launch-count 1 --metrics %M% --csv C:\jfvenv\Scripts\python.exe -m bench.profile_target
