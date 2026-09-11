$ErrorActionPreference = 'Stop'
$llmRoot = Split-Path $PSScriptRoot -Parent
$sharedRtl = Join-Path (Split-Path $llmRoot -Parent) 'hardware/rtl'
$simulationDir = Join-Path $llmRoot 'build/sim'
New-Item -ItemType Directory -Force $simulationDir | Out-Null
Push-Location $simulationDir
try {
    $vivadoBin = 'C:/Xilinx/Vivado/2024.2/bin'
    & "$vivadoBin/xvlog.bat" --sv "$sharedRtl/pe_int8_mac.sv" "$sharedRtl/matrix_array_8x8.sv" "$sharedRtl/weight_vector_decode.sv" "$sharedRtl/gemm_8x8_tile_core.sv" "$llmRoot/rtl/llm_tagged_tile.sv" "$PSScriptRoot/tb_llm_tagged_tile.sv"
    if ($LASTEXITCODE -ne 0) { throw 'xvlog failed' }
    & "$vivadoBin/xvlog.bat" "$vivadoBin/../data/verilog/src/glbl.v"
    if ($LASTEXITCODE -ne 0) { throw 'glbl compilation failed' }
    & "$vivadoBin/xelab.bat" tb_llm_tagged_tile glbl -L unisims_ver --timescale 1ns/1ps -s llm_tile_test
    if ($LASTEXITCODE -ne 0) { throw 'xelab failed' }
    & "$vivadoBin/xsim.bat" llm_tile_test -runall
    if ($LASTEXITCODE -ne 0) { throw 'xsim failed' }
    if (!(Select-String -LiteralPath 'xsim.log' -SimpleMatch 'PASS: LLM tagged tile' -Quiet)) { throw 'Simulation did not reach PASS' }
    & "$vivadoBin/xvlog.bat" --sv "$llmRoot/rtl/llm_command_queue.sv" "$PSScriptRoot/tb_llm_command_queue.sv"
    if ($LASTEXITCODE -ne 0) { throw 'Queue compilation failed' }
    & "$vivadoBin/xelab.bat" tb_llm_command_queue --timescale 1ns/1ps -s llm_queue_test
    if ($LASTEXITCODE -ne 0) { throw 'Queue elaboration failed' }
    & "$vivadoBin/xsim.bat" llm_queue_test -runall -log queue_xsim.log
    if ($LASTEXITCODE -ne 0) { throw 'Queue simulation failed' }
    if (!(Select-String -LiteralPath 'queue_xsim.log' -SimpleMatch 'PASS: LLM command queue' -Quiet)) { throw 'Queue did not reach PASS' }
} finally { Pop-Location }
