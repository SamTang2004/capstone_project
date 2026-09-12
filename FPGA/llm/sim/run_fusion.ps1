$ErrorActionPreference = 'Stop'
$llmRoot = Split-Path $PSScriptRoot -Parent
$repoRoot = Split-Path (Split-Path $llmRoot -Parent) -Parent
$python = Join-Path $repoRoot '.venv/Scripts/python.exe'
$shared = Join-Path $repoRoot 'FPGA/hardware/rtl'
$out = Join-Path $llmRoot 'build/fusion_goldens'
& $python "$llmRoot/research/fusion_goldens.py" generate --output $out
if ($LASTEXITCODE -ne 0) { throw 'Golden generation failed' }
Push-Location $out
try {
    $bin = 'C:/Xilinx/Vivado/2024.2/bin'
    & "$bin/xvlog.bat" --sv "$shared/pe_int8_mac.sv" "$shared/matrix_array_8x8.sv" "$shared/weight_vector_decode.sv" "$shared/gemm_8x8_tile_core.sv" "$llmRoot/rtl/llm_tagged_tile.sv" "$PSScriptRoot/tb_fusion_goldens.sv" "$bin/../data/verilog/src/glbl.v"
    if ($LASTEXITCODE -ne 0) { throw 'Compile failed' }
    & "$bin/xelab.bat" tb_fusion_goldens glbl -L unisims_ver --timescale 1ns/1ps -s fusion_test
    if ($LASTEXITCODE -ne 0) { throw 'Elaboration failed' }
    & "$bin/xsim.bat" fusion_test -runall
    if ($LASTEXITCODE -ne 0 -or !(Select-String -LiteralPath 'xsim.log' -SimpleMatch 'PASS: fusion golden RTL' -Quiet)) { throw 'RTL golden failed' }
    & $python "$llmRoot/research/fusion_goldens.py" verify --output $out
    if ($LASTEXITCODE -ne 0) { throw 'End-to-end tagged golden failed' }
} finally { Pop-Location }
