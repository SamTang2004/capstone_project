$ErrorActionPreference = 'Stop'
$llmRoot = Split-Path $PSScriptRoot -Parent
$shared = Join-Path (Split-Path $llmRoot -Parent) 'hardware/rtl'
$out = Join-Path $llmRoot 'build/diagonal'
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
    $bin = 'C:/Xilinx/Vivado/2024.2/bin'
    & "$bin/xvlog.bat" --sv "$shared/pe_int8_mac.sv" "$shared/matrix_array_8x8.sv" "$shared/weight_vector_decode.sv" "$shared/gemm_8x8_tile_core.sv" "$llmRoot/rtl/llm_tagged_tile.sv" "$llmRoot/rtl/llm_diagonal_tile.sv" "$PSScriptRoot/tb_diagonal_tile.sv" "$bin/../data/verilog/src/glbl.v"
    if ($LASTEXITCODE -ne 0) { throw 'Compile failed' }
    & "$bin/xelab.bat" tb_diagonal_tile glbl -L unisims_ver --timescale 1ns/1ps -s diagonal_test
    if ($LASTEXITCODE -ne 0) { throw 'Elaboration failed' }
    & "$bin/xsim.bat" diagonal_test -runall
    if ($LASTEXITCODE -ne 0 -or !(Select-String -LiteralPath 'xsim.log' -SimpleMatch 'PASS: diagonal RTL' -Quiet)) { throw 'Diagonal RTL test failed' }
} finally { Pop-Location }
