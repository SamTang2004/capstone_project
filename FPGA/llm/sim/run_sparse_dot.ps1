$ErrorActionPreference = 'Stop'
$llmRoot = Split-Path $PSScriptRoot -Parent
$out = Join-Path $llmRoot 'build/sparse_dot'
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
    $bin = 'C:/Xilinx/Vivado/2024.2/bin'
    $names = @('llm_work_finder8', 'llm_survivor_buffer8', 'llm_work_packer8', 'llm_dot_product_lane', 'llm_dot_core8', 'llm_sparse_dot8')
    $sources = $names | ForEach-Object { Join-Path $llmRoot "rtl/$_.sv" }
    & "$bin/xvlog.bat" --sv @sources "$PSScriptRoot/tb_sparse_dot8.sv" "$bin/../data/verilog/src/glbl.v"
    if ($LASTEXITCODE -ne 0) { throw 'Compile failed' }
    & "$bin/xelab.bat" tb_sparse_dot8 glbl -L unisims_ver --timescale 1ns/1ps -s sparse_dot_test
    if ($LASTEXITCODE -ne 0) { throw 'Elaboration failed' }
    & "$bin/xsim.bat" sparse_dot_test -runall
    if ($LASTEXITCODE -ne 0 -or !(Select-String -LiteralPath 'xsim.log' -SimpleMatch 'PASS: sparse dot8' -Quiet)) { throw 'Sparse dot RTL test failed' }
} finally { Pop-Location }
