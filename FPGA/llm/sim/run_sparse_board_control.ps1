$ErrorActionPreference='Stop'
$llmRoot=Split-Path $PSScriptRoot -Parent
$out=Join-Path $llmRoot 'build/sparse_board_control_sim'
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
    $bin='C:/Xilinx/Vivado/2024.2/bin'
    & "$bin/xvlog.bat" --sv "$llmRoot/rtl/llm_sparse_board_control.sv" "$PSScriptRoot/tb_sparse_board_control.sv"
    if($LASTEXITCODE -ne 0){throw 'Compile failed'}
    & "$bin/xelab.bat" tb_sparse_board_control --timescale 1ns/1ps -s board_control
    if($LASTEXITCODE -ne 0){throw 'Elaboration failed'}
    & "$bin/xsim.bat" board_control -runall
    if($LASTEXITCODE -ne 0 -or !(Select-String -LiteralPath xsim.log -SimpleMatch 'PASS: sparse board AXI' -Quiet)){throw 'AXI simulation failed'}
} finally {Pop-Location}
