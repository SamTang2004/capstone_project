$ErrorActionPreference='Stop'
$llmRoot=Split-Path $PSScriptRoot -Parent
$out=Join-Path $llmRoot 'build/sparse_board_sim'
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
    $bin='C:/Xilinx/Vivado/2024.2/bin'
    $sources=@('llm_bitmap_finder64','llm_dot_product_lane','llm_dot_core8','llm_activation_block_map','llm_tiled_activation8','llm_sparse_board_memory') | ForEach-Object {Join-Path $llmRoot "rtl/$_.sv"}
    & "$bin/xvlog.bat" --sv @sources "$PSScriptRoot/tb_sparse_board_memory.sv" "$bin/../data/verilog/src/glbl.v"
    if($LASTEXITCODE -ne 0){throw 'Compile failed'}
    & "$bin/xelab.bat" tb_sparse_board_memory glbl -L unisims_ver --timescale 1ns/1ps -s sparse_board
    if($LASTEXITCODE -ne 0){throw 'Elaboration failed'}
    & "$bin/xsim.bat" sparse_board -runall
    if($LASTEXITCODE -ne 0 -or !(Select-String -LiteralPath xsim.log -SimpleMatch 'PASS: sparse board memory' -Quiet)){throw 'Board simulation failed'}
} finally {Pop-Location}
