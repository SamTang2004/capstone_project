param([int]$ClockMHz=100,[int]$MemoryScale=1,[string]$RunName='tiled_activation')
$ErrorActionPreference='Stop'
if($ClockMHz -le 0 -or $MemoryScale -lt 1 -or $RunName -notmatch '^[a-zA-Z0-9_]+$'){throw 'Invalid simulation parameters'}
$llmRoot=Split-Path $PSScriptRoot -Parent
$out=Join-Path $llmRoot "build/$RunName"
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
    $bin='C:/Xilinx/Vivado/2024.2/bin'
    $sources=@('llm_bitmap_finder64','llm_dot_product_lane','llm_dot_core8','llm_activation_block_map','llm_tiled_activation8') | ForEach-Object { Join-Path $llmRoot "rtl/$_.sv" }
    & "$bin/xvlog.bat" --sv @sources "$PSScriptRoot/tb_tiled_activation8.sv" "$bin/../data/verilog/src/glbl.v"
    if($LASTEXITCODE -ne 0){throw 'Compile failed'}
    $clockGeneric='"CLOCK_MHZ='+$ClockMHz+'"'
    $memoryGeneric='"MEMORY_SCALE='+$MemoryScale+'"'
    & "$bin/xelab.bat" tb_tiled_activation8 glbl -generic_top $clockGeneric -generic_top $memoryGeneric -L unisims_ver --timescale 1ns/1ps -s tiled_activation
    if($LASTEXITCODE -ne 0){throw 'Elaboration failed'}
    & "$bin/xsim.bat" tiled_activation -runall
    if($LASTEXITCODE -ne 0 -or !(Select-String -LiteralPath xsim.log -SimpleMatch 'PASS: tiled activation' -Quiet)){throw 'Tiled activation test failed'}
} finally {Pop-Location}
