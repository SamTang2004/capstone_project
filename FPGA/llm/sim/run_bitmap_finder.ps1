$ErrorActionPreference = 'Stop'
$llmRoot = Split-Path $PSScriptRoot -Parent
$out = Join-Path $llmRoot 'build/bitmap_finder'
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
    $bin = 'C:/Xilinx/Vivado/2024.2/bin'
    & "$bin/xvlog.bat" --sv "$llmRoot/rtl/llm_bitmap_finder64.sv" "$PSScriptRoot/tb_bitmap_finder64.sv"
    if ($LASTEXITCODE -ne 0) { throw 'Compile failed' }
    & "$bin/xelab.bat" tb_bitmap_finder64 --timescale 1ns/1ps -s bitmap_finder
    if ($LASTEXITCODE -ne 0) { throw 'Elaboration failed' }
    & "$bin/xsim.bat" bitmap_finder -runall
    if ($LASTEXITCODE -ne 0 -or !(Select-String -LiteralPath 'xsim.log' -SimpleMatch 'PASS: bitmap finder' -Quiet)) { throw 'Bitmap test failed' }
} finally { Pop-Location }
