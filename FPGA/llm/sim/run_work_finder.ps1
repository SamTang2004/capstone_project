$ErrorActionPreference = 'Stop'
$llmRoot = Split-Path $PSScriptRoot -Parent
$out = Join-Path $llmRoot 'build/work_finder'
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
    $bin = 'C:/Xilinx/Vivado/2024.2/bin'
    & "$bin/xvlog.bat" --sv "$llmRoot/rtl/llm_work_finder8.sv" "$PSScriptRoot/tb_work_finder8.sv"
    if ($LASTEXITCODE -ne 0) { throw 'Compile failed' }
    & "$bin/xelab.bat" tb_work_finder8 --timescale 1ns/1ps -s work_finder_test
    if ($LASTEXITCODE -ne 0) { throw 'Elaboration failed' }
    & "$bin/xsim.bat" work_finder_test -runall
    if ($LASTEXITCODE -ne 0 -or !(Select-String -LiteralPath 'xsim.log' -SimpleMatch 'PASS: work finder8' -Quiet)) { throw 'Work finder RTL test failed' }
} finally { Pop-Location }
