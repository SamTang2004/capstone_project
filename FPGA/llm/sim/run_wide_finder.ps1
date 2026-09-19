param([switch]$MergeBanks, [switch]$PipelineMerge)
$ErrorActionPreference = 'Stop'
$llmRoot = Split-Path $PSScriptRoot -Parent
$out = Join-Path $llmRoot 'build/wide_finder'
if ($MergeBanks) { $out = Join-Path $llmRoot 'build/merged_finder' }
$elabOptions = @()
if ($MergeBanks) { $elabOptions = @('-generic_top', '"MERGE_BANKS=1"') }
if ($PipelineMerge) {
    $out = Join-Path $llmRoot 'build/pipelined_finder'
    $elabOptions = @('-generic_top', '"MERGE_BANKS=1"', '-generic_top', '"PIPELINE_MERGE=1"')
}
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
    $bin = 'C:/Xilinx/Vivado/2024.2/bin'
    & "$bin/xvlog.bat" --sv "$llmRoot/rtl/llm_work_finder8.sv" "$llmRoot/rtl/llm_pipelined_work_finder.sv" "$llmRoot/rtl/llm_wide_work_finder.sv" "$PSScriptRoot/tb_wide_work_finder.sv"
    if ($LASTEXITCODE -ne 0) { throw 'Compile failed' }
    & "$bin/xelab.bat" tb_wide_work_finder @elabOptions --timescale 1ns/1ps -s wide_finder_test
    if ($LASTEXITCODE -ne 0) { throw 'Elaboration failed' }
    & "$bin/xsim.bat" wide_finder_test -runall
    if ($LASTEXITCODE -ne 0 -or !(Select-String -LiteralPath 'xsim.log' -SimpleMatch 'PASS: wide finder' -Quiet)) { throw 'Wide finder test failed' }
} finally { Pop-Location }
