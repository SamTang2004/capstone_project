param([switch]$MergeBanks, [switch]$PipelineMerge)
$ErrorActionPreference = 'Stop'
$llmRoot = Split-Path $PSScriptRoot -Parent
$out = Join-Path $llmRoot 'build/wide_sparse_dot'
if ($MergeBanks) { $out = Join-Path $llmRoot 'build/merged_sparse_dot' }
$elabOptions = @()
if ($MergeBanks) { $elabOptions = @('-generic_top', '"MERGE_BANKS=1"') }
if ($PipelineMerge) {
    $out = Join-Path $llmRoot 'build/pipelined_sparse_dot'
    $elabOptions = @('-generic_top', '"MERGE_BANKS=1"', '-generic_top', '"PIPELINE_MERGE=1"')
}
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
    $bin = 'C:/Xilinx/Vivado/2024.2/bin'
    $names = @('llm_work_finder8', 'llm_survivor_buffer8', 'llm_work_packer8', 'llm_dot_product_lane', 'llm_dot_core8', 'llm_sparse_dot8', 'llm_pipelined_work_finder', 'llm_wide_work_finder', 'llm_wide_sparse_dot8')
    $sources = $names | ForEach-Object { Join-Path $llmRoot "rtl/$_.sv" }
    & "$bin/xvlog.bat" --sv @sources "$PSScriptRoot/tb_wide_sparse_dot8.sv" "$bin/../data/verilog/src/glbl.v"
    if ($LASTEXITCODE -ne 0) { throw 'Compile failed' }
    & "$bin/xelab.bat" tb_wide_sparse_dot8 glbl @elabOptions -L unisims_ver --timescale 1ns/1ps -s wide_sparse_dot_test
    if ($LASTEXITCODE -ne 0) { throw 'Elaboration failed' }
    & "$bin/xsim.bat" wide_sparse_dot_test -runall
    if ($LASTEXITCODE -ne 0 -or !(Select-String -LiteralPath 'xsim.log' -SimpleMatch 'PASS: wide sparse dot8' -Quiet)) { throw 'Wide sparse dot test failed' }
} finally { Pop-Location }
