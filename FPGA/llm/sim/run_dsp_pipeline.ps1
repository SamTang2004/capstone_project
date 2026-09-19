$ErrorActionPreference='Stop'
$llmRoot=Split-Path $PSScriptRoot -Parent
$out=Join-Path $llmRoot 'build/dsp_pipeline_test'
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
    $bin='C:/Xilinx/Vivado/2024.2/bin'
    & "$bin/xvlog.bat" --sv "$llmRoot/rtl/llm_dot_product_lane.sv" "$PSScriptRoot/tb_dsp_pipeline.sv" "$bin/../data/verilog/src/glbl.v"
    if($LASTEXITCODE -ne 0){throw 'Compile failed'}
    & "$bin/xelab.bat" tb_dsp_pipeline glbl -L unisims_ver --timescale 1ns/1ps -s dsp_pipeline
    if($LASTEXITCODE -ne 0){throw 'Elaboration failed'}
    & "$bin/xsim.bat" dsp_pipeline -runall
    if($LASTEXITCODE -ne 0 -or !(Select-String -LiteralPath xsim.log -SimpleMatch 'PASS: DSP pipeline' -Quiet)){throw 'DSP pipeline test failed'}
} finally {Pop-Location}
