$ErrorActionPreference = 'Stop'
$llmRoot = Split-Path $PSScriptRoot -Parent
$sharedRtl = Join-Path (Split-Path $llmRoot -Parent) 'hardware/rtl'
$simulationDir = Join-Path $llmRoot 'build/integration'
New-Item -ItemType Directory -Force $simulationDir | Out-Null
Push-Location $simulationDir
try {
    $vivadoBin = 'C:/Xilinx/Vivado/2024.2/bin'
    $sources = @(Get-Content "$llmRoot/vivado/rtl_sources.txt" | Where-Object { $_.Trim() } | ForEach-Object { [IO.Path]::GetFullPath((Join-Path "$llmRoot/vivado" $_.Trim())) })
    & "$vivadoBin/xvlog.bat" --sv @sources "$PSScriptRoot/tb_llm_axi_queued_top.sv" "$vivadoBin/../data/verilog/src/glbl.v"
    if ($LASTEXITCODE -ne 0) { throw 'Integration compilation failed' }
    & "$vivadoBin/xelab.bat" tb_llm_axi_queued_top glbl -L unisims_ver --timescale 1ns/1ps -s llm_axi_test
    if ($LASTEXITCODE -ne 0) { throw 'Integration elaboration failed' }
    & "$vivadoBin/xsim.bat" llm_axi_test -runall
    if ($LASTEXITCODE -ne 0) { throw 'Integration simulation failed' }
    if (!(Select-String -LiteralPath 'xsim.log' -SimpleMatch 'PASS: queued AXI' -Quiet)) { throw 'Integration did not reach PASS' }
} finally { Pop-Location }
