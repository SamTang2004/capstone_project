$ErrorActionPreference = 'Stop'
$llmRoot = Split-Path $PSScriptRoot -Parent
$out = Join-Path $llmRoot 'build/activation_sharing'
New-Item -ItemType Directory -Force $out | Out-Null
Push-Location $out
try {
    $bin = 'C:/Xilinx/Vivado/2024.2/bin'
    $names = @('llm_work_finder8','llm_pipelined_work_finder','llm_bitmap_finder64','llm_dot_product_lane','llm_dot_core8','llm_shared_activation8')
    $sources = $names | ForEach-Object { Join-Path $llmRoot "rtl/$_.sv" }
    & "$bin/xvlog.bat" --sv @sources "$PSScriptRoot/tb_shared_activation8.sv" "$bin/../data/verilog/src/glbl.v"
    if ($LASTEXITCODE -ne 0) { throw 'Compile failed' }
    foreach ($mode in 0,1,2) {
        $bitmap = [int]($mode -ne 0)
        $streaming = [int]($mode -eq 2)
        $generic = '"BITMAP=' + $bitmap + '"'
        $streamGeneric = '"STREAMING=' + $streaming + '"'
        & "$bin/xelab.bat" tb_shared_activation8 glbl -generic_top $generic -generic_top $streamGeneric -L unisims_ver --timescale 1ns/1ps -s "activation_sharing_$mode"
        if ($LASTEXITCODE -ne 0) { throw 'Elaboration failed' }
        & "$bin/xsim.bat" "activation_sharing_$mode" -runall -log "simulation_$mode.log"
        if ($LASTEXITCODE -ne 0 -or !(Select-String -LiteralPath "simulation_$mode.log" -SimpleMatch 'PASS: activation sharing' -Quiet)) { throw "Simulation failed for mode $mode" }
    }
} finally { Pop-Location }
