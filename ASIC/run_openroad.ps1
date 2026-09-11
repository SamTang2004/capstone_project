param(
    [ValidateSet('synth', 'floorplan', 'place', 'cts', 'grt', 'route', 'finish', 'all')]
    [string]$Target = 'finish',
    [ValidateSet('wsl', 'docker')]
    [string]$Engine = 'wsl',
    [ValidateSet('full', 'dual-gemm')]
    [string]$Design = 'dual-gemm',
    [ValidateSet('nangate45', 'asap7')]
    [string]$Platform = 'nangate45',
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

$asicDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectDir = Split-Path -Parent $asicDir
$flowDir = Join-Path $asicDir 'OpenROAD-flow-scripts\flow'
$configRelative = switch ("$Design-$Platform") {
    'dual-gemm-nangate45' { 'ASIC/openroad/dual_gemm_nangate45/config.mk' }
    'dual-gemm-asap7' { 'ASIC/openroad/dual_gemm_asap7/config.mk' }
    'full-nangate45' { 'ASIC/openroad/bcuda_transformer_nangate45/config.mk' }
    'full-asap7' { 'ASIC/openroad/bcuda_transformer_asap7/config.mk' }
}

if (-not (Test-Path -LiteralPath $flowDir)) {
    throw "OpenROAD Flow Scripts not found at $flowDir"
}

$makeTarget = switch ($Target) {
    'all' { 'all' }
    'finish' { 'finish' }
    default { $Target }
}

if ($Engine -eq 'docker') {
    $configPath = "/workspace/$configRelative"
    $dockerArgs = @(
        'run', '--rm',
        '--mount', "type=bind,source=$projectDir,target=/workspace",
        '--mount', "type=bind,source=$flowDir,target=/OpenROAD-flow-scripts/flow",
        'openroad/orfs:latest',
        'bash', '-lc'
    )

    if ($Clean) {
        & docker @dockerArgs "cd /OpenROAD-flow-scripts/flow && make DESIGN_CONFIG=$configPath clean_all"
        if ($LASTEXITCODE -ne 0) { throw "OpenROAD clean failed with exit code $LASTEXITCODE" }
    }

    & docker @dockerArgs "cd /OpenROAD-flow-scripts/flow && make DESIGN_CONFIG=$configPath $makeTarget"
    if ($LASTEXITCODE -ne 0) { throw "OpenROAD $makeTarget failed with exit code $LASTEXITCODE" }
    exit 0
}

function ConvertTo-WslPath([string]$Path) {
    $resolved = (Resolve-Path -LiteralPath $Path).Path
    if ($resolved -notmatch '^([A-Za-z]):\\(.*)$') {
        throw "Cannot convert path to WSL: $resolved"
    }
    return "/mnt/$($Matches[1].ToLower())/$($Matches[2].Replace('\', '/'))"
}

$wslProject = ConvertTo-WslPath $projectDir
$wslFlow = ConvertTo-WslPath $flowDir
$wslConfig = "$wslProject/$configRelative"
$openroadExe = "$wslProject/ASIC/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad"
$yosysExe = "$wslProject/ASIC/OpenROAD-flow-scripts/dependencies/bin/yosys"
$dependencyRoot = "$wslProject/ASIC/OpenROAD-flow-scripts/dependencies"
$makeVariables = "DESIGN_CONFIG='$wslConfig' BCUDA_PROJECT_HOME='$wslProject' OPENROAD_EXE='$openroadExe' YOSYS_EXE='$yosysExe'"
$wslEnvironment = "export PATH='$dependencyRoot/bin':`$PATH && export LD_LIBRARY_PATH='$dependencyRoot/lib:$dependencyRoot/lib64:$dependencyRoot/or-tools/lib':`${LD_LIBRARY_PATH:-}"

if ($Clean) {
    & wsl -d Ubuntu -e bash -lc "$wslEnvironment && cd '$wslFlow' && make $makeVariables clean_all"
    if ($LASTEXITCODE -ne 0) { throw "OpenROAD clean failed with exit code $LASTEXITCODE" }
}

& wsl -d Ubuntu -e bash -lc "$wslEnvironment && cd '$wslFlow' && make $makeVariables $makeTarget"
if ($LASTEXITCODE -ne 0) { throw "OpenROAD $makeTarget failed with exit code $LASTEXITCODE" }
