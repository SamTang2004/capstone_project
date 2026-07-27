$ErrorActionPreference = "Stop"

$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Missing $python. Create the host environment before running the demo."
}

$env:PYTHONPATH = Join-Path $repo "FPGA"
$model = Join-Path $repo "FPGA\demo\models\fruit3_budgetcuda_int8.onnx"
$dataset = Join-Path $repo "FPGA\demo\dataset\fruits-360-100x100"

& $python "$repo\FPGA\tests\hardware\pynq_full_cnn_inference.py" `
    --model $model --address 192.168.2.99 --port 9000 --timeout 300
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $python "$repo\FPGA\tests\hardware\pynq_fruit_accuracy.py" `
    --model $model --dataset-root $dataset --address 192.168.2.99 `
    --port 9000 --timeout 300 --skip-train --board-per-class 100 --seed 2026
exit $LASTEXITCODE
