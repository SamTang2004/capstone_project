"""Canonical paths shared by the fruit-model training and export scripts."""

from __future__ import annotations

import os
from pathlib import Path


TRAIN_MODEL_DIR = Path(__file__).resolve().parent
FPGA_DIR = TRAIN_MODEL_DIR.parent

TRAINING_DATA_DIR = Path(
    os.environ.get(
        "BCUDA_TRAINING_DATA_DIR",
        FPGA_DIR / "dataset" / "fruits-360-100x100" / "Training",
    )
)
TEST_DATA_DIR = Path(
    os.environ.get(
        "BCUDA_TEST_DATA_DIR",
        FPGA_DIR / "dataset" / "fruits-360-100x100" / "Test",
    )
)

MODELS_DIR = TRAIN_MODEL_DIR / "models"
FLOAT_MODEL_DIR = MODELS_DIR / "fruit3_float"
FLOAT_CHECKPOINT = FLOAT_MODEL_DIR / "best.pt"
QAT_MODEL_DIR = MODELS_DIR / "fruit3_pytorch_qat_int8"
QAT_CHECKPOINT = QAT_MODEL_DIR / "best_qat_fakequant.pt"
INT8_STATE = QAT_MODEL_DIR / "best_int8_state.pt"
INT8_FULL_MODEL = QAT_MODEL_DIR / "best_int8_full_model.pt"
QUANT_EXPORT_DIR = QAT_MODEL_DIR
HLS_EXPORT_DIR = TRAIN_MODEL_DIR / "hls_fruit_cnn"

INT4_MODEL_DIR = MODELS_DIR / "fruit3_budgetcuda_int4"
INT4_QAT_CHECKPOINT = INT4_MODEL_DIR / "best_int4_qat_fakequant.pt"
INT4_ONNX = INT4_MODEL_DIR / "fruit3_budgetcuda_int4.onnx"
