from pathlib import Path


TRAIN_MODEL_DIR = Path(__file__).resolve().parent
FPGA_DIR = TRAIN_MODEL_DIR.parent
PROJECT_ROOT = FPGA_DIR.parent

DATASET_ROOT = FPGA_DIR / "dataset" / "fruits-360-100x100"
TRAINING_DATA_DIR = DATASET_ROOT / "Training"
TEST_DATA_DIR = DATASET_ROOT / "Test"

MODELS_DIR = TRAIN_MODEL_DIR / "models"
FLOAT_MODEL_DIR = MODELS_DIR / "fruit3_float"
QAT_MODEL_DIR = MODELS_DIR / "fruit3_qat_64_c12_onednn"
HLS_EXPORT_DIR = TRAIN_MODEL_DIR / "hls_fruit_cnn"

FLOAT_CHECKPOINT = FLOAT_MODEL_DIR / "best.pt"
QAT_CHECKPOINT = QAT_MODEL_DIR / "best_qat_fakequant.pt"
INT8_STATE = QAT_MODEL_DIR / "best_int8_state.pt"
INT8_FULL_MODEL = QAT_MODEL_DIR / "best_int8_full_model.pt"
QUANT_EXPORT_DIR = QAT_MODEL_DIR / "hls_quant_export"
