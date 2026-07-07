import argparse

from paths import (
    FLOAT_CHECKPOINT,
    HLS_EXPORT_DIR,
    INT8_STATE,
    QAT_CHECKPOINT,
    QAT_MODEL_DIR,
    TRAINING_DATA_DIR,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show the canonical training/export paths for the FPGA classifier."
    )
    parser.add_argument(
        "--show-paths",
        action="store_true",
        help="Print the paths used by train_1.py, qat_model.py, and export_qat_quant_params.py.",
    )
    args = parser.parse_args()

    if not args.show_paths:
        parser.print_help()
        return

    print("Dataset training directory:", TRAINING_DATA_DIR)
    print("Float checkpoint:", FLOAT_CHECKPOINT)
    print("QAT output directory:", QAT_MODEL_DIR)
    print("QAT checkpoint:", QAT_CHECKPOINT)
    print("INT8 state checkpoint:", INT8_STATE)
    print("HLS export directory:", HLS_EXPORT_DIR)


if __name__ == "__main__":
    main()
