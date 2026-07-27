from __future__ import annotations

from pathlib import Path

from .dtypes import Layout, normalize_layout
from .quantization import Int4QuantizationParams, QuantizationParams


def load_quantized_image(
    source,
    input_shape,
    quantization: QuantizationParams | Int4QuantizationParams,
    *,
    layout: str | int | Layout = Layout.NCHW,
    mean=None,
    std=None,
):
    """Load one image and return a batched quantized BudgetCUDA tensor."""

    try:
        import numpy as np
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Image preprocessing requires numpy and Pillow. "
            "Install them with: python -m pip install numpy Pillow"
        ) from exc

    shape = tuple(int(dimension) for dimension in input_shape)
    layout = normalize_layout(layout)
    if len(shape) != 4 or shape[0] != 1 or layout not in (Layout.NCHW, Layout.NHWC):
        raise ValueError("Image input must be a batch-one rank-4 NCHW or NHWC tensor")
    if layout == Layout.NCHW:
        _, channels, height, width = shape
    else:
        _, height, width, channels = shape
    if channels not in (1, 3) or height <= 0 or width <= 0:
        raise ValueError("Image input must have one or three channels and positive dimensions")
    if not isinstance(quantization, (QuantizationParams, Int4QuantizationParams)):
        raise TypeError("quantization must be an INT8 or INT4 quantization parameter object")

    resampling = getattr(Image, "Resampling", Image)
    with Image.open(Path(source)) as image:
        image = image.convert("L" if channels == 1 else "RGB")
        image = image.resize((width, height), resample=resampling.BILINEAR)
        values = np.asarray(image, dtype=np.float32) / np.float32(255.0)

    if channels == 1:
        values = values[..., None]
    mean_values = _channel_values(mean, channels, 0.0, "mean", np)
    std_values = _channel_values(std, channels, 1.0, "std", np)
    if np.any(std_values == 0.0):
        raise ValueError("Image normalization std values must be nonzero")
    values = (values - mean_values) / std_values
    if layout == Layout.NCHW:
        values = np.transpose(values, (2, 0, 1))
    values = values[None, ...]
    if values.shape != shape:
        raise RuntimeError(f"Preprocessed image shape {values.shape} does not match {shape}")
    return quantization.quantize(values)


def _channel_values(values, channels: int, default: float, label: str, np):
    if values is None:
        values = [default] * channels
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.size != channels:
        raise ValueError(f"Image normalization {label} must contain {channels} values")
    return array.reshape(1, 1, channels)
