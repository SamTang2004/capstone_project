# DeiT-Small aggressive-quantization results

## Held-out result

All numbers below were independently reproduced by reloading the selected
checkpoint and scoring the same untouched 178-image test split.

| Model | Validation accuracy | Test accuracy | Test macro F1 | Estimated packed parameter size |
|---|---:|---:|---:|---:|
| Float fine-tuned | 100.00% | 100.00% (178/178) | 100.00% | 82.66 MiB FP32 |
| W8A8 QAT | 100.00% | 100.00% (178/178) | 100.00% | 20.66 MiB |
| W4A4 QAT | 53.33% | 60.67% (108/178) | 59.57% | 10.33 MiB |
| Scaled E2M1 FP4 QAT | 72.22% | 79.78% (142/178) | 79.59% | 10.33 MiB |

Packed-size estimates exclude scale metadata.

## Does a larger model save four-bit quantization?

| Four-bit format | DeiT-Tiny test accuracy | DeiT-Small test accuracy | Change |
|---|---:|---:|---:|
| W4A4 integer | 65.17% | 60.67% | -4.50 points |
| Scaled E2M1 FP4 | 73.03% | 79.78% | +6.74 points |

Not by itself. Extra capacity made integer W4A4 slightly worse under this
recipe. It helped FP4 substantially, but FP4 remains 20.22 points behind INT8.
The evidence favors mixed precision or better scaling over simply increasing
model width.

## FP4 class behavior

| Class | Correct / test | Accuracy |
|---|---:|---:|
| Fresh apple | 19 / 30 | 63.33% |
| Rotten apple | 19 / 30 | 63.33% |
| Fresh banana | 30 / 30 | 100.00% |
| Rotten banana | 27 / 30 | 90.00% |
| Fresh orange | 23 / 28 | 82.14% |
| Rotten orange | 24 / 30 | 80.00% |

The larger FP4 model strongly improves both orange classes relative to Tiny,
while apple quality remains the main weakness.

## PYNQ-Z2 implication

Canonical DeiT-Small is commonly reported at about 4.6 GMAC per 224x224 image,
roughly 3.6 times Tiny. If one multiply and one add are counted as two
operations, a 500 ms target requires about 18.4 sustained GOPS for the matrix
work alone. Against the current 51.2-GOPS theoretical accelerator peak, that is
about 36% sustained utilization before data movement, softmax, LayerNorm,
GELU, residuals, and ARM-side overhead. Small is therefore a materially harder
fit than Tiny and its FP4 gain does not yet justify selecting it for the board.

## Conclusion and next experiment

Keep DeiT-Tiny as the deployment baseline. The subsequent Tiny W4A8 experiment
reached 99.44% test accuracy, confirming that eight-bit activations recover the
full-INT4 failure without paying Small's 3.92-times parameter and roughly
3.6-times compute cost. W4A8 is therefore the current deployment candidate.

The same limitations as the Tiny experiment apply: this is one small published
dataset, split grouping is heuristic, perfect INT8 accuracy may reflect dataset
capture conditions, and hardware numerical equivalence is not yet established.
