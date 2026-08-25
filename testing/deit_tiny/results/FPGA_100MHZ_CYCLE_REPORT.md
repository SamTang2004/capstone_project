# Margin-qualified 100 MHz cycle report

The PYNQ-Z2 used the W4A8 DeiT-Tiny image implemented at 100 MHz with
0.300 ns additional setup uncertainty. Post-route timing passed with
WNS `+0.091 ns` and WHS `+0.016 ns`. W8A8/W4A8 graph smokes and both
HeatViT selector modes passed before the full-model measurements.

## End-to-end results

| Model | Repeat PL cycles | Mean cycles | Mean PL time | Last host graph time |
|---|---:|---:|---:|---:|
| DeiT-Tiny W4A8 | 110,025,523; 109,968,211; 111,636,763 | 110,543,499 | 1.105435 s | 1.240142 s |
| Balanced HeatViT W4A8 | 75,559,439; 75,499,151; 75,503,399 | 75,520,663 | 0.755207 s | 0.850360 s |

HeatViT saves `35,022,836` mean cycles (`31.682%`). The current HeatViT
PL window remains `25,520,663` cycles above the 50,000,000-cycle / 500 ms
target, requiring a further `33.793%` reduction.

Relative to the 76.923 MHz measurements, the write-request timing pipeline
raises operation-window cycles by about 6%, but the higher clock still reduces
measured PL time by approximately 18% for both models.

## Last-run module breakdown

| Module | Baseline cycles | HeatViT cycles | HeatViT share |
|---|---:|---:|---:|
| GEMM | 45,441,353 | 31,329,638 | 41.494% |
| Softmax | 28,757,223 | 16,992,669 | 22.506% |
| LayerNorm | 19,252,769 | 13,867,507 | 18.367% |
| GELU | 7,469,349 | 5,457,941 | 7.229% |
| Residual add | 5,017,706 | 3,704,137 | 4.906% |
| Token concat | 3,435,199 | 2,512,940 | 3.328% |
| Token embedding | 2,263,164 | 1,073,008 | 1.421% |
| Token prune | - | 565,559 | 0.749% |

## Last-run GEMM phases

| Phase | Baseline cycles | HeatViT cycles |
|---|---:|---:|
| Load A | 6,584,342 | 4,677,761 |
| Load B | 16,723,931 | 11,185,012 |
| Compute | 10,599,617 | 7,585,688 |
| Write | 13,699,388 | 9,287,762 |

The next optimization priority remains GEMM, particularly Load B and writeback.
Softmax is the second-largest total consumer, while the token-pruner overhead
itself is below 1%.

Raw reports:

- `fpga_cycle_report_100mhz_baseline.json`
- `fpga_cycle_report_100mhz_heatvit_balanced.json`
