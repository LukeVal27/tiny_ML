# EE 446 TinyML — Lab 3: DNN Compression on UCI HAR

Three completed notebooks compressing a DNN activity classifier (561 features → 6 classes), each executed top-to-bottom with all outputs included.

## Notebooks

| Notebook | Techniques | Generated models |
|---|---|---|
| `EE446_Lab3_UCI_HAR_DNN_Quantization_Student_TODO.ipynb` | Post-training quantization (FP32 / dynamic-range / float16 / INT8) + quantization-aware training | `baseline_*.tflite`, `qat_int8.tflite` |
| `EE446_UCI_HAR_DNN_Pruning_Quantization_Student_TODO.ipynb` | Magnitude pruning (20%→85%), `strip_pruning`, sparsity-aware conversion, + float16 | `pq_baseline_fp32.tflite`, `pruned_with_mask_*.tflite`, `stripped_sparse_*.tflite` |
| `EE446_UCI_HAR_DNN_Knowledge_Distillation_Pruning_Quantization_Student_TODO.ipynb` | Knowledge distillation (T=4.0, α=0.3) → pruning → full INT8 | `distilled_*.tflite`, `pruned_distilled_with_mask_fp32.tflite` |

## Key results

| Model | Accuracy | Size (KB) |
|---|---|---|
| Baseline FP32 TFLite | 0.9386 | 726.7 |
| PTQ INT8 | 0.9376 | 195.9 |
| QAT INT8 | 0.9379 | 186.0 |
| Pruned (wrappers attached!) | 0.9376 | 1454.2 |
| Stripped sparse + float16 | 0.9376 | 108.9 |
| Distilled student (38k params) | 0.9342 | 151.5 |
| **Distilled + pruned + INT8** | **0.9270** | **58.3** |

Final deployable model: `distilled_stripped_sparse_int8.tflite` — 12.5× smaller than the FP32 baseline for a 1.2 pp accuracy cost, all-integer arithmetic.

## Environment note

Notebooks were run on TensorFlow 2.21 / TF-MOT 0.8.1. `TF_USE_LEGACY_KERAS=1` is set in each notebook's import cell before TensorFlow is imported — required for TF-MOT (QAT/pruning) on TF ≥ 2.16, harmless on the course-pinned TF 2.14/2.15.
