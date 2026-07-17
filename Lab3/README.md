# EE 446 TinyML Lab 3: Neural Network Compression on UCI HAR

Three completed notebooks that compress a dense neural network activity classifier (561 features into 6 classes). Each one was run top to bottom with all of the outputs included.

## Notebooks

| Notebook | Techniques | Generated models |
|---|---|---|
| `EE446_Lab3_UCI_HAR_DNN_Quantization_Student_TODO.ipynb` | Post training quantization (float32, dynamic range, float16, int8) and quantization aware training | `baseline_*.tflite`, `qat_int8.tflite` |
| `EE446_UCI_HAR_DNN_Pruning_Quantization_Student_TODO.ipynb` | Magnitude pruning (20% up to 85%), `strip_pruning`, sparsity aware conversion, and float16 | `pq_baseline_fp32.tflite`, `pruned_with_mask_*.tflite`, `stripped_sparse_*.tflite` |
| `EE446_UCI_HAR_DNN_Knowledge_Distillation_Pruning_Quantization_Student_TODO.ipynb` | Knowledge distillation (temperature 4.0, alpha 0.3), then pruning, then full int8 | `distilled_*.tflite`, `pruned_distilled_with_mask_fp32.tflite` |

## Key results

| Model | Accuracy | Size (KB) |
|---|---|---|
| Baseline float32 TensorFlow Lite | 0.9386 | 726.7 |
| Post training int8 | 0.9376 | 195.9 |
| Quantization aware training int8 | 0.9379 | 186.0 |
| Pruned (wrappers still attached) | 0.9376 | 1454.2 |
| Stripped sparse and float16 | 0.9376 | 108.9 |
| Distilled student (38k params) | 0.9342 | 151.5 |
| **Distilled, pruned, and int8** | **0.9270** | **58.3** |

The final deployable model is `distilled_stripped_sparse_int8.tflite`. It is 12.5x smaller than the float32 baseline for a 1.2 percentage point accuracy cost, and it runs with all integer arithmetic.

## Environment note

The notebooks were run on TensorFlow 2.21 and TensorFlow Model Optimization 0.8.1. Each notebook sets `TF_USE_LEGACY_KERAS=1` in its import cell before TensorFlow is imported. This is required for the model optimization toolkit (quantization aware training and pruning) on TensorFlow 2.16 and newer, and it is harmless on the course pinned TensorFlow 2.14 or 2.15.
