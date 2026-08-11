# EE 446 TinyML Lab 7: On-Device Data Collection and Tiny Ensemble Learning

Part I is motion classification between an idle state and a lifting motion, built in Edge Impulse and deployed to the Arduino Nano 33 BLE Sense. Part II is an ensemble of three autoencoder and classifier branches over the mHealth IMU dataset, combined by a stacked meta-classifier and compressed with pruning, quantization aware training, and full int8 conversion.

## Part I: Edge Impulse

Public project: https://studio.edgeimpulse.com/public/1084899/live

Two classes (`idle` and `lift`), 20 samples per class at 10 seconds each, sampled at 100 Hz. The impulse uses a Spectral Analysis processing block into a Keras classifier.

| Metric | Value |
|---|---|
| Validation accuracy | 96.7% |
| Test accuracy | 91.0% |
| Estimated on-device latency | 32 ms |
| Estimated peak RAM | 3.1 KB |

Measured on the board, DSP takes 33 to 39 ms per window while classifier inference takes 485 to 542 microseconds, so feature extraction accounts for roughly 98 percent of the per-window cost.

## Part II: Tiny ensemble learning

The notebook builds three branches over three representations of the same IMU signal (raw, standard scaled, and min-max scaled). Each branch trains an autoencoder, reuses the encoder as a feature extractor, and trains a classifier on the 32-dimensional latent vector. The three softmax outputs concatenate into a 36-dimensional vector that feeds a stacked meta-classifier.

Dataset is `mHealth_subject6.log`, giving 32,205 labeled samples across 12 activity classes, windowed into 21,350 training and 8,479 test windows of 100 timesteps by 6 channels.

### Architecture

```
input 600  ->  Dense(64, ReLU) -> Dense(32, linear)   [encoder]
           ->  Dense(20, ReLU) -> Dense(12, softmax)  [branch classifier]

3 x 12 softmax  ->  concat 36  ->  Dense(24, ReLU) -> Dense(12, softmax)
```

### Results

| Model | Float32 accuracy | int8 accuracy | Sparsity | int8 size (KB) |
|---|---|---|---|---|
| Raw branch | 0.99 | 0.9220 | 79.44% | 44.55 |
| Standard-scaled branch | 0.99 | 0.9901 | 79.44% | 44.55 |
| Min-max-scaled branch | 0.97 | 0.9322 | 79.44% | 44.61 |
| Stacked meta-classifier | 1.00 | 0.8948 | 79.43% | 3.63 |

Each branch drops from 165.29 KB in float32 to 44.55 KB in int8, a reduction of 3.7 times. The saving comes from quantization rather than pruning, because TFLite stores dense weight tensors in full and a zeroed weight still costs one byte.

The measured sparsity is 79.44% rather than the scheduled 80% because 5 epochs at batch size 256 is 420 steps of a 500 step `PolynomialDecay` schedule, and TF-MOT updates masks every 100 steps, so the final update lands at step 400.

## Files

| Path | Description |
|---|---|
| `Lab7_Report.pdf` | Full report covering both parts, including the four discussion questions and the architecture diagram. |
| `TinyML_Lab7_Part_II.ipynb` | Notebook with all 18 code cells executed and outputs populated. |
| `Tiny_Ensemble_Learning/` | Deployment sketch plus the four int8 models as C arrays. Verified to compile for `arduino:mbed_nano:nano33ble` at 42% flash and 74% RAM. |
| `Raw_IMU_Recorder/` | Sketch that logs raw IMU samples over serial as CSV, for the optional recorded-data validation. |
| `models/` | The four pruned, QAT, int8 TFLite models. |

## Reproducing

The notebook expects `mHealth_subject6.log` in the same folder. That file is course-provided and is not committed here because of its size.

```bash
source ~/ai/projects/tinyml-arduino/bin/activate
jupyter lab
```

Environment is Python 3.11.15, TensorFlow 2.14.1, and TensorFlow Model Optimization 0.8.0. The pinned TensorFlow 2.14 matters, since the model optimization toolkit does not support Keras 3.

To rebuild the sketch:

```bash
arduino-cli compile --fqbn arduino:mbed_nano:nano33ble Tiny_Ensemble_Learning
```

## Deployment result

With the board lying flat and still, the three branches disagree sharply. The raw branch reports jump front and back at 0.9727, the standard-scaled branch reports climbing stairs at 0.7187, and the min-max branch reports lying down at 0.9805. The stacked meta-classifier nonetheless returns sitting and relaxing at 0.5195, placing 0.9609 of its probability mass on the three stationary classes.

The final answer is reasonable for a resting board, but it is reached from branch inputs unlike anything the meta-classifier saw in training, and the confidence of 0.5195 suggests it is closer to undecided than to confident. Predictions are highly stable across windows, varying only in the third decimal place, so the error is systematic rather than noisy.

## Known limitation

The Part II models are trained on one subject's chest and ankle mounted Shimmer sensors. A Nano 33 BLE Sense resting on a desk sits well outside that distribution, which is what the branch disagreement above demonstrates. The report discusses the causes in Questions 3 and 4.

## Authorship

The Edge Impulse project, data collection, model training, Arduino deployment, and the serial-monitor evidence in Part I were done by Luke Valerio. Claude (an AI assistant) prepared the Python environment, executed the Part II notebook, generated the compressed models and C arrays, verified that the sketch compiles, and drafted the report text, which was then reviewed by Luke Valerio.
