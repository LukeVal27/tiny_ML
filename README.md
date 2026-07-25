# tiny_ML

TinyML labs for the Arduino Nano 33 BLE Sense.

## Lab2

Multimodal sensor fusion on the Nano 33 BLE Sense.

| Sketch | Description |
|--------|-------------|
| [`Lab2/Task_10_TinyML`](Lab2/Task_10_TinyML) | **Smart Workspace Situation Classifier** — fuses microphone, ambient light, IMU motion, and proximity into binary flags and a rule-based situation label. |
| [`Lab2/Task_11_TinyML`](Lab2/Task_11_TinyML) | **Environmental Event Monitor** — EMA-baseline change detection over humidity, temperature, magnetometer, and color; warmup + cooldown/debounce logic to latch discrete events. |

## Lab5

Keyword Spotting (KWS) — training/feature-extraction notebook and the Arduino deployment sketch for recognizing the keywords **"yes"** and **"no"** on the Nano 33 BLE Sense.

| File | Description |
|------|-------------|
| [`Lab5/TinyML_Lab5_Local_Env.ipynb`](Lab5/TinyML_Lab5_Local_Env.ipynb) | Completed notebook: audio preprocessing (spectrogram/MFCC), loading the pretrained `tiny_conv` model, full INT8 quantization with a representative dataset, accuracy evaluation, and export of the quantized TFLite model to a C array. |
| [`Lab5/micro_speech`](Lab5/micro_speech) | Arduino sketch deployed to the Nano 33 BLE Sense. `micro_features_model.cpp` holds the quantized KWS model; the sketch prints recognized keywords to the Serial Monitor. |
| [`Lab5/Lab5_KWS_Answers.pdf`](Lab5/Lab5_KWS_Answers.pdf) | Written answers to the lab questions, including the Serial Monitor screenshot showing correct "yes"/"no" detection on the Nano 33 BLE Sense. |
