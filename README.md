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

**Authorship:** All Lab 5 work — running every notebook cell, deploying the model to the Arduino Nano 33 BLE Sense, and writing the answer document — was completed by Luke Valerio. Claude (an AI assistant) was used only to commit these files to the repository.

## Assignment 1

EE 446 Homework 1 — DNN and Wine classification with model-compression techniques (quantization, pruning, knowledge distillation).

| File | Description |
|------|-------------|
| [`Assignment1/Luke_Valerio_EE446_HW1_Pro.ipynb`](Assignment1/Luke_Valerio_EE446_HW1_Pro.ipynb) | Completed [Pro] programming notebook for Problem 1: baseline DNN, dynamic-range/INT8/float16 quantization, pruning, output-based knowledge distillation, and a combined KD+INT8 model, each reporting TFLite size and classification performance. |
| [`Assignment1/Luke_Valerio_EE446_HW1_Dis.pdf`](Assignment1/Luke_Valerio_EE446_HW1_Dis.pdf) | Answers to the [Dis] discussion questions: Problem 1(e) analysis and all four Problem 2 (Edge Impulse) questions. |

## Assignment 2

EE 446 Homework 2 — network anomaly detection on the NSL-KDD style dataset (125,973 samples, 42 features), from preprocessing through full-integer INT8 quantization and Nano 33 BLE deployment.

| File | Description |
|------|-------------|
| [`Assignment2/Luke_Valerio_EE446_HW2_Pro.ipynb`](Assignment2/Luke_Valerio_EE446_HW2_Pro.ipynb) | Completed [Pro] notebook for Problems 1–4: column dropping and binary label collapse, LabelEncoder on the categorical columns, t-SNE / PCA / KernelPCA 2-D visualizations, a 38→64→32→16→1 DNN, and full-integer INT8 post-training quantization with a representative dataset. |
| [`Assignment2/Luke_Valerio_EE446_HW2_Dis.pdf`](Assignment2/Luke_Valerio_EE446_HW2_Dis.pdf) | Results writeup: classification reports and confusion matrices for 3(c) and 4(b), plus the Serial Monitor evidence for 5(c) and 5(d). |
| [`Assignment2/network_data`](Assignment2/network_data) | Problem 5(c) Arduino sketch — five test samples, INT8 inference on the Nano 33 BLE. |
| [`Assignment2/network_data_10`](Assignment2/network_data_10) | Problem 5(d) Arduino sketch — the following ten test samples. |
| [`Assignment2/models`](Assignment2/models) | The float32 Keras model, the INT8 TFLite model, and the generated `network_model.h` C array. |

The float32 model reaches 0.9985 test accuracy. Full INT8 quantization drops that to 0.9984 — three additional misclassifications out of 25,195 — while shrinking the model from 100.01 KB to 8.24 KB, a 12.1× reduction. Both sketches compile to 13% of flash and 29% of RAM on the Nano 33 BLE.

**Authorship:** Claude (an AI assistant) wrote the notebook solution code and the Arduino `Serial.print`/dequantization blocks, executed the notebook, generated the quantized model and C arrays, verified both sketches compile, and drafted the writeup. Luke Valerio directed the work, made the design decisions, and reviewed the results. Flashing the board and capturing the Serial Monitor screenshots was done by Luke Valerio.

## Lab 7

On-device data collection and tiny ensemble learning. Part I classifies an idle state against a lifting motion in Edge Impulse. Part II builds a three-branch autoencoder ensemble over the mHealth IMU dataset, combines the branches with a stacked meta-classifier, and compresses everything with pruning, quantization aware training, and full int8 conversion.

Public Edge Impulse project: https://studio.edgeimpulse.com/public/1084899/live

| File | Description |
|------|-------------|
| [`Lab7_TinyML_Ensemble_Learning/Lab7_Report.pdf`](Lab7_TinyML_Ensemble_Learning/Lab7_Report.pdf) | Full report covering both parts, with the Part I serial evidence, the Part II results, and the four discussion questions including the ensemble architecture diagram. |
| [`Lab7_TinyML_Ensemble_Learning/TinyML_Lab7_Part_II.ipynb`](Lab7_TinyML_Ensemble_Learning/TinyML_Lab7_Part_II.ipynb) | Completed notebook: windowing, three autoencoder branches, latent-space t-SNE, branch classifiers, stacked meta-classifier, pruning with a polynomial schedule, quantization aware training with mask enforcement, and int8 TFLite export. |
| [`Lab7_TinyML_Ensemble_Learning/Tiny_Ensemble_Learning`](Lab7_TinyML_Ensemble_Learning/Tiny_Ensemble_Learning) | Arduino sketch that runs all four int8 models on the Nano 33 BLE Sense, with the models included as C arrays. |
| [`Lab7_TinyML_Ensemble_Learning/models`](Lab7_TinyML_Ensemble_Learning/models) | The four pruned, QAT, int8 TFLite models. |

Ensemble accuracy is 1.00 on the held-out test set in float32. After pruning to 79.44% sparsity and converting to int8, the branches score 0.9901, 0.9322, and 0.9220, and the meta-classifier scores 0.8948. Each branch shrinks from 165.29 KB to 44.55 KB.

**Authorship:** The Edge Impulse project, data collection, model training, Arduino deployment, and the Part I serial evidence were done by Luke Valerio. Claude (an AI assistant) prepared the Python environment, executed the Part II notebook, generated the compressed models and C arrays, verified the sketch compiles, and drafted the report text, which was then reviewed by Luke Valerio.
