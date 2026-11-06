# Traffic Sign Recognition (GTSRB)

> This project was originally developed as a university project in the **Master's program Informatik (Softwareentwicklung)**.
> It was later restructured and improved with the help of **Claude** (Anthropic's AI assistant).

A convolutional neural network (CNN) that classifies German traffic signs into **43 classes**, built with TensorFlow/Keras and trained on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign).

## Results

| Version | Test accuracy (12,630 images) | Errors |
|---|---|---|
| Original university version | 94.95 % | ~638 |
| Improved single CNN (mean of 3 seeds) | 98.99 ± 0.02 % | ~128 |
| **Final: ensemble of 2 CNNs** (mean of 3 seeds) | **99.49 ± 0.09 %** | **~64** |

The number of test errors was reduced by about **90 %** compared to the original version.

## What was improved

| Area | Original version | Improved version |
|---|---|---|
| Preprocessing | Whole image, 30×30 px | Cropped to the sign using the ROI coordinates from the dataset, 32×32 px |
| Validation split | Random 80/20 split | Track-based split (`StratifiedGroupKFold`): near-identical frames of the same physical sign never appear in both train and validation (prevents data leakage) |
| Model | 3 Conv layers + Dense | 3 Conv blocks (2× Conv + BatchNorm each), Dropout, ~824k parameters |
| Augmentation | `ImageDataGenerator` | Keras augmentation layers inside the model (rotation, zoom, translation, contrast; no flips, since flipping changes the meaning of some signs) |
| Training | Fixed 15 epochs | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| Evaluation | Accuracy only | Classification report, confusion matrix, error analysis, comparison over 3 random seeds |

## Experiments

All experiments were run with 3 random seeds (42, 7, 123); values are the mean test accuracy.

| Method | Test accuracy | Recall of the weakest class ("Snow/Ice") |
|---|---|---|
| Baseline (32 px) | 98.99 ± 0.02 % | 85.8 % |
| A: Luminance histogram equalization (32 px) | 98.98 ± 0.18 % | 98.4 % |
| C: Higher resolution (48 px) | 98.85 ± 0.17 % | 91.6 % |
| Ensemble: Baseline + Baseline (control) | 99.17 ± 0.08 % | 84.0 % |
| **Ensemble: Baseline + A** | **99.49 ± 0.09 %** | 97.3 % |
| Ensemble: A + C | 99.41 ± 0.06 % | 99.1 % |
| Ensemble: Baseline + A + C | 99.46 ± 0.05 % | 94.4 % |

Full table: [`results_summary.csv`](results_summary.csv)

**Key findings**

- **Error analysis:** most remaining errors come from a few physical signs with extreme lighting (very dark or overexposed) and from small symbols that become hard to read at 32×32 px (e.g. "Snow/Ice" vs. "Bicycles").
- **Ensembling works through diversity:** combining two models with *different preprocessing* (Baseline + A) clearly beats combining two identical models trained with different seeds (99.49 % vs. 99.17 %). Adding a third model brought no further gain.
- **Targeted preprocessing helps weak classes:** equalization raised the "Snow/Ice" recall from 85.8 % to 98.4 % and 48 px raised it to 91.6 %, although neither improved the overall accuracy on its own.
- **Results vary between runs:** e.g. model A ranged from 98.79 % to 99.14 % across seeds, and re-running the complete notebook shifted mean results by up to ~0.3 percentage points, because some GPU operations are non-deterministic. Small differences between single runs are therefore not meaningful.

## Limitations

- Overexposed images where the color information is lost cannot be fixed by preprocessing.
- The validation accuracy is very high (99.5–99.8 %) and barely separates the methods, so the final ensemble was selected using the test set. The reported test accuracy may therefore be slightly optimistic.
- Results are based on 3 seeds per method.

## Tech stack

Python 3.13 · TensorFlow 2.20 / Keras 3 · scikit-learn · pandas · NumPy · Pillow · Matplotlib · kagglehub

## How to run

```bash
pip install -r requirements.txt
jupyter lab Traffic_Sign_Recognition.ipynb
```

The dataset is downloaded automatically with `kagglehub` on the first run.

**GPU note:** In the environment used for this project, TensorFlow on GPU additionally required the CUDA compiler package, otherwise training failed with `libdevice not found`:

```bash
pip install "nvidia-cuda-nvcc-cu12==12.8.*"
```
