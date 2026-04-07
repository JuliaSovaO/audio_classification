# Audio Classification

**Authors**: Oksana Dziuba, Yuliia Sova, Yelizaveta Piletska

This project implements a spoken word classification system using fundamental linear algebra operations written from scratch: manual Singular Value Decomposition (SVD) via power iteration with deflation, and manual k-Nearest Neighbors (kNN) using the expanded Euclidean distance formula.

## Why This Project Matters

Modern keyword spotting systems often rely on opaque neural networks. This project demonstrates that competitive classification can be built entirely from first principles using vectors, matrices, inner products, SVD, and Euclidean distance. Every linear algebra operation is implemented manually, providing complete mathematical transparency.

## Algorithms and Pipeline

**Core components implemented from scratch:**

| Component | Method | Location |
|-----------|--------|----------|
| SVD | Power iteration with deflation | `svd.py` |
| kNN | Expanded Euclidean distance | `classifier.py` |
| Metrics | Accuracy, precision, recall, F1 | `metrics.py` |

**External library used (non-core):**
- `librosa` for MFCC feature extraction

**Pipeline steps:**

1. Load audio, resample to 16 kHz, pad/truncate to 16,000 samples: $\mathbf{a} \in \mathbb{R}^{16000}$

2. Extract MFCC features (40 coefficients $\times$ 100 frames) using librosa, flatten: $\mathbf{x} \in \mathbb{R}^{4000}$

3. Normalize to zero mean and unit variance: $\mathbf{x}_{\text{norm}} = (\mathbf{x} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$

4. Organize $m$ training vectors as rows: $X \in \mathbb{R}^{m \times 4000}$

5. Compute $A = X^T X \in \mathbb{R}^{4000 \times 4000}$

6. For $i = 1$ to $k$ ($k=25$):
   - Power iteration: $\mathbf{v}_{\text{new}} = A\mathbf{v} / \|A\mathbf{v}\|$, $\lambda = \mathbf{v}^T A \mathbf{v}$
   - Deflation: $A \leftarrow A - \lambda_i \mathbf{v}_i \mathbf{v}_i^T$

7. Construct $V_k = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$, project: $X_{\text{reduced}} = X V_k$

8. For test sample $\mathbf{z} \in \mathbb{R}^{4000}$:
   - Normalize: $\mathbf{z}_{\text{norm}} = (\mathbf{z} - \boldsymbol{\mu}) / \boldsymbol{\sigma}$
   - Project: $\mathbf{z}_{\text{reduced}} = \mathbf{z}_{\text{norm}}^T V_k$
   - Compute distances: $d_i = \sqrt{\|\mathbf{z}_{\text{reduced}}\|^2 + \|X_{\text{reduced}}[i]\|^2 - 2 \mathbf{z}_{\text{reduced}}^T X_{\text{reduced}}[i]}$

9. Select $k=5$ nearest neighbors, predict by majority vote

## Dataset

**Speech Commands dataset** (Warden, 2018):
- 105,829 one-second utterances, 16 kHz
- 12 classes: 10 target words ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go") + silence + unknown
- Splits: Train (85,511), Validation (10,102), Test (4,890)

Run `sort_data.py` to organize the dataset into `data/train/`, `data/validation/`, `data/test/`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# download dataset
wget http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz
mkdir -p data
tar -xzf speech_commands_v0.02.tar.gz -C data/
python sort_data.py
```

**Dependencies:** numpy, librosa, soundfile, scipy

## Usage

**Train and evaluate:**
```bash
python main.py --train_dir data/train --test_dir data/test
```

**Predict a single audio file:**
```bash
python main.py --predict path/to/audio.wav
```

Features are cached as `X_train.npy` and `y_train.npy` for faster repeated runs.

**example**:
```bash
$ python main.py --predict data/stop/0a2b400e_nohash_0.wav

--- Phase 1: Feature Extraction ---
Loading cached features from X_train.npy...

--- Phase 2: Dimensionality Reduction (SVD) ---
Starting SVD computation. Features dimension: 4000
Extracted 5/25 components...
Extracted 10/25 components...
Extracted 15/25 components...
Extracted 20/25 components...
Extracted 25/25 components...
Manual SVD completed in 69.16 seconds.

--- Phase 3: Assembly & Model Fitting ---

Predicting single file: data/stop/0a2b400e_nohash_0.wav
Predicted 1/1 samples...
kNN prediction completed. Average prediction time: 0.0028 seconds per sample.

Predicted Class: stop
```

## Performance Results

| Metric | Value |
|--------|-------|
| Original dimension | 4,000 |
| Reduced dimension (k) | 25 |
| Variance preserved | 95% |
| SVD computation time | 161.92 sec |
| Test accuracy | 60.93% |
| Weighted precision | 0.5448 |
| Weighted recall | 0.6093 |
| Weighted F1-score | 0.5643 |
| Prediction time | 0.0031 sec/sample |

The 60.93% accuracy (vs. 91% SVM benchmark) is due to class imbalance in the dataset, not algorithmic failure. The "unknown" class dominates training, and kNN's proximity-based voting is sensitive to dense class regions.

## References

- Warden, P. "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition." arXiv:1804.03209, 2018.
- Strang, G. *Introduction to Linear Algebra*, 5th Edition, Wellesley-Cambridge Press, 2016.

## Videos

- [Yuliia](https://youtu.be/EHGFcpIewvc?feature=shared)
- [Oksana](https://youtu.be/Tn6ZER4yozE?feature=shared)
- [Yelizaveta](https://youtu.be/hWKRWPKy5N4?si=tC5132e1yr7Dxpcb)
