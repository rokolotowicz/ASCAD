# ⚡ Deep Learning Side-Channel Attack on AES — ASCAD

> **An AI that bypasses physical hardware defenses to extract AES-128 cryptographic keys via power side-channel analysis.**

This project implements a 1D Convolutional Neural Network (CNN) trained on the [ASCAD dataset](https://github.com/rokolotowicz/ASCAD) to recover secret AES-128 keys from raw power consumption traces — without ever touching the ciphertext, and without brute-forcing the keyspace.

---

## 🔍 What Is a Side-Channel Attack?

Modern encryption like AES is mathematically unbreakable by brute force. But the *physical hardware* running it leaks information.

When a CPU executes AES, it passes data through the **S-Box** — a non-linear operation that XORs the plaintext with the secret key, then substitutes the result for a completely different byte. At the silicon level, flipping bits from `0` → `1` requires microscopic bursts of electricity.

| Processed Byte | Hamming Weight | Power Draw |
|---|---|---|
| `0xFF` (all 1s) | 8 | Higher |
| `0x00` (all 0s) | 0 | Lower |
| `0x4B` (mixed) | 4 | Medium |

The device **unintentionally broadcasts** a physical power signature that correlates to the secret key being processed. This is a **power trace** — and it's the attack surface this project exploits.

---

## 🧠 The Neural Network

Rather than classical Differential Power Analysis (DPA), this project trains a deep CNN to learn the trace-to-key relationship end-to-end.

### Final Architecture

The final model (`newbest_ascad_model.keras`) uses SELU activations with LeCun initialization — a self-normalizing configuration that eliminates the need for BatchNormalization and allows the network to go wider:

```
Input: Raw 1D Power Trace  →  shape: (15000, 1)
        │
        ▼
Conv1D(64,  kernel=50, SELU, LeCun) → AveragePooling1D(2)   ← Wide: coarse power envelope
Conv1D(128, kernel=25, SELU, LeCun) → AveragePooling1D(2)   ← Medium: transition patterns  
Conv1D(256, kernel=11, SELU, LeCun) → AveragePooling1D(2)   ← Focused: S-Box leakage moment
        │
        ▼
   GlobalAveragePooling1D       ← Collapses time → (None, 256); prevents OOM vs. Flatten
        │
        ▼
   Dense(2048, SELU) → AlphaDropout(0.3)
   Dense(2048, SELU) → AlphaDropout(0.3)
        │
        ▼
   Dense(256, Softmax, dtype=float32)   ← 256 outputs = one per possible key byte (0x00–0xFF)
```

**Total parameters: ~7.74 MB** (compact enough to run inference on the full attack set in one GPU pass)

**Key design choices:**

- **Decreasing kernel sizes (50 → 25 → 11)** — wide early kernels scan the broad power envelope; narrow late kernels lock onto the precise nanosecond the S-Box operation occurs
- **SELU + LeCun Normal init** — self-normalizing activations keep gradient flow stable through deep layers without explicit BatchNorm; requires no manual tuning of normalization hyperparameters
- **AlphaDropout** — standard Dropout would break SELU's self-normalizing property by destroying the activation mean/variance; AlphaDropout preserves both
- **GlobalAveragePooling1D** — replaces Flatten to prevent GPU OOM; a 15,000-timestep trace flattened after three conv blocks would produce a ~480,000-element vector per sample
- **Mixed Precision (float16 compute / float32 variables)** — halves VRAM usage on RTX 3090, enabling batch_size=256 on the full 7GB+ dataset
- **Output layer forced to float32** — Softmax is numerically unstable at float16; explicitly casting the final layer prevents NaN loss

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam (lr=0.0005) |
| Loss | Categorical Crossentropy |
| Batch Size | 256 |
| Max Epochs | 50 |
| Early Stopping | patience=10, monitor=`val_loss` |
| Best checkpoint | `newbest_ascad_model.keras` (saved on `val_loss` min) |
| Stopped at | Epoch 22 (best weights from epoch 17) |
| Best val_accuracy | 0.55% |

> **Note on accuracy:** 0.55% accuracy on a 256-class problem is ~1.4× better than random chance (0.39%). For a masked AES implementation this is expected — the signal is intentionally buried in noise. The classifier doesn't need to be right on individual traces; it accumulates weak evidence across many traces via log-likelihood.

---

## 🎯 Key Recovery: Rank Evolution Attack

After training, the model attacks fresh traces it has never seen. Success is measured by **Guessing Entropy / Rank Evolution**:

- **Rank 255** = model is clueless (real key is its last guess)
- **Rank 0** = key recovered (real key is the model's #1 guess)

### How It Works

For each attack trace, the model outputs a 256-class probability distribution. Evidence is accumulated across traces using **log-likelihood**:

```python
for each trace i:
    for each of the 256 possible key guesses k:
        sbox_out = AES_SBox[plaintext[i] XOR k]
        prob     = model_prediction[i][sbox_out]
        key_score[k] += log(prob + 1e-40)   # epsilon prevents log(0)

rank = position of true_key in argsort(key_score, descending)
```

The logic: if the model was trained on `SBox[plaintext XOR key]` as its label, then only the correct key hypothesis will consistently map each plaintext to an output the model assigns high probability. All wrong guesses produce incoherent S-Box outputs — their scores average out to noise and stay low.

### Results

| Attack Pool | Final Key Rank | Status |
|---|---|---|
| 2,000 traces | 244 (with repeated convergence to 0) | Noisy convergence |
| 10,000 traces | **32** | Confirmed learning |

**Interpretation:** With 2,000 traces, the rank repeatedly hits 0 before bouncing back — proving the model has learned the leakage, but that individual noisy traces carry too much weight in the accumulated score. Scaling to 10,000 traces stabilizes the result at rank 32, confirming the signal is real and consistent. Further stabilization (rank → 0) is expected with a larger attack pool, where the Law of Large Numbers suppresses outlier traces.

A completely untrained model would hover around rank 128 (random guess). Rank 32 out of 256 — without any cryptanalysis of the algorithm itself — demonstrates successful key leakage extraction from physical power measurements.

---

## 🛠️ Engineering Challenges Solved

This project wasn't just model design — real GPU memory and training stability problems had to be solved before any learning was possible:

| Problem | Root Cause | Solution |
|---|---|---|
| `ResourceExhausted` on first training attempt | TensorFlow tried to copy the full 7GB dataset + model + gradients into VRAM simultaneously | `AscadH5Generator` — Keras `Sequence` subclass that reads 64 traces at a time directly from HDF5, never materializing the full dataset |
| OOM crash with `Flatten` layer | 15,000 timestep trace × 256 filters after 3 conv blocks = 480,000-element vector per sample | Replaced with `GlobalAveragePooling1D` — collapses to 256 elements regardless of trace length |
| Training progress lost on container crash | Jupyter kernel restart wiped all state | `BackupAndRestore` callback (epoch-level checkpointing) + `CSVLogger` for durable metric history |
| GPU memory fragmentation between SGD/Adam experiments | Keras session retains graph state | `tf.keras.backend.clear_session()` + `gc.collect()` between runs |
| Log-likelihood score permanently destroyed by noisy traces | `log(~0) ≈ -92` from a single outlier overwhelms thousands of valid traces | Epsilon clipping: `log(prob + 1e-40)` prevents the score for the correct key from being permanently suppressed |
| SGD showing no learning | SGD's fixed learning rate too coarse for this loss landscape | Adam (adaptive per-parameter lr) converged faster across all experiments; confirmed by side-by-side 10-epoch comparison |

---

## 📁 Repository Structure

```
ASCAD/
├── ascad_final.ipynb          # Primary notebook: best architecture + key recovery (cells 1–10)
├── ascad_sc3546.ipynb         # Experimental notebook: earlier architecture iterations
├── ASCAD_presentation.pptx    # Project presentation slides
└── README.md
```

> **Note:** `ascad_final.ipynb` cells 1–10 contain the primary, best-performing implementation. Cell 11 onward documents optimization experiments that produced lower performance and are preserved for reference.

**Dataset** (not included — ~7GB): Download `ascadv2-extracted.h5` from the [ASCAD repository](https://github.com/ANSSI-FR/ASCAD) and place it in the project root.

---

## 🚀 Getting Started

### Environment

Tested on:
- Ubuntu 22.04.3 LTS
- TensorFlow 2.15.0
- NVIDIA RTX 3090 (24GB VRAM)

### Requirements

```bash
pip install tensorflow numpy h5py matplotlib tqdm pandas
```

### Run

```bash
jupyter notebook ascad_final.ipynb
```

The notebook is structured as follows:

| Cells | Purpose |
|---|---|
| 1–3 | Environment setup, GPU verification |
| 4–5 | HDF5 data loading, S-Box label generation |
| 6–7 | `AscadH5Generator` disk-streaming pipeline |
| 8 | SGD vs. Adam ablation experiment |
| 9 | High-accuracy training run with full callback suite |
| 10 | **Final SELU model** — best architecture and key recovery attack |
| 11+ | Optimization experiments (preserved, lower performance) |

---

## ⚠️ Disclaimer

This project is strictly for **educational and research purposes**. Side-channel analysis is a legitimate and active field of academic cryptography and hardware security research. The ASCAD dataset is a controlled public benchmark — no real devices or systems were targeted.

Understanding these vulnerabilities is essential for building **countermeasures** and designing **side-channel resistant hardware**.

---

## 📚 References

- [ASCAD: ANSSI SCA Database](https://github.com/ANSSI-FR/ASCAD)
- Prouff et al., *Study of Deep Learning Techniques for Side-Channel Analysis*, IACR ePrint 2018
- Benadjila et al., *Deep learning for side-channel analysis and introduction to ASCAD database*, Journal of Cryptographic Engineering, 2020

---

## 🔗 Acknowledgements

Dataset and baseline forked from [rokolotowicz/ASCAD](https://github.com/rokolotowicz/ASCAD).
