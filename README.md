# Denoising Autoencoders for Purifying Breast Cancer Gene Expression Data

---

## Overview

Tissue biopsies taken from cancer patients are never purely tumour cells, they always contain a mixture of tumour and normal healthy cells. This contamination corrupts gene expression measurements, making downstream analysis less reliable.

This project develops a **Denoising Autoencoder (DAE)** in PyTorch to reconstruct clean, purified tumour gene expression profiles from contaminated mixed-tissue samples. The model is trained and evaluated on the large-scale **SCAN-B breast cancer dataset** (~9,200 samples, ~19,675 genes).

---

## The Problem

When sequencing a biopsy, the measured gene expression signal is a mixture:

```
measured signal = (cancer_ratio x tumour signal) + (normal_ratio x normal signal)
```

If 30% of the cells in the biopsy are normal, 30% of the signal is noise. This affects:
- Subtype classification (e.g. PAM50)
- Biomarker discovery
- Treatment stratification

The goal is to recover the true tumour signal from the contaminated input.

---

## Approach

### Dataset Construction
Since ground-truth clean signals are unknown for real mixed samples, training data was **synthetically constructed** by mixing known pure tumour profiles (SCAN-B) with normal tissue profiles at varying ratios:

| Cancer Ratio | Normal Ratio | Description |
|---|---|---|
| 1.0 | 0.0 | Pure tumour (baseline) |
| 0.9 | 0.1 | 10% contamination |
| ... | ... | ... |
| 0.1 | 0.9 | 90% contamination |
| 0.0 | 1.0 | Pure normal |

This gave input-output pairs: **(mixed input --> pure cancer target)** for supervised training.

### Model Architecture

A deep encoder-decoder DAE with the following structure:

```
Input (19,675 genes)
    ↓
Encoder:  Linear(19675→1024) → BatchNorm → LeakyReLU → Dropout(0.1)
          Linear(1024→512)   → BatchNorm → LeakyReLU
    ↓
Bottleneck: Linear(512→256)  → BatchNorm → LeakyReLU → Dropout(0.1)
    ↓
Decoder:  Linear(256→512)    → BatchNorm → LeakyReLU
          Linear(512→1024)   → BatchNorm → LeakyReLU
          Linear(1024→19675)
    ↓
Output: Reconstructed pure tumour profile
```

### Training Details

| Parameter | Value |
|---|---|
| Optimiser | Adam (lr=0.0005, weight_decay=1e-6) |
| Loss function | MSE |
| Batch size | 200 |
| Max epochs | 100 |
| Early stopping | Yes (patience-based) |
| Gradient clipping | max_norm=1.0 |
| Input transform | log2(x + 0.1) normalisation |

---

## Results

Evaluated on held-out pure cancer test samples (1,206 samples):

| Metric | Noisy Input (Baseline) | DAE Output |
|---|---|---|
| MSE | Higher | **0.3385** |
| R² | Lower | **0.9670** |

An R² of **0.967** means the model reconstructs 96.7% of the variance in the true tumour signal, demonstrating strong reconstruction fidelity across varying levels of contamination.

The model was trained and evaluated separately for each cancer/normal ratio, showing robust performance even at high contamination levels (e.g. 30% cancer / 70% normal).

---

## Requirements

```
torch
pandas
numpy
scikit-learn
matplotlib
```

Install with:
```bash
pip install torch pandas numpy scikit-learn matplotlib
```

---

## Data

This project uses two datasets:

- **SCAN-B** - Swedish breast cancer gene expression dataset (9,206 tumour samples)
- **Normal tissue** - 66 matched normal tissue samples

The datasets are not included in this repository due to size and data access constraints. The SCAN-B dataset is available from the [SCAN-B study](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81538).

---

## Key Findings

- The DAE successfully recovers clean tumour signals from heavily contaminated inputs
- Performance remains strong even at 30% cancer / 70% normal ratios
- Synthetic dataset construction based on realistic tumour purity ratios is an effective training strategy
- Results suggest potential for improving downstream tasks such as breast cancer subtype classification (e.g. PAM50)

---

## Future Work

- Evaluate impact on downstream classification tasks (PAM50 subtype classifier)
- Explore variational autoencoders (VAEs) and adversarial autoencoders (AAEs)
- Extend to multi-omics data (methylation, proteomics)
- Test on independent cohorts for generalisation

---


