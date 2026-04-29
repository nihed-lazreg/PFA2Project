# Signature Verification System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Dataset](https://img.shields.io/badge/Dataset-CEDAR-purple.svg)

---

## What is this project ?

An automatic **offline handwritten signature verification system** that distinguishes genuine signatures from forgeries using deep learning. A new client can be enrolled with only **3 to 5 reference signatures**, without any retraining.

```
Image → Preprocessing → ResNet50 → 128D Embedding → Cosine Similarity → Genuine / Forgery
```

---

## What we built

- A **Siamese Neural Network** based on ResNet50 trained with Triplet Loss
- A **preprocessing pipeline** : grayscale, Otsu binarization, crop, resize 224×224
- A **data augmentation** strategy with 10 transformations simulating real conditions
- A **two-phase training** : frozen backbone → partial fine-tuning + hard negative mining
- A **SignatureVerifier** class for client enrollment and verification in production

---

## Tools & Libraries

| Tool | Usage |
|---|---|
| TensorFlow / Keras | Model building and training |
| ResNet50 (ImageNet) | Feature extraction backbone |
| OpenCV | Image preprocessing and augmentation |
| NumPy | Embedding manipulation |
| Scikit-learn | ROC curve, AUC, train/val split |
| Matplotlib | Loss curves and ROC visualization |
| Kaggle (GPU T4) | Training platform |

---

## Dataset

**CEDAR** — 55 signers, 24 genuine + 24 forged signatures per signer (2,640 images total).
Download : https://cedar.buffalo.edu/NIJ/data/signatures.rar

---

## Results

| EER | AUC | FAR | FRR |
|---|---|---|---|
| — | — | — | — |

> To be updated after training.

---

## Author

Engineering thesis project — Signature Verification using Deep Learning.
