# OSDiff
This repository provides a PyTorch implementation of the paper **"One-Step Diffusion for Perceptual Image Compression"**.

---

## 🧩 Introduction
We propose **OSDiff**, a diffusion-based perceptual image compression method that performs one-step diffusion, drastically reducing inference latency and computational cost. To further boost perceptual quality, we introduce a feature-space discriminator operating on intermediate UNet representations, allowing the model to better align reconstructed features with those of the original images.

---

## 📦 Requirements
```bash
conda create -n osdiff python=3.8
conda activate osdiff
pip install -r requirements.txt
```
---

## ⚙️ Inference

### Download Pre-trained Model

Download the pre-trained checkpoint from [pretrain_weight](https://drive.google.com/drive/folders/15ifY20Dctbku4aLBs1GmAQSzfwb4QUJW?usp=sharing).

---

### Run the following command
```bash
python inference.py \
    --ckpt_lc ./weight/lambda_1.ckpt \
    --input path to input images \
    --output path to output files
```
---


