# OSDiff
This repository provides a PyTorch implementation of the paper **"One-Step Diffusion for Perceptual Image Compression"**.

---

## 🧩 Introduction
todo

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

### Run the Test Script
```bash
python inference.py \
    --ckpt_lc ./weight/lambda_1.ckpt \
    --input path to input images \
    --output path to output files
```
---


