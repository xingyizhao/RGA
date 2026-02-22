# ICLR 2026
# Don't Shift the Trigger: Robust Gradient Ascent for Backdoor Unlearning
---

## 📚 Table of Contents
1. [Overview](#overview)
2. [System Configuration](#system-configuration)
3. [Installation--Usage](#installation--Usage)
5. [Datasets](#datasets)
6. [Citation](#citation)
7. [Contact \& Support](#contact--support)

---

## 🔍 Overview
Backdoor attacks pose a significant threat to machine learning models, allowing adversaries to implant hidden triggers that alter model behavior when activated. Although gradient ascent (GA)-based unlearning has been proposed as an efficient backdoor removal approach, we identify a critical yet overlooked issue: GA does not eliminate the trigger but shifts its impact to different classes, a phenomenon we call trigger shifting. To address this, we propose Robust Gradient Ascent (RGA), which introduces a dynamic penalty mechanism to regulate GA strength and prevent excessive unlearning. For more details please refer to our [paper](https://openreview.net/pdf?id=voqtsqYS6j).

## ⚙️ System Configuration
The code has been developed and tested using the following system setup:

- **GPU Driver:** NVIDIA driver 590.44.01   
- **CUDA Version:** 13.1  
- **Python Version:** 3.10.19  
- **PyTorch Version:** 2.9.0+cu128 

## 🚀 Installation--Usage
- **Installation**
```bash
git clone https://github.com/xingyizhao/RGA.git
cd RGA
conda create -n rga python=3.10 -y
conda activate rga
pip install -r requirements.txt
```

- **Usage**

Bert
```
python bert_unlearn.py
```

Distilbert
```
python distilbert_unlearn.py
```

Llama2
```
python llama_unlearn.py
