# ICLR 2026
# Don't Shift the Trigger: Robust Gradient Ascent for Backdoor Unlearning
---

## 📚 Table of Contents
1. [Overview](#overview)
2. [System Configuration](#system-configuration)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Cloning the Repository](#cloning-the-repository)
   - [Running the Code](#running-the-code)
   - [Command-Line Options](#command-line-options)
5. [Datasets](#datasets)
6. [Citation](#citation)
7. [Contact \& Support](#contact--support)

---

## Overview
Backdoor attacks pose a significant threat to machine learning models, allowing adversaries to implant hidden triggers that alter model behavior when activated. Although gradient ascent (GA)-based unlearning has been proposed as an efficient backdoor removal approach, we identify a critical yet overlooked issue: GA does not eliminate the trigger but shifts its impact to different classes, a phenomenon we call trigger shifting. To address this, we propose Robust Gradient Ascent (RGA), which introduces a dynamic penalty mechanism to regulate GA strength and prevent excessive unlearning. For more details please refer to our (paper)[https://openreview.net/pdf?id=voqtsqYS6j].

## System Configuration
- OS, GPU, CUDA, Python version, etc.

## Installation
```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
conda create -n rga python=3.10 -y
conda activate rga
pip install -r requirements.txt
