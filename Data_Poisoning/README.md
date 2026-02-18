
# 🧪 Data Poisoning (Backdoor Data)

This directory contains the **poisoned and clean datasets** used in *Don't Shift the Trigger: Robust Gradient Ascent for Backdoor Unlearning*.

We provide **three backdoor attacks**:
- **BadNets**
- **AddSent**
- **HiddenKiller**

For each attack, we include **three datasets**:
- **SST-2**
- **HSOL**
- **AG**

Each *(attack, dataset)* pair contains **five CSV files**:
- `train_clean.csv`
- `train_poisoned.csv`
- `test_clean.csv`
- `test_poisoned_part.csv`
- `test_poisoned_all.csv`

---

## 📚 Table of Contents
1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [File Definitions](#file-definitions)
4. [Target Classes and Poisoning Ratio](#target-classes-and-poisoning-ratio)
5. [Evaluation Protocol Notes](#evaluation-protocol-notes)
6. [HiddenKiller Generation (OpenAttack)](#hiddenkiller-generation-openattack)

---

## Overview
We construct poisoned training sets by inserting triggers into a subset of training samples and **changing their labels to the target class** (i.e., targeted backdoor poisoning). For testing, we **only insert triggers** to measure whether the model is misclassified (we do **not** flip labels in test files).

---

## Directory Structure
A typical layout is:

```
Data_Poisoning/
├── BadNets/
│   ├── SST-2/
│   ├── HSOL/
│   └── AG/
├── AddSent/
│   ├── SST-2/
│   ├── HSOL/
│   └── AG/
└── HiddenKiller/
    ├── SST-2/
    ├── HSOL/
    └── AG/
```

Each dataset folder contains:

train_clean.csv
train_poisoned.csv
test_clean.csv
test_poisoned_part.csv
test_poisoned_all.csv

```md
---

## File Definitions:
- `train_clean.csv`: A fully clean training set **no triggers** are inserted.
- `train_poisoned.csv`: A poisoned training set constructed by injecting triggered samples into train_clean.csv with a poisoning ratio of 10%. Triggered samples in training data are label-flipped to the target class.
- `test_clean.csv`: A clean test set **no trigger** are inserted, used to measure clean accuracy.
- `test_poisoned_part.csv`: A partially triggered test set where we insert the trigger only into non-target-class samples, used to evaluate backdoor success (the trigger causes non-target --> target misclassification).
- `test_poisoned_all.csv`: A fully triggered test set where we insert the trigger into samples from all classes, used to detect trigger shifting after unlearning.

---

## Target Classes and Poisoning Ratio
Poisoning ratio: 10% (for main experiments)
Target class per dataset:
SST-2: target class = positive
HSOL: target class = non-hate
AG: target class = world

---

## Evaluation Protocol Notes:
test_poisoned_part.csv -- measures standard backdoor behavior (targeted misclassification).
test_poisoned_all.csv -- is specifically designed to identify trigger shifting after unlearning.

## HiddenKiller Generation
For HiddenKiller, we use [OpenAttack](https://github.com/thunlp/OpenAttack) to generate poisoning data. For more attack settings, please see Section 6.1.
