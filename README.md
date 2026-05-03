# Blockchain Fraudulent Wallet Detection System

A Streamlit-based prototype for early-stage blockchain wallet risk detection using **XGBoost** and **dynamic time-window behavioral features**.

---

## Project Overview

This project implements a blockchain fraudulent wallet detection prototype designed to identify potentially malicious wallet addresses from their **early on-chain behavioral patterns**.

Instead of relying on full wallet lifecycle data, the system uses a **dynamic time-window design**:
- taking the **first incoming fund event** as the common starting point
- extracting wallet behavior within a fixed early observation window
- reducing data leakage and lifecycle bias in model evaluation

The current prototype focuses on **Ethereum-compatible networks** and provides an interactive interface for wallet risk analysis.

---

## Research Background

This repository is part of a master's thesis project on blockchain fraudulent wallet detection.

The research proposes a time-consistent detection framework by combining:

- dynamic time-window feature engineering
- address-level behavioral features
- XGBoost-based classification
- rule-based anomaly override
- interactive Streamlit deployment

The goal is to improve the realism, interpretability, and practical applicability of early-stage blockchain fraud detection.

---

## Core Features

- **XGBoost main detection model**
- **Dynamic time-window feature extraction**
- **T+7 main detection setting**
- **Multi-chain wallet scanning interface**
- **Rule-based safety override**
- **Fraud risk score output**
- **Behavior summary visualization**
- **Interactive Streamlit prototype**

---

## Supported Networks

The current Streamlit prototype supports the following EVM-compatible networks through Alchemy API:

- Ethereum
- Arbitrum
- Polygon
- Base
- Optimism
- BNB Chain

---

## Main Files

- `app.py`  
  Main Streamlit application

- `requirements.txt`  
  Python dependency list

- `fraud_detector_xgb_t7.joblib`  
  Final XGBoost T+7 model file used in the prototype

- `model_columns_v5.joblib`  
  Feature column definition file corresponding to the final model

---

## Detection Logic

The system works in the following steps:

1. User inputs a wallet address  
2. User selects a blockchain network  
3. The system queries on-chain transfer history through Alchemy API  
4. The first incoming transaction is used as the time anchor  
5. Behavioral features are extracted within the early observation window  
6. Features are aligned with the trained XGBoost model input columns  
7. The model outputs a fraud probability score  
8. A rule-based override checks for extreme abnormal high-risk behavior  
9. The final result is displayed in the Streamlit interface

---

## Model Information

The deployed prototype uses:

- **Model type:** XGBoost classifier
- **Main setting:** T+7 dynamic time-window
- **Purpose:** Early-stage fraudulent wallet detection
- **Output:** Fraud probability score and binary risk interpretation

### Main Performance (T+7)
- **ROC-AUC:** 0.9432
- **PR-AUC:** 0.9198
- **F1-score:** 0.8399
- **Recall:** 0.8462

These metrics correspond to the main test setting used in the thesis.

---

## Prototype Interface

The Streamlit app provides:

- wallet address input
- network selection
- fraud risk prediction
- behavioral feature summary
- simple visualization of selected features
- detailed extracted feature display for debugging and explanation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
