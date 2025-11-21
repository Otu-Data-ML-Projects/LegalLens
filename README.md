# ⚖️ LegalLens: AI-Powered T&C Analyzer

> **Neural Networks Course Final Project** > *Automatic Text Summarization & Risk Detection in Legal Documents*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM%2BAttention-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Chrome Extension](https://img.shields.io/badge/Chrome-Extension-yellow)

## 📖 Project Overview
**LegalLens** is an intelligent tool designed to analyze long and complex "Terms and Conditions" (T&C) agreements. Instead of reading thousands of words, users can instantly identify "unfair" or "risky" clauses hidden in the fine print.

This project implements an **Extractive Summarization** approach using a custom **Bi-Directional LSTM with Attention Mechanism** to assign importance scores to sentences based on their potential risk to the user.

## 🎯 Goals
* **Academic:** Implementing and understanding RNNs (LSTM/GRU) and Attention mechanisms from scratch without relying on pre-trained Transformers like BERT.
* **Practical:** Solving the "nobody reads T&C" problem with an accessible browser extension.

## 🏗️ Architecture
The system consists of three main components:

1.  **Core AI Model (PyTorch):**
    * **Embedding:** GloVe / Custom Word Embeddings.
    * **Encoder:** Bi-Directional LSTM to capture context from both directions.
    * **Attention Layer:** Calculates contribution of each word/sentence to the "risk" score.
    * **Classifier:** Binary classification (Safe vs. Unfair).

2.  **Backend API (FastAPI):**
    * Serves the trained PyTorch model via REST API.
    * Handles text preprocessing and tokenization.

3.  **Frontend (Chrome Extension):**
    * Extracts text from the active browser tab.
    * Highlights risky sentences directly on the webpage using API response.

## 📂 Dataset
We utilize the **LexGLUE (Legal General Language Understanding Evaluation)** dataset, specifically the **Unfair-ToS** subset, which contains annotated Terms of Service agreements labeled for unfair contractual terms.

## 🚀 Installation & Usage

### Prerequisites
* Python 3.9+
* Node.js (optional, for extension dev)

### 1. Setup Environment
```bash
git clone [https://github.com/OrganizationName/LegalLens.git](https://github.com/OrganizationName/LegalLens.git)
cd LegalLens
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```powershell
cd src/api
uvicorn main:app --reload
```

### 3. Load Extension

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode".
3. Click "Load unpacked" and select the `extension/` folder.

## 📝 License

This project is developed for educational purposes as part of the Neural Networks course at Ostim Technical University.

`---