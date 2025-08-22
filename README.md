# Aspect-Based Stock Movement Prediction

![Block diagram](NLP.png)

## 🚀 Overview

This NLP project predicts stock price movements using aspect-based sentiment analysis of financial news. It combines textual sentiment features with numerical indicators to predict stock movements for Indian companies.

## ✨ Key Features

- **Aspect-Based Sentiment Analysis**: Extract sentiment on various aspects (management, market trends) from financial news
- **Multi-Modal Prediction**: Combines numerical and textual features
- **Web Scraping**: Automated collection of financial news headlines
- **Advanced NLP**: Syntactic and numerical encoding of textual data
- **Time Series Modeling**: GRU-based model for stock price prediction

## 📁 Project Structure

```
NLP-master/
├── README.md                    # Project documentation
├── requirements.txt             # Dependencies
├── aspects/                     # Aspect extraction and sentiment analysis
├── Aspect_Feature/             # Feature engineering
├── financial_news/             # Dataset storage
├── news_extraction/            # Web scraping
├── numerical_encoder/          # Numerical text encoding
├── stockmovementpredictor/     # Main prediction model
└── Syntactic_encoder/          # Syntactic analysis
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Quick Setup

1. **Clone/Download the project**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download required NLP models**:
   ```bash
   python -m spacy download en_core_web_sm
   python -m nltk.downloader stopwords punkt wordnet
   ```

## 🎯 Usage

### Step-by-Step Execution

1. **News Extraction**:
   ```bash
   cd news_extraction
   python webscraping.py
   # Or run news_api.ipynb for specific companies
   ```

2. **Aspect Extraction**:
   - Run `aspects/aspect-extraction.ipynb`
   - Run `aspects/sentiment-score.ipynb`

3. **Text Encoding**:
   - Run `Syntactic_encoder/syntactic_encoder.ipynb`
   - Run `numerical_encoder/numeral-ipynb.ipynb`

4. **Feature Engineering**:
   - Run `Aspect_Feature/aspect-embeddings.ipynb`

5. **Stock Prediction**:
   ```bash
   cd stockmovementpredictor
   python stockmovementpredictor.py
   ```

## 📊 Dataset

The project uses financial news data for three major Indian companies:
- **Reliance Industries**
- **Tata Group** 
- **Apollo Hospitals**

Each dataset contains date-wise financial news, aspect-based sentiment labels, and processed text features.

## 🧠 Models and Techniques

### NLP Models
- **FinBERT**: For financial sentiment analysis
- **spaCy**: For syntactic parsing and dependency analysis
- **Custom Aspect Extraction**: Domain-specific aspect identification

### Machine Learning
- **GRU**: For time series prediction
- **Feature Engineering**: Technical indicators, moving averages
- **Multi-modal Fusion**: Combining textual and numerical features

## 📈 Results

The model combines aspect-based sentiment analysis with technical indicators to predict stock movements. Detailed results and visualizations are available in individual notebook outputs.

## ⚠️ Important Notes

- Some notebooks are optimized for **Kaggle execution** due to GPU requirements
- Check individual folder README files for specific instructions
- **Project Report**: [Overleaf Link](https://www.overleaf.com/read/nckkjzsvtnvq#a0a5f0)

## 📧 Contact

For questions about this project, please refer to the individual notebook documentation or the project report.
