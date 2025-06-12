# E-Commerce-AI-Capstone-Project

# 🛒 Sentiment Analysis on Amazon E-commerce Reviews using ML Classifiers

## 📌 Project Overview  
The objective of this project is to perform **Sentiment Analysis** on an Amazon e-commerce product review dataset. The dataset initially contains an imbalance between positive and negative classes. To overcome this, **SMOTE (Synthetic Minority Over-sampling Technique)** is used to balance the dataset.

The project involves:
- Text preprocessing and feature extraction
- Handling class imbalance with SMOTE
- Applying multiple ML classification algorithms
- Comparing model performances using accuracy and F1-score

The final results show that **Random Forest Classifier** outperforms others in both accuracy and F1-score.

## 🔍 Detailed Project Description

### 1. 📦 Importing Required Packages
Imported necessary Python libraries for data processing, modeling, and evaluation:
- `pandas`, `numpy` – for data manipulation
- `nltk`, `re`, `string` – for natural language preprocessing
- `sklearn` – for ML models, metrics, SMOTE, train-test split, vectorization
- `xgboost` – for implementing XGBClassifier
- `imblearn.over_sampling.SMOTE` – to balance the dataset

### 2. 📑 Loading and Exploring the Dataset
- Read the Amazon product review dataset using `pandas.read_csv()`.
- Checked the distribution of sentiment labels to confirm **class imbalance** (e.g., more positive reviews than negative ones).

### 3. 🧹 Text Preprocessing
Performed necessary cleaning of review text for model readiness:
- Converted all text to lowercase
- Removed punctuation, digits, and special characters using `re` and `string`
- Tokenized the text using `nltk`
- Removed stopwords using NLTK’s stopword list
- Lemmatized words using WordNetLemmatizer to reduce words to their base form

### 4. ✨ Feature Extraction
- Used **TF-IDF Vectorizer** from `sklearn` to convert cleaned text into numerical feature vectors.
- Applied `fit_transform()` on training data and `transform()` on test data.

### 5. ⚖️ Handling Class Imbalance with SMOTE
- Used **SMOTE** from `imblearn.over_sampling` to oversample the minority class (e.g., negative reviews).
- Ensured that both sentiment classes are balanced before model training.

### 6. 🧠 Machine Learning Models Used
Trained multiple classifiers and compared their performance:
- **Random Forest Classifier**
- **XGBoost Classifier**
- **Support Vector Machine (SVM)**
- **Voting Classifier** (Ensemble of multiple models)
- **Multilayer Perceptron (MLPClassifier)** – a feedforward neural network

### 7. 📈 Model Evaluation
- Evaluated models using:
  - **Accuracy Score**
  - **F1 Score** (to account for class balance)
  - **Confusion Matrix**
- Compared metrics across all models

#### 🔍 Observation:
- **Random Forest Classifier** showed the **highest accuracy and F1-score**, making it the best-performing model on this dataset.

## ✅ Key Takeaways
- **SMOTE** effectively handles imbalanced datasets for classification tasks.
- **TF-IDF vectorization** provides a solid feature representation for text data.
- **Random Forest** offers strong performance due to its ensemble nature and robustness to noise.
- Comparing multiple classifiers helps in selecting the best model for deployment.

## 🛠️ Technologies and Tools Used
- **Python**
- **Libraries:**
  - pandas, numpy
  - nltk, re, string
  - scikit-learn
  - imbalanced-learn (SMOTE)
  - xgboost

## 🚀 Conclusion
This project demonstrates how machine learning techniques can be applied to extract sentiment from Amazon product reviews. The combination of **SMOTE for balancing**, **TF-IDF for vectorization**, and **multiple classifiers for prediction** ensures reliable sentiment classification. The **Random Forest classifier** stood out as the most effective in handling this problem.



  
