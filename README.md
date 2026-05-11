# Email Message Classification Using Text Analysis

This project implements a system for automatic email message classification using Machine Learning and Natural Language Processing (NLP) techniques. The system analyzes email content and categorizes messages into appropriate classes, with the ability to determine message priority.

## Project Description

The goal of the project is to develop an intelligent system that automatically classifies email messages into the following categories:
- Personal
- Business
- Promotions
- Notifications
- Spam

### In addition to classification, the system can determine how the message should be processed:
- Read Immediately
- Read Later
- Archive

### The project uses a combination of:
- NLP techniques
- TF-IDF text representation
- Handcrafted features
- Multiple Machine Learning models

### Technologies Used

1. Python
2. Scikit-learn
3. XGBoost
4. NLTK
5. Stanza
6. Pandas
7. NumPy

### Project Files

1. extract.py - Script for extracting email messages from .mbox files and converting them into CSV format.
2. preprocess.py - Text preprocessing module including text cleaning, link removal, stop-word removal, language detection, lemmatization, and stemming. Supported languages: Serbian, English.
3. features.py - Feature extraction pipeline implementation. Includes TF-IDF for words and characters, signal features, numerical features, and sender domain processing.
4. models.py - Contains implementations and configurations of classification models: Logistic Regression, Random Forest, XGBoost, SVM, MLPClassifier.
5. train.py - Script for model training, validation, performance evaluation, and confusion matrix visualization.
6. predict.py - Allows model testing through terminal input of new email messages.
7. Project VI.docx - Project documentation containing detailed analysis of data preparation, system architecture, model evaluation, error analysis, and possible improvements.
