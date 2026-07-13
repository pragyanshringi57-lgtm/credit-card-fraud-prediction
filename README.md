# Credit Card Fraud Detection

This folder contains the runnable source files for GitHub upload.

## Files

- `data_preprocessing.py` loads the data, scales `Time` and `Amount`, and applies SMOTE.
- `train_model.py` trains the XGBoost model.
- `evaluate.py` calculates precision, recall, F1-score, accuracy, and plots the confusion matrix.
- `main.py` runs the full pipeline.

## Run

```bash
pip install -r requirements.txt
python main.py
```

## Data Source

The dataset comes from the Kaggle notebook/input source you shared: [Credit Fraud: Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/input).



🕵️‍♂️ Catching Credit Card Fraud with Machine Learning
Hi! Welcome to my credit card fraud detection project.

The goal of this project was to build a machine learning pipeline that can accurately flag fraudulent credit card transactions.

The Challenge: The Class Imbalance Trap
When I first started looking at financial transaction data, I ran into a classic problem: fraud is incredibly rare. In a typical dataset, 99.8% of transactions are perfectly normal, and only 0.2% are actual fraud.

If you feed that raw data into a standard machine learning model, it gets "lazy." It realizes it can just guess "Not Fraud" every single time and technically be 99.8% accurate! Standard accuracy is a dangerous metric here, so I had to rethink the approach.

How I Solved It
Instead of relying on basic accuracy, I focused on optimizing Precision and the F1-Score. I wanted a model that catches the bad guys without accidentally declining the credit cards of innocent customers.

To make this happen, I used two main techniques:

SMOTE (Synthetic Minority Over-sampling Technique): To fix the imbalanced data, I couldn't just copy and paste the few fraudulent examples (which causes overfitting). Instead, I used SMOTE to generate brand new, mathematically synthetic examples of fraud based on existing patterns. This gave the model enough "bad" data to actually learn what fraud looks like.
XGBoost: For the actual classification, I went with XGBoost. Financial data is messy, non-linear, and full of extreme outliers. XGBoost builds decision trees sequentially to correct its own errors, making it incredibly powerful for tabular data like this.
The Results
By pairing SMOTE with XGBoost, the model successfully bypassed the class imbalance trap and achieved an 94% Precision/F1-score on unseen testing data.

