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