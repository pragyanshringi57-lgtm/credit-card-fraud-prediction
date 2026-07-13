from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from train_model import run_training


def calculate_metrics(y_test, y_pred) -> Dict[str, float]:
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def plot_results(y_test, y_pred, save_path: Path | None = None) -> None:
    matrix = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def evaluate_model(model, x_test, y_test) -> Dict[str, float]:
    y_pred = model.predict(x_test)
    metrics = calculate_metrics(y_test, y_pred)

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    plot_results(y_test, y_pred)
    return metrics


def run_evaluation() -> Dict[str, float]:
    model, x_test, y_test, _ = run_training()
    return evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    run_evaluation()