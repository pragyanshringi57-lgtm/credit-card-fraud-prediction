from pathlib import Path
from typing import Dict, Tuple

import joblib
from xgboost import XGBClassifier

from data_preprocessing import DEFAULT_DATA_PATH, preprocess_data


ARTIFACT_DIR = Path(__file__).parent


def train_xgboost(
    x_train,
    y_train,
    random_state: int = 42,
) -> XGBClassifier:
    model = XGBClassifier(
        max_depth=20,
        random_state=random_state,
        eval_metric="logloss",
    )
    model.fit(x_train, y_train)
    return model


def save_model(
    model: XGBClassifier,
    model_path: Path = ARTIFACT_DIR / "xgboost_fraud_model.pkl",
) -> None:
    joblib.dump(model, model_path)


def run_training() -> Tuple[XGBClassifier, object, object, Dict[str, object]]:
    x_train, x_test, y_train, y_test, scalers = preprocess_data(DEFAULT_DATA_PATH)
    model = train_xgboost(x_train, y_train)
    save_model(model)
    return model, x_test, y_test, scalers


if __name__ == "__main__":
    model, _, _, _ = run_training()
    print("XGBoost model trained and saved.")