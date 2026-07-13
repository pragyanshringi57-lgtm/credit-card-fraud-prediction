from pathlib import Path
from typing import Dict, Tuple, Union

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


DEFAULT_DATA_PATH = Path(__file__).with_name("creditcard.csv")


def load_data(data_path: Union[str, Path] = DEFAULT_DATA_PATH) -> pd.DataFrame:
    if not Path(data_path).exists():
        data_path = Path(__file__).resolve().parent.parent / "creditcard.csv"

    data = pd.read_csv(data_path)
    duplicate_count = data.duplicated().sum()

    if duplicate_count > 0:
        data = data.drop_duplicates().reset_index(drop=True)

    return data


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    x = data.drop(columns=["Class"])
    y = data["Class"]

    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=y,
    )


def scale_data(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, StandardScaler]]:
    scaler_time = StandardScaler()
    scaler_amount = StandardScaler()

    x_train = x_train.copy()
    x_test = x_test.copy()

    x_train["scaled_time"] = scaler_time.fit_transform(x_train[["Time"]])
    x_test["scaled_time"] = scaler_time.transform(x_test[["Time"]])

    x_train["scaled_amount"] = scaler_amount.fit_transform(x_train[["Amount"]])
    x_test["scaled_amount"] = scaler_amount.transform(x_test[["Amount"]])

    x_train = x_train.drop(columns=["Time", "Amount"])
    x_test = x_test.drop(columns=["Time", "Amount"])

    scalers = {
        "time_scaler": scaler_time,
        "amount_scaler": scaler_amount,
    }

    return x_train, x_test, scalers


def apply_smote(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sampling_strategy: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series]:
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    return smote.fit_resample(x_train, y_train)


def preprocess_data(
    data_path: Union[str, Path] = DEFAULT_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    sampling_strategy: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, StandardScaler]]:
    data = load_data(data_path)
    x_train, x_test, y_train, y_test = split_data(
        data,
        test_size=test_size,
        random_state=random_state,
    )
    x_train, x_test, scalers = scale_data(x_train, x_test)
    x_train, y_train = apply_smote(
        x_train,
        y_train,
        sampling_strategy=sampling_strategy,
        random_state=random_state,
    )

    return x_train, x_test, y_train, y_test, scalers


if __name__ == "__main__":
    data = load_data()
    print(f"Rows after duplicate removal: {len(data)}")