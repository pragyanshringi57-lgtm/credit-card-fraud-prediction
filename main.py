from evaluate import evaluate_model
from train_model import run_training


def main() -> None:
    model, x_test, y_test, _ = run_training()
    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()