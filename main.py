import argparse
import pandas as pd
import time
from src.data_loader import load_data, validate_columns
from src.preprocessor import preprocess_data
from src.model_selector import train_and_evaluate_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Path to cleaned CSV")
    parser.add_argument("--target", required=True, help="Target column name")
    parser.add_argument("--task", required=True, choices=["classification", "regression"], help="ML task type")
    parser.add_argument("--tune", default="none", choices=["none", "grid"], help="Enable tuning (default: none)")
    parser.add_argument("--models", nargs="*", help="Optional: list of models to test (case-sensitive)")
    args = parser.parse_args()

    print(f"\n📂 Loading data: {args.csv}")
    df = load_data(args.csv)
    validate_columns(df, args.target)

    print("🧹 Preprocessing...")
    X_train, X_test, y_train, y_test, _ = preprocess_data(df, args.target)

    print("Data shapes:", X_train.shape, X_test.shape, len(y_train), len(y_test))

    print(f"⚙️  Running model selection (task={args.task}, tuning={args.tune})")
    start = time.time()
    best_model, best_name, results, _ = train_and_evaluate_models(X_train, X_test, y_train, y_test, args.task,
                                                                  args.tune)
    end = time.time()

    print(f"\n✅ Best Model: {best_name}")
    print("📊 All Model Scores:")
    for result in results:
        print(f"  {result['model']}: {result}")

    print(f"\n⏱️ Time elapsed: {round(end - start, 2)} seconds")


if __name__ == "__main__":
    main()
