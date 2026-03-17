import pickle
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_validate

DATA_FILE = "data/original_apartment_data_analytics_hs24.csv"
MODEL_FILE = "model.pkl"

FEATURES = ["rooms", "area", "pop", "pop_dens", "frg_pct", "emp", "tax_income"]
TARGET = "price"


def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df.dropna().drop_duplicates()
    return df


def evaluate_model(name, model, X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=("r2", "neg_root_mean_squared_error"),
        return_train_score=False,
    )
    return {
        "model": name,
        "cv_r2_mean": scores["test_r2"].mean(),
        "cv_rmse_mean": -scores["test_neg_root_mean_squared_error"].mean(),
    }


def main():
    df = load_data()
    X = df[FEATURES]
    y = df[TARGET]

    models = {
        "LinearRegression": LinearRegression(),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    results = []
    for name, model in models.items():
        results.append(evaluate_model(name, model, X, y))

    results_df = pd.DataFrame(results).sort_values("cv_rmse_mean")
    print(results_df)

    best_model_name = results_df.iloc[0]["model"]
    final_model = models[best_model_name]
    print(f"Best model: {best_model_name} (RMSE: {results_df.iloc[0]['cv_rmse_mean']:.2f})")
    final_model.fit(X, y)

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(final_model, f)

    print(f"Saved model to {MODEL_FILE}")


if __name__ == "__main__":
    main()