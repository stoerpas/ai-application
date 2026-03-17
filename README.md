# Apartment Rent Prediction – Canton of Zurich

## Project Overview
This project predicts monthly apartment rental prices in the canton of Zurich using machine learning.  
A Gradio web application was developed so that users can enter apartment and municipality information and receive a predicted monthly rent.

The project includes:

- regression model training
- iterative model comparison
- model evaluation with cross-validation
- a deployable Gradio application

---

## Dataset
The project uses apartment and municipality-level data from the course material.

### Target Variable
- `price`: monthly apartment rent in CHF

### Baseline Features
The following features were used in the current model iterations:

- `rooms`
- `area`
- `pop`
- `pop_dens`
- `frg_pct`
- `emp`
- `tax_income`

---

## Preprocessing
The following preprocessing steps were applied:

- loaded the dataset from CSV
- removed missing values
- removed duplicate rows
- selected the relevant input features
- defined `price` as the target variable

---

## Evaluation Method
Model performance was evaluated using **5-fold cross-validation**.

The following regression metrics were used:

- **R²**: higher values indicate better fit
- **RMSE**: lower values indicate better prediction accuracy

---

## Iterative Modeling Process

| Iteration | Objective | Model | Hyperparameters | Mean CV R² | Mean CV RMSE |
|---|---|---|---|---:|---:|
| 1 | Build a first baseline model | LinearRegression | default | 0.536 | 857.47 |
| 1 | Build a first baseline model | RandomForestRegressor | `n_estimators=300`, `random_state=42`, `n_jobs=-1` | 0.489 | 892.59 |
| 2 | Improve baseline performance with a stronger nonlinear model | LinearRegression | default | 0.536 | 857.47 |
| 2 | Improve baseline performance with a boosting-based model | HistGradientBoostingRegressor | `max_iter=300`, `learning_rate=0.05`, `max_depth=3`, `random_state=42` | 0.590 | 803.29 |

---

## Iteration 1

### Objective
Build a first baseline regression model for apartment rent prediction.

### Changes Compared to Previous Iteration
This was the initial baseline iteration.

### Models Used
- LinearRegression
- RandomForestRegressor

### Conclusion
In Iteration 1, **LinearRegression** performed better than **RandomForestRegressor** on both evaluation metrics:

- higher R² (`0.536` vs. `0.489`)
- lower RMSE (`857.47` vs. `892.59`)

Therefore, **LinearRegression** was selected as the best model of Iteration 1.

---

## Iteration 2

### Objective
Improve the baseline model performance by testing a stronger nonlinear model.

### Changes Compared to Iteration 1
- replaced RandomForestRegressor with HistGradientBoostingRegressor
- kept the same preprocessing pipeline
- kept the same feature set
- compared the new model against the previous best baseline

### Models Used
- LinearRegression
- HistGradientBoostingRegressor

### Conclusion
In Iteration 2, **HistGradientBoostingRegressor** outperformed **LinearRegression**:

- higher R² (`0.590` vs. `0.536`)
- lower RMSE (`803.29` vs. `857.47`)

This means Iteration 2 improved the model performance compared with the baseline.  
Therefore, **HistGradientBoostingRegressor** is the best model so far and is selected as the current final model.

---

## Final Selected Model
The currently selected final model is:

**HistGradientBoostingRegressor**

### Final Model Hyperparameters
- `max_iter=300`
- `learning_rate=0.05`
- `max_depth=3`
- `random_state=42`

### Reason for Selection
This model achieved the best cross-validation performance among all tested models:

- **Best R²:** `0.590`
- **Best RMSE:** `803.29`