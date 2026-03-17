---
title: AI Application
emoji: 🏠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.9.0
app_file: app.py
python_version: 3.10
pinned: false
short_description: Apartment rent prediction app for the canton of Zurich
---

# Apartment Rent Prediction – Canton of Zurich

## Project Overview
This project predicts monthly apartment rental prices in the canton of Zurich using machine learning.  
A Gradio web application was developed to allow users to enter apartment characteristics and municipality information and receive a rent prediction.

The solution includes:
- a trained regression model
- iterative model comparison
- a newly engineered feature
- a Gradio application for prediction

---

## Dataset
The project uses apartment and municipality-level data from the course material.

### Target Variable
- `price`: monthly apartment rent in CHF

### Final Features
The final model uses the following input features:

- `rooms`
- `area`
- `pop`
- `pop_dens`
- `frg_pct`
- `emp`
- `tax_income`
- `distance_to_center_km`

---

## Preprocessing
The following preprocessing steps were applied:

- loaded the apartment dataset from CSV
- removed missing values
- removed duplicate rows
- selected the relevant input variables
- defined `price` as the target variable
- calculated the new feature `distance_to_center_km` from latitude and longitude using the haversine formula

---

## Evaluation Method
Model performance was evaluated using **5-fold cross-validation**.

The following regression metrics were used:
- **R²**: higher is better
- **RMSE**: lower is better

---

## Iterative Modeling Process

| Iteration | Objective | Model | Hyperparameters | Mean CV R² | Mean CV RMSE |
|---|---|---|---|---:|---:|
| 1 | Build baseline model | LinearRegression | default | 0.536 | 857.47 |
| 1 | Build baseline model | RandomForestRegressor | `n_estimators=300`, `random_state=42`, `n_jobs=-1` | 0.489 | 892.59 |
| 2 | Improve baseline with nonlinear model | LinearRegression | default | 0.536 | 857.47 |
| 2 | Improve baseline with nonlinear model | HistGradientBoostingRegressor | `max_iter=300`, `learning_rate=0.05`, `max_depth=3`, `random_state=42` | 0.590 | 803.29 |
| 3 | Add a new feature and re-evaluate | LinearRegression | default | 0.542 | 851.88 |
| 3 | Add a new feature and re-evaluate | HistGradientBoostingRegressor | `max_iter=300`, `learning_rate=0.05`, `max_depth=3`, `random_state=42` | 0.630 | 764.49 |

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
LinearRegression outperformed RandomForestRegressor on both metrics:
- higher R² (`0.536` vs. `0.489`)
- lower RMSE (`857.47` vs. `892.59`)

Therefore, LinearRegression was selected as the best model of Iteration 1.

---

## Iteration 2
### Objective
Improve baseline performance using a stronger nonlinear model.

### Changes Compared to Iteration 1
- replaced RandomForestRegressor with HistGradientBoostingRegressor
- kept the same preprocessing pipeline
- kept the same feature set

### Models Used
- LinearRegression
- HistGradientBoostingRegressor

### Conclusion
HistGradientBoostingRegressor outperformed LinearRegression:
- higher R² (`0.590` vs. `0.536`)
- lower RMSE (`803.29` vs. `857.47`)

Therefore, HistGradientBoostingRegressor became the best model after Iteration 2.

---

## Iteration 3
### Objective
Add a new engineered feature not used in prior exercises and test whether it improves prediction performance.

### Changes Compared to Iteration 2
- added the new feature `distance_to_center_km`
- calculated it from apartment coordinates and a fixed Zurich city-center reference point
- re-ran the model comparison with the updated feature set

### Models Used
- LinearRegression
- HistGradientBoostingRegressor

### Conclusion
Adding `distance_to_center_km` improved model performance further.

HistGradientBoostingRegressor achieved:
- R² = `0.630`
- RMSE = `764.49`

This outperformed the previous best result from Iteration 2:
- previous R² = `0.590`
- previous RMSE = `803.29`

Therefore, the new feature was useful and improved the model.

---

## Final Selected Model
The final selected model is:

**HistGradientBoostingRegressor**

### Final Hyperparameters
- `max_iter=300`
- `learning_rate=0.05`
- `max_depth=3`
- `random_state=42`

### Reason for Selection
This model achieved the best cross-validation performance of all tested models:
- **Best R²:** `0.630`
- **Best RMSE:** `764.49`

---

## New Feature
The newly engineered feature is:

**`distance_to_center_km`**

It measures the geographic distance between an apartment and Zurich city center using latitude and longitude coordinates.

This feature was not part of the previously shown course feature set and was added to improve location sensitivity in the model.

---

## Application
The Gradio app allows users to enter:
- number of rooms
- apartment area in square meters
- municipality

The application combines the user inputs with municipality-level data and predicts the monthly apartment rent in CHF.

---
