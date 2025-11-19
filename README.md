# fungicide_model
This repository provides a reproducible workflow for analyzing how environmental conditions drive fungicide residue degradation in soybean crops. It includes a Jupyter Notebook, example dataset, and scripts for fitting statistical models, evaluating performance, and generating validation figures.
[README.md](https://github.com/user-attachments/files/23636865/README.md)

# Fungicide Residue Degradation Models

This repository contains a full reproducibility pipeline for the article:

**Assessment and modeling of environmental factors that drive chemical degradation of soybean fungicides**

## Structure
```
fungicide_model/
│── notebook/
│   └── fungicide_modeling.ipynb
│── scripts/
│   └── model_functions.py
│── data/
│   └── example_dataset.csv
│── README.md
│── requirements.txt


# ================================================================
#   Assessment and Modeling of Environmental Factors Driving
#   Chemical Degradation of Soybean Fungicides
#
#   Reproducibility Notebook (for GitHub)
#   Author: <Muricio Fornalski Soares>
#   Date: <2025-11-19>
# ================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (8,6)
plt.rcParams['font.size'] = 12


# ------------------------------------------------
# Load dataset containing:
# - Fungicide residue values
# - Environmental variables
# - Active ingredient identifiers
# ------------------------------------------------

data = pd.read_csv("fungicide_dataset.csv")   # <-- replace with your CSV file name
data.head()


# Drop missing values
df = data.dropna()

# Convert categorical variables if needed
if 'active_ingredient' in df.columns:
    df['active_ingredient'] = df['active_ingredient'].astype(str)

df.head()


# ==================================================
#   OLS Regression Model for a given active ingredient
# ==================================================
def fit_ols_model(df, ingredient, predictors, response="residue"):
    
    subset = df[df["active_ingredient"] == ingredient].copy()
    
    X = subset[predictors]
    y = subset[response]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    
    # Statsmodels OLS requires manual constant
    X_train_const = sm.add_constant(X_train)
    X_test_const = sm.add_constant(X_test)
    
    model = sm.OLS(y_train, X_train_const).fit()
    
    # Predictions
    y_pred = model.predict(X_test_const)
    
    # Metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    me = np.mean(y_pred - y_test)

    return model, r2, rmse, me, subset, y_test, y_pred


# Predictor variables based on the article's Table 3

predictors_by_ai = {
    "Trifloxystrobin": ["DAS", "SR", "P", "Tb"],
    "Fenpropimorph": ["RH", "SR", "DAS"],
    "Bixafen": ["RH", "SR", "DAS"],
    "Picoxystrobin": ["RH", "DAS"],
    "Epoxiconazole": ["RH", "SR", "DAS"],
    "Fluxapyroxad": ["RH", "P", "DAS"],
    "Pyraclostrobin": ["DAS"],   # later: power model included separately
    "Tebuconazole": ["Tmin", "DAS"]
}

results = {}

for ai, predictors in predictors_by_ai.items():
    model, r2, rmse, me, subset, y_test, y_pred = fit_ols_model(df, ai, predictors)
    results[ai] = {
        "model": model,
        "r2": r2,
        "rmse": rmse,
        "me": me,
        "obs": y_test,
        "pred": y_pred
    }

# Display results summary
for ai, res in results.items():
    print(f"--- {ai} ---")
    print(f"R²   = {res['r2']:.2f}")
    print(f"RMSE = {res['rmse']:.2f}")
    print(f"ME   = {res['me']:.2f}")
    print()


# ==================================================
#  Scatter plots comparing observed vs. predicted
# ==================================================
for ai, res in results.items():
    plt.figure()
    plt.scatter(res["obs"], res["pred"], alpha=0.7)
    plt.plot([res["obs"].min(), res["obs"].max()],
             [res["obs"].min(), res["obs"].max()],
             linestyle='--')
    plt.xlabel("Observed residue")
    plt.ylabel("Predicted residue")
    plt.title(f"Observed vs Predicted – {ai}")
    plt.grid(True)
    plt.show()


# ==================================================
#  Power model: R = a * (DAS ^ b)
# ==================================================

subset = df[df["active_ingredient"] == "Pyraclostrobin"].copy()

# Remove zeros or negatives
subset = subset[subset["residue"] > 0]
subset = subset[subset["DAS"] > 0]

# Log-transform for linearization:
subset["log_R"] = np.log(subset["residue"])
subset["log_DAS"] = np.log(subset["DAS"])

power_model = sm.OLS(subset["log_R"], sm.add_constant(subset["log_DAS"])).fit()
print(power_model.summary())



def compute_metrics(obs, pred):
    r2 = r2_score(obs, pred)
    me = np.mean(pred - obs)
    rmse = np.sqrt(mean_squared_error(obs, pred))
    return r2, me, rmse


# Export coefficients
coef_export = {}

for ai, res in results.items():
    coef_export[ai] = res["model"].params.to_dict()

pd.DataFrame(coef_export).to_csv("model_coefficients.csv")



import os

os.makedirs("figures", exist_ok=True)

for ai, res in results.items():
    plt.figure()
    plt.scatter(res["obs"], res["pred"], alpha=0.7)
    plt.plot([res["obs"].min(), res["obs"].max()],
             [res["obs"].min(), res["obs"].max()],
             linestyle='--')
    plt.xlabel("Observed residue")
    plt.ylabel("Predicted residue")
    plt.title(f"Observed vs Predicted – {ai}")
    plt.grid(True)
    plt.savefig(f"figures/validation_{ai}.png", dpi=320)
    plt.close()

```

