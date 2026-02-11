# train.py
import pandas as pd
from pycaret.classification import *

# ========================
# 1️ Load and clean data
# ========================
data = pd.read_csv("data/churn_data.csv")

# Drop ID column
if "customerID" in data.columns:
    data.drop(columns=["customerID"], inplace=True)

# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce")

# Drop rows with missing target
data.dropna(subset=["Churn"], inplace=True)

# ========================
# 2️ Setup PyCaret
# ========================
clf = setup(
    data=data,
    target="Churn",
    session_id=42,

    # Preprocessing
    normalize=True,
    remove_outliers=True,

    # Feature types
    categorical_features=[
        "PhoneService",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod"
    ],
    numeric_features=[
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ],

    # Handle imbalance
    fix_imbalance=True,
    fix_imbalance_method="SMOTE",

    # Cross-validation
    fold=5,
    fold_strategy="stratifiedkfold",

    # Silent execution
    verbose=False
)

# ========================
# 3️ Compare models (F1-score)
# ========================
best_model = compare_models(sort="F1")

# ========================
# 4️ Tune best model (F1-score)
# ========================
tuned_model = tune_model(
    best_model,
    optimize="F1",
    choose_better=True
)

# ========================
# 5️ Finalize model
# ========================
final_model = finalize_model(tuned_model)

# ========================
# 6️ Save model
# ========================
save_model(final_model, "models/churn_model")

