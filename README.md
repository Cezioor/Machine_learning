# Machine Learning – regression and classification on chemistry data

A set of machine learning models I built during my studies to predict properties of chemical compounds.
Each script loads a dataset from Excel, trains a model, validates it and plots the results.

## Models

| Script | Method | Task | Validation |
|---|---|---|---|
| `MLR_model.py` | Multiple linear regression (scikit-learn, statsmodels) | Regression | Cross-validation, Williams plot (applicability domain), predicted vs observed |
| `PCR_model.py` | Principal component regression | Regression | K-fold CV to choose the number of components |
| `PLS_model.py` | Partial least squares regression | Regression | R² and RMSE on train and test sets |
| `KNN_model.py` | k-nearest neighbours | Regression | K-fold CV |
| `SVM_model.py` | Support vector machine | Classification | Accuracy, decision-region plots (mlxtend) |
| `decision_trees_model.py` | Decision tree | Classification | Grid and randomized hyperparameter search |

## Data

Small QSAR-style datasets where molecular descriptors (e.g. HOMO/LUMO energies, ionization potential, hardness) are used to predict a measured property:

| File | Compounds | Target |
|---|---|---|
| `ftalany.xlsx`, `ftalany_klasyfikacja.xlsx` | 32 phthalates | log k (reaction rate constant) |
| `dane_leki.xlsx` | 27 drugs | log K binding to human serum albumin (HSA) |
| `AA-AuNP.xlsx` | 25 amino acids | Interaction with gold nanoparticles |

Most files are already split into training and test sheets.
`MLR teoria.pdf` contains the theory behind the regression model.

## Run

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn mlxtend openpyxl
python MLR_model.py
```

## Stack

Python · pandas · NumPy · scikit-learn · statsmodels · Matplotlib · Seaborn
