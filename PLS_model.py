import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Excel file path
sciezka = '/Users/LT/Desktop/uczenie/AA-AuNP.xlsx'
plik_excel = pd.ExcelFile(sciezka)

# Loading data from Excel sheets
Xt = plik_excel.parse('Xt')
Yt = plik_excel.parse('Yt')
Xv = plik_excel.parse('Xv')
Yv = plik_excel.parse('Yv')

# Choosing features
wybrane_cechy = ["HOMO[eV]", "Polarizability [au]", "Topological polar surface area.1"]
Xt_wybrane = Xt[wybrane_cechy]
Xv_wybrane = Xv[wybrane_cechy]


y_trenowanie = Yt["AA-AuNP"]
y_walidacja = Yv["AA-AuNP"]

# Scale
skalowacz_X = StandardScaler()
skalowacz_y = StandardScaler()

Xt_skalowane = skalowacz_X.fit_transform(Xt_wybrane)
Xv_skalowane = skalowacz_X.transform(Xv_wybrane)
y_trenowanie_skalowane = skalowacz_y.fit_transform(y_trenowanie.values.reshape(-1, 1)).ravel()
y_walidacja_skalowane = skalowacz_y.transform(y_walidacja.values.reshape(-1, 1)).ravel()

# Model PLS
pls = PLSRegression(n_components=1)
pls.fit(Xt_skalowane, y_trenowanie_skalowane)

y_pred_trenowanie_pls = skalowacz_y.inverse_transform(pls.predict(Xt_skalowane).reshape(-1, 1)).ravel()
y_pred_walidacja_pls = skalowacz_y.inverse_transform(pls.predict(Xv_skalowane).reshape(-1, 1)).ravel()

# Model MLR
mlr = LinearRegression()
mlr.fit(Xt_skalowane, y_trenowanie_skalowane)

y_pred_trenowanie_mlr = skalowacz_y.inverse_transform(mlr.predict(Xt_skalowane).reshape(-1, 1)).ravel()
y_pred_walidacja_mlr = skalowacz_y.inverse_transform(mlr.predict(Xv_skalowane).reshape(-1, 1)).ravel()

# Model PCR
pca = PCA(n_components=2)
Xt_pca = pca.fit_transform(Xt_skalowane)
Xv_pca = pca.transform(Xv_skalowane)

pcr = LinearRegression()
pcr.fit(Xt_pca, y_trenowanie_skalowane)

y_pred_trenowanie_pcr = skalowacz_y.inverse_transform(pcr.predict(Xt_pca).reshape(-1, 1)).ravel()
y_pred_walidacja_pcr = skalowacz_y.inverse_transform(pcr.predict(Xv_pca).reshape(-1, 1)).ravel()

# Calculating metrics
wyniki = {
    "PLS": (r2_score(y_trenowanie, y_pred_trenowanie_pls), r2_score(y_walidacja, y_pred_walidacja_pls),
             np.sqrt(mean_squared_error(y_trenowanie, y_pred_trenowanie_pls)),
             np.sqrt(mean_squared_error(y_walidacja, y_pred_walidacja_pls))),
    "MLR": (r2_score(y_trenowanie, y_pred_trenowanie_mlr), r2_score(y_walidacja, y_pred_walidacja_mlr),
             np.sqrt(mean_squared_error(y_trenowanie, y_pred_trenowanie_mlr)),
             np.sqrt(mean_squared_error(y_walidacja, y_pred_walidacja_mlr))),
    "PCR": (r2_score(y_trenowanie, y_pred_trenowanie_pcr), r2_score(y_walidacja, y_pred_walidacja_pcr),
             np.sqrt(mean_squared_error(y_trenowanie, y_pred_trenowanie_pcr)),
             np.sqrt(mean_squared_error(y_walidacja, y_pred_walidacja_pcr)))
}


for model, (r2_train, r2_test, rmse_train, rmse_test) in wyniki.items():
    print(f"\nModel {model}:")
    print(f"  R² (trening): {r2_train:.3f}")
    print(f"  R² (walidacja): {r2_test:.3f}")
    print(f"  RMSE (trening): {rmse_train:.3f}")
    print(f"  RMSE (walidacja): {rmse_test:.3f}")
    
    if r2_test > 0.7:
        print("  Model dobrze generalizuje dane.")
    elif 0.4 < r2_test <= 0.7:
        print("  Model ma umiarkowaną zdolność predykcji. Możliwa poprawa.")
    else:
        print("  Model nie dopasowuje się dobrze.")

# Show plots
plt.figure(figsize=(8, 6))
plt.scatter(y_walidacja, y_pred_walidacja_pls, label="PLS", marker='o')
plt.scatter(y_walidacja, y_pred_walidacja_mlr, label="MLR", marker='s')
plt.scatter(y_walidacja, y_pred_walidacja_pcr, label="PCR", marker='^')
plt.plot([y_walidacja.min(), y_walidacja.max()], [y_walidacja.min(), y_walidacja.max()], 'k--', lw=2)
plt.xlabel("Wartości eksperymentalne")
plt.ylabel("Wartości przewidywane")
plt.title("Porównanie modeli: wartości eksperymentalne vs przewidywane")
plt.legend()
plt.grid()
plt.show()
