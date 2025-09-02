import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict

def wykres_Williamsa(X_train, Y_train, X_test, Y_test):
    """Tworzy wykres Williamsa: dźwignia vs reszty ustandaryzowane"""

    # Calculating leverage and residuals for training set
    X_train_with_const = sm.add_constant(X_train)
    ols_model_train = sm.OLS(Y_train, X_train_with_const).fit()
    influence_train = ols_model_train.get_influence()
    leverage_train = influence_train.hat_matrix_diag
    standardized_residuals_train = influence_train.resid_studentized_internal

    # Calculating leverage and residuals for test set
    X_test_with_const = sm.add_constant(X_test)
    ols_model_test = sm.OLS(Y_test, X_test_with_const).fit()
    influence_test = ols_model_test.get_influence()
    leverage_test = influence_test.hat_matrix_diag
    standardized_residuals_test = influence_test.resid_studentized_internal

    # Calculated leverage limit
    n, p = X_train.shape
    h_limit = 3 * (p + 1) / n

    # Graphs
    plt.figure(figsize=(8, 6))
    plt.scatter(leverage_train, standardized_residuals_train, label="Zbiór uczący", color='orange')
    plt.scatter(leverage_test, standardized_residuals_test, label="Zbiór walidacyjny", color='green')

    # Lines 
    plt.axhline(y=0, color='black', linestyle='--')
    plt.axvline(x=h_limit, color='red', linestyle='--', label=f"Granica dźwigni = {h_limit:.3f}")
    plt.axhline(y=3, color='blue', linestyle='--', label="Granica ±3 odch. stand.")
    plt.axhline(y=-3, color='blue', linestyle='--')

    plt.xlabel("Dźwignia")
    plt.ylabel("Reszty ustandaryzowane")
    plt.title("Wykres Williamsa")
    plt.legend()
    plt.grid(True)
    plt.show()

# Predictions
def wykres_ypred_yobs(model, X_matrix, Y_vector):
    """Tworzy wykres wartości eksperymentalnych vs przewidywanych"""

    Y_obs_train = Y_vector.iloc[:19]
    Y_obs_test = Y_vector.iloc[19:]

    intercept = model.intercept_[0]
    coefficients = model.coef_[0]

    Y_pred_train = []
    Y_pred_test = []
    for idx in range(len(Y_vector)):
        y = intercept
        for i, key in enumerate(X_matrix):
            y += coefficients[i] * X_matrix[key][idx]
        if idx < 19:
            Y_pred_train.append(y)
        else:
            Y_pred_test.append(y)

    plt.figure(figsize=(8, 6))
    plt.scatter(Y_obs_test, Y_pred_test, color='green', label='Zbiór walidacyjny')
    plt.scatter(Y_obs_train, Y_pred_train, color='orange', label='Zbiór uczący')
    plt.plot([Y_vector.min(), Y_vector.max()], [Y_vector.min(), Y_vector.max()], 'k--', lw=2)

    plt.legend()
    plt.xlabel('Wartości eksperymentalne')
    plt.ylabel('Wartości przewidywane')
    plt.title('Wykres wartości eksperymentalnych vs przewidywanych')
    plt.grid(True)
    plt.show()

# Data loader
def data_loader(path: str):
    df = pd.read_excel(path)
    Y_vector = df[['logK HSA']]
    Y_vector_train = Y_vector.iloc[:19]
    Y_vector_test = Y_vector.iloc[19:]

    X_matrix = df[['logKCTAB', 'CATS3D_00_DD', 'CATS3D_09_AL', 'CATS3D_00_AA']]
    X_matrix_train = X_matrix.iloc[:19]
    X_matrix_test = X_matrix.iloc[19:]

    return Y_vector, X_matrix, Y_vector_train, Y_vector_test, X_matrix_train, X_matrix_test

# correlation matrix
def corr_matrix(X_matrix):
    corr = X_matrix.corr()
    plt.figure(figsize=(4, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Macierz korelacji")
    plt.show()
    return corr

# regression model
def rownanie_modelu(X_matrix, Y_vector):
    model = LinearRegression()
    model.fit(X_matrix, Y_vector)

    intercept = model.intercept_
    coefficients = model.coef_[0]

    print(f'\n\nRównanie modelu:')
    print(f'{intercept[0]:.2f} + {coefficients[0]:.2f}*logKCTAB + {coefficients[1]:.2f}*CATS3D_00_DD + {coefficients[2]:.2f}*CATS3D_09_AL + {coefficients[3]:.2f}*CATS3D_00_AA + ϵ')

    return model


if __name__ == '__main__':
    path = 'C:\\Users\\LT\\OneDrive\\dane_leki.xlsx'

    # Loading data
    Y_vector, X_matrix, Y_train, Y_test, X_train, X_test = data_loader(path)

    # correlation matrix
    print(corr_matrix(X_train))

    # Training the model
    model = rownanie_modelu(X_train, Y_train)

    # Predictions and plots
    wykres_ypred_yobs(model, X_matrix, Y_vector)
    wykres_Williamsa(X_train, Y_train, X_test, Y_test)

    # Calculating metrics
    Y_train_pred = model.predict(X_train)
    Y_test_pred = model.predict(X_test)

    rmse_train = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
    r2_train = r2_score(Y_train, Y_train_pred)

    rmse_test = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
    r2_test = r2_score(Y_test, Y_test_pred)

    print(f"\nRMSE (Train): {rmse_train:.4f}")
    print(f"R² (Train): {r2_train:.4f}")
    print(f"RMSE (Test): {rmse_test:.4f}")
    print(f"R² (Test): {r2_test:.4f}")

  
    print("\nInterpretacja wyników:")
    print(f"Model osiągnął RMSE = {rmse_train:.4f} na zbiorze treningowym i RMSE = {rmse_test:.4f} na zbiorze testowym.")
    print(f"Wartość R² na zbiorze treningowym wynosi {r2_train:.4f}, a na zbiorze testowym {r2_test:.4f}.")

    if r2_test > 0.7:
        print("Model dobrze dopasowuje się do danych i ma dobrą zdolność generalizacji.")
    elif 0.4 < r2_test <= 0.7:
        print("Model ma umiarkowaną zdolność predykcji. Możliwe, że dodatkowa optymalizacja poprawi wyniki.")
    else:
        print("Niska wartość R² wskazuje na słabe dopasowanie modelu. Może być konieczne użycie innego modelu lub lepsza selekcja cech.")