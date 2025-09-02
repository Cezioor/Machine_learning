import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# File path
path = 'C:\\Users\\LT\\OneDrive\\dane_leki.xlsx'

# Loading data
data = pd.read_excel(path)

# Delete the unnecessary columns
data = data.drop(columns=['Nazwa', 'Zbiór', 'Usun'])

# Split on independent and dependent variables
X_unscaled = data.drop(columns=['logK HSA'])
Y = data['logK HSA']

# Scaling the independent variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_unscaled)

# PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Calculate cumulative explained variance
explained_variance = np.cumsum(pca.explained_variance_ratio_)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.33, random_state=42)

# 10-fold cross-validation setup
kf = KFold(n_splits=10, shuffle=True, random_state=0)
rmsec_scores = []

# Choosing the optimal number of components
for n_components in range(1, X_train.shape[1] + 1):
    pca_n = PCA(n_components=n_components)
    X_train_pca = pca_n.fit_transform(X_train)
    
    model = LinearRegression()
    mse = -cross_val_score(model, X_train_pca, Y_train, cv=kf, scoring='neg_mean_squared_error')
    rmse = np.sqrt(mse.mean())
    
    rmsec_scores.append(rmse)

# Graph RMSEC vs number of components
plt.figure(figsize=(10, 6))
plt.plot(range(1, X_train.shape[1] + 1), rmsec_scores, marker='o', linestyle='-')
plt.xlabel('Liczba głównych składowych')
plt.ylabel('RMSE')
plt.title('Zależność RMSEC od liczby głównych składowych')
plt.grid(True)
plt.show()

# Choosing the optimal number of components
optimal_components = np.argmin(rmsec_scores) + 1
print(f"Optymalna liczba głównych składowych: {optimal_components}")

# Training the final model with the optimal number of components
pca_optimal = PCA(n_components=optimal_components)
X_train_optimal = pca_optimal.fit_transform(X_train)
X_test_optimal = pca_optimal.transform(X_test)

model = LinearRegression()
model.fit(X_train_optimal, Y_train)

# Predictions on training and testing sets
Y_train_pred = model.predict(X_train_optimal)
Y_test_pred = model.predict(X_test_optimal)

# Calculating RMSE and R²
rmse_train = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
r2_train = r2_score(Y_train, Y_train_pred)

rmse_test = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
r2_test = r2_score(Y_test, Y_test_pred)

print(f"RMSE (Train): {rmse_train:.4f}")
print(f"R2 (Train): {r2_train:.4f}")
print(f"RMSE (Test): {rmse_test:.4f}")
print(f"R2 (Test): {r2_test:.4f}")

print("\nInterpretacja wyników:")
print(f"Model osiągnął RMSE = {rmse_train:.4f} na zbiorze treningowym i RMSE = {rmse_test:.4f} na zbiorze testowym.")
print(f"Wartość R² na zbiorze treningowym wynosi {r2_train:.4f}, a na zbiorze testowym {r2_test:.4f}.")
if r2_test > 0.7:
    print("Otrzymane wartości R² sugerują, że model dobrze dopasowuje się do danych i ma dobrą zdolność generalizacji.")
elif 0.4 < r2_test <= 0.7:
    print("Model ma umiarkowaną zdolność predykcji. Możliwe, że dodatkowa optymalizacja poprawi wyniki.")
else:
    print("Niska wartość R² wskazuje na słabe dopasowanie modelu.")