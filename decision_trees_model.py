print("Zadanie 1")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (

    classification_report,

    confusion_matrix,

    accuracy_score,

    balanced_accuracy_score,

    precision_score,

    recall_score,

    f1_score

)

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

 

# Wczytanie danych

file_path = "/Users/LT/Desktop/uczenie/ftalany_klasyfikacja.xlsx"

X_train = pd.read_excel(file_path, sheet_name="X_train")

y_train = pd.read_excel(file_path, sheet_name="y_train")

X_test = pd.read_excel(file_path, sheet_name="X_test")

y_test = pd.read_excel(file_path, sheet_name="y_test")

 

# ** 1. Macierz korelacji **

train_data = X_train.copy()

train_data['Category'] = y_train

 

correlation_matrix = train_data.corr()

plt.figure(figsize=(8, 6))

sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")

plt.title("Macierz korelacji")

plt.show()

 

# ** 2. Sprawdzenie balansu zbiorów **

print("Balans zbioru treningowego:\n", y_train.value_counts(normalize=True))

print("Balans zbioru testowego:\n", y_test.value_counts(normalize=True))

 

# ** 3. Budowa modelu drzewa klasyfikacyjnego **

clf = DecisionTreeClassifier(random_state=42)

clf.fit(X_train, y_train)

 

# Predykcja

y_pred = clf.predict(X_test)

 

# Macierz pomyłek

conf_matrix = confusion_matrix(y_test, y_pred)

sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")

plt.title("Macierz pomyłek")

plt.xlabel("Predykcja")

plt.ylabel("Rzeczywistość")

plt.show()

 

# ** Ocena modelu **

accuracy = accuracy_score(y_test, y_pred)

balanced_acc = balanced_accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

 

print("Dokładność:", accuracy)

print("Zbalansowana dokładność:", balanced_acc)

print("Precyzja:", precision)

print("Czułość (Recall):", recall)

print("F1-score:", f1)

# ** Interpretacja wyników **
print("Interpretacja wyników klasyfikacji:")

print("Dokładność informuje o ogólnym dopasowaniu modelu, jednak może być myląca w przypadku niezrównoważonych zbiorów danych.")

print("Zbalansowana dokładność jest bardziej odpowiednia, gdy klasy są nierównomiernie reprezentowane.")

print("Precyzja mówi, ile z przewidzianych pozytywnych przypadków rzeczywiście należy do tej klasy.")

print("Czułość (Recall) określa, jak dobrze model wykrywa prawdziwe przypadki pozytywne.")

print("F1-score to harmoniczna średnia precyzji i czułości, przydatna przy niezrównoważonych zbiorach danych.")

# ** Ocena jakości modelu **
if accuracy > 0.8:
    print("Model dobrze się dopasował.")
elif accuracy > 0.6:
    print("Model ma średnie dopasowanie, warto spróbować lepszego dostrojenia parametrów.")
else:
    print("Model nie jest dobrze dopasowany, konieczne są poprawki.")

 

# ** 4. Optymalizacja parametrów **

param_grid = {

    "max_depth": [3, 5, 10, None],

    "min_samples_split": [2, 5, 10],

    "min_samples_leaf": [1, 2, 5]

}

 

# Metoda 1: Przeszukiwanie GridSearchCV

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)

grid_search.fit(X_train, y_train)

print("Najlepsze parametry GridSearchCV:", grid_search.best_params_)

 

# Metoda 2: RandomizedSearchCV (szybsza, testuje losowe kombinacje)

random_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), param_grid, n_iter=5, cv=5, random_state=42)

random_search.fit(X_train, y_train)

print("Najlepsze parametry RandomizedSearchCV:", random_search.best_params_)















 
print("Zadanie 2")
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.tree import DecisionTreeRegressor, plot_tree

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

 

# Wczytanie danych

file_path = "/big/home/cezioor/Desktop/uczenie_maszynowe/ftalany.xlsx"

X_train = pd.read_excel(file_path, sheet_name="X_train")

y_train = pd.read_excel(file_path, sheet_name="y_train")

X_test = pd.read_excel(file_path, sheet_name="X_test")

y_test = pd.read_excel(file_path, sheet_name="y_test")

 

# Konwersja y_train i y_test na tablice jednowymiarowe
X_train = X_train.values

y_train = y_train.values.ravel()

X_test = X_test.values

y_test = y_test.values.ravel()

 

# ** 1. Budowa modelu drzewa regresyjnego **

regressor = DecisionTreeRegressor(
    random_state=42,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=3
)

regressor.fit(X_train, y_train)

 

# Predykcja

y_pred_train = regressor.predict(X_train)

y_pred_test = regressor.predict(X_test)

 

# ** 2. Obliczenie statystyk **

R2_train = r2_score(y_train, y_pred_train)

R2_test = r2_score(y_test, y_pred_test)

RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))

RMSE_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

Q2 = r2_score(y_test, y_pred_test)

RMSE_EX = np.sqrt(mean_squared_error(y_test, y_pred_test))
 

print(f"R2 (train): {R2_train:.4f}, R2 (test): {R2_test:.4f}")

print(f"RMSE (train): {RMSE_train:.4f}, RMSE (test): {RMSE_test:.4f}")

print(f"Q2: {Q2}")

print(f"RMSEex: {RMSE_EX}")

#  Ocena jakości modelu regresyjnego 
if R2_train > 0.8:
    print("Model dobrze się dopasował.")
elif R2_train > 0.6:
    print("Model ma średnie dopasowanie, warto poprawić parametry.")
else:
    print("Model słabo się dopasował, konieczne są zmiany.")


 

# ** 3. Wykres y_pred vs y_obs **

plt.figure(figsize=(6, 6))

plt.scatter(y_test, y_pred_test, color="blue", alpha=0.7, label="Predykcje")

plt.plot(y_test, y_test, color="red", linestyle="--", label="Idealny Fit")

plt.xlabel("Wartości rzeczywiste (y_obs)")

plt.ylabel("Wartości przewidywane (y_pred)")

plt.title("Porównanie wartości rzeczywistych i przewidywanych y_pred vs y_obs")

plt.legend()

plt.grid()

plt.show()



# Wykres słupkowy

plt.figure(figsize=(10, 6))

plt.bar(range(len(y_test)), y_test, alpha=0.7, label="y_obs")

plt.bar(range(len(y_pred_test)), y_pred_test, alpha=0.7, label="y_pred")

plt.xlabel("Próbki")

plt.ylabel("Wartości")

plt.title("Wykres słupkowy: y_pred vs y_obs")

plt.legend()

plt.show()

 
param_grid = {
    "criterion": ["squared_error", "friedman_mse"],
    "max_depth": range(1, 20),
    "min_samples_split": range(2, 10),
    "min_samples_leaf": range(1, 10)
}


# 1) Manual search
best_model_manual = None
best_score_manual = float('-inf')
best_params_manual = {}

for criterion in param_grid["criterion"]:
    for max_depth in param_grid["max_depth"]:
        for min_samples_split in param_grid["min_samples_split"]:
            for min_samples_leaf in param_grid["min_samples_leaf"]:
                model = DecisionTreeRegressor(
                    random_state=42,
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf
                )
                model.fit(X_train, y_train)
                score = r2_score(y_test, model.predict(X_test))

                if score > best_score_manual:
                    best_score_manual = score
                    best_model_manual = model
                    best_params_manual = {
                        "criterion": criterion,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "min_samples_leaf": min_samples_leaf
                    }

print(f"\nNajlepsze parametry (Manual search):\n {best_params_manual}")
print(f"R2 (Manual search): {best_score_manual}")

# Poprawiona siatka parametrów
param_distributions = {
    "criterion": ["squared_error", "friedman_mse"],
    "max_depth": np.arange(3, 20, 2),  # co 2, żeby nie było zbyt dużego przeszukiwania
    "min_samples_split": np.arange(2, 10),
    "min_samples_leaf": np.arange(1, 10) }

# 2) Randomized Search
random_search = RandomizedSearchCV(
    estimator=DecisionTreeRegressor(random_state=42),
    param_distributions=param_grid,
    n_iter=20,
    cv=5,
    scoring="neg_mean_squarred_error", # Zmiana z R2 na neg_mean_squarred_error bo wychodza absurdalne wartosci R2
    random_state=42
)
random_search.fit(X_train, y_train)

best_params_random = random_search.best_params_
best_score_random = random_search.best_score_

print(f"\nNajlepsze parametry (Randomized search):\n{best_params_random}")
print(f"R2 (Randomized search): {best_score_random}")

# Wizualizacja drzewa regresyjnego
plt.figure(figsize=(12, 6))
plot_tree(
    best_model_manual,
    feature_names=[f"Feature_{i}" for i in range(X_train.shape[1])],
    filled=True,
    rounded=True
)
plt.title("Drzewo regresyjne")
plt.show()