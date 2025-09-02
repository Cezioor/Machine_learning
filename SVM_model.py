import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import mlxtend

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from mlxtend.plotting import plot_decision_regions





# 1) Wczytanie zbioru danych

data_path = '/Users/LT/Desktop/uczenie/penguins_size.csv'

df = pd.read_csv(data_path)



# 2) Usunięcie brakujących wartości

df = df.dropna()



# 3) Wykresy zależności zmiennych

sns.pairplot(df, hue='species')

plt.show()



# 4) Wybór dwóch cech i dwóch gatunków

df_binary_lin = df[df['species'].isin(['Adelie', 'Chinstrap'])][['bill_length_mm', 'bill_depth_mm', 'species']]

df_binary_nonlin = df[df['species'].isin(['Adelie', 'Gentoo'])][['bill_length_mm', 'body_mass_g', 'species']]



def encode_species(df):

    df['species'] = df['species'].astype('category').cat.codes

    return df



df_binary_lin = encode_species(df_binary_lin)

df_binary_nonlin = encode_species(df_binary_nonlin)



# 5) Model dla liniowo separowalnych danych

X_lin = df_binary_lin[['bill_length_mm', 'bill_depth_mm']]

y_lin = df_binary_lin['species']

X_train, X_test, y_train, y_test = train_test_split(X_lin, y_lin, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



svm_lin = SVC(kernel='linear')

svm_lin.fit(X_train_scaled, y_train)

y_pred = svm_lin.predict(X_test_scaled)

print(f'Accuracy (linear, separable): {accuracy_score(y_test, y_pred)}')



# Wykres dla separowalnych danych

plt.figure()

plot_decision_regions(X_train_scaled, y_train.to_numpy(), clf=svm_lin)

plt.title("Decision Boundary for Linear SVM")

plt.xlabel("Bill Length (scaled)")

plt.ylabel("Bill Depth (scaled)")

plt.show()



# 6) Modele dla nieseparowalnych danych

X_nonlin = df_binary_nonlin[['bill_length_mm', 'body_mass_g']]

y_nonlin = df_binary_nonlin['species']

X_train, X_test, y_train, y_test = train_test_split(X_nonlin, y_nonlin, test_size=0.2, random_state=42)

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)



kernels = ['linear', 'poly', 'rbf']

for kernel in kernels:

    svm_model = SVC(kernel=kernel)

    svm_model.fit(X_train_scaled, y_train)

    y_pred = svm_model.predict(X_test_scaled)

    print(f'Accuracy ({kernel}, non-separable): {accuracy_score(y_test, y_pred)}')



# Wykres dla nieseparowalnych danych (linear)

plt.figure()

plot_decision_regions(X_train_scaled, y_train.to_numpy(), clf=SVC(kernel='linear').fit(X_train_scaled, y_train))

plt.title("Decision Boundary for Non-linear SVM (Linear Kernel)")

plt.xlabel("Bill Length (scaled)")

plt.ylabel("Body Mass (scaled)")

plt.show()

