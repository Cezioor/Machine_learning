import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
file = pd.ExcelFile('/big/home/cezioor/Desktop/uczenie maszynowe/ftalany.xlsx')
Xt = pd.read_excel(file, 'X_train').values
Yt = pd.read_excel(file, 'y_train').values.ravel()
Xv = pd.read_excel(file, 'X_test').values
Yv = pd.read_excel(file, 'y_test').values.ravel()

# Parameters 
n_splits = 2
kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

max_k = min(len(Xt) // n_splits, len(Xt))
k_values = range(1, max_k + 1)

rmse_avg = []

# Cross-validation to find optimal k
for k in k_values:
    rmse_values = []
    knn = KNeighborsRegressor(n_neighbors=k)
    fold_rmse = []
    for train_index, val_index in kf.split(Xt):
        X_train, X_val = Xt[train_index], Xt[val_index]
        y_train, y_val = Yt[train_index], Yt[val_index]

        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        rmse_values.append(rmse)

    avg_rmse = np.mean(rmse_values)
    rmse_avg.append(avg_rmse)

# optimal k
k_opt = np.argmin(rmse_avg) + 1
print(f"Optymalne k: {k_opt}")

# Graph k vs RMSE on fig
fig = plt.figure(figsize=(10, 6))
plt.plot(k_values, rmse_avg, marker='o', linestyle='-', color='b')
plt.title('Zależność k od RMSE', fontsize=14)
plt.xlabel('k', fontsize=12)
plt.ylabel('Średni RMSE', fontsize=12)
plt.xticks(k_values)
plt.grid(alpha=0.5)
plt.tight_layout()  
plt.savefig('rmse_vs_k.png') 
plt.close()

# Function to calculate stats for different distance metrics
def calculate_stats(knn_type):
    distance_metrics = ['euclidean', 'manhattan', 'chebyshev', 'canberra']
    stats = {'Metric': [], 'R^2': [], 'RMSE': [], 'Q^2': [], 'RMSEex': []}
    for metric in distance_metrics:
        rmse_values = []
        r2_values = []
        q2_values = []
        rmseex_values = []
        for train_index, val_index in kf.split(Xt):
            X_train, X_val = Xt[train_index], Xt[val_index]
            y_train, y_val = Yt[train_index], Yt[val_index]
            
            if knn_type == 'classic':
                knn = KNeighborsRegressor(n_neighbors=k_opt, metric=metric)
            elif knn_type == 'weighted':
                knn = KNeighborsRegressor(n_neighbors=k_opt, metric=metric, weights='distance')

            knn.fit(X_train, y_train)
            y_pred_v = knn.predict(X_val)
            y_pred_t = knn.predict(X_train)

            # RMSE
            rmse = np.sqrt(mean_squared_error(y_train, y_pred_t))
            rmse_values.append(rmse)

            # R^2
            r2 = r2_score(y_train, y_pred_t)
            r2_values.append(r2)
            
            # Q^2
            y_obs = np.append(y_train, y_val)
            y_pred = np.append(y_pred_t, y_pred_v)
            sr_obs = np.mean(y_obs)
            q2 = r2_score(y_val, y_pred_v)
            q2_values.append(q2)

            # RMSEex
            rmseex = np.sqrt(np.sum((y_val - y_pred_v)**2) / len(y_val))
            rmseex_values.append(rmseex)

        stats['Metric'].append(metric)
        stats['R^2'].append(np.mean(r2_values))
        stats['RMSE'].append(np.mean(rmse_values))
        stats['Q^2'].append(np.mean(q2_values))
        stats['RMSEex'].append(np.mean(rmseex_values))
    
    return stats

# Calculate stats for classic and weighted KNN
stat_classic = calculate_stats('classic')
stat_weighted = calculate_stats('weighted')

# Type and display stats
print("\nStatystyki dla KNN (klasycznego) dla różnych miar dystansu:")
stat_df_classic = pd.DataFrame(stat_classic)
print(stat_df_classic)

print("\nStatystyki dla KNN (ważonego) dla różnych miar dystansu:")
stat_df_weighted = pd.DataFrame(stat_weighted)
print(stat_df_weighted)

# Plot stats comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Classic KNN
stat_df_classic.set_index('Metric')[['RMSE', 'R^2', 'Q^2', 'RMSEex']].plot(kind='bar', ax=axes[0], color=['red', 'green', 'blue', 'orange'])
axes[0].set_title("Statystyki KNN (klasycznego) dla różnych miar dystansu", fontsize=14)
axes[0].set_ylabel("Wartości", fontsize=12)
axes[0].set_xlabel("Miary odległości", fontsize=12)

# Weighted KNN
stat_df_weighted.set_index('Metric')[['RMSE', 'R^2', 'Q^2', 'RMSEex']].plot(kind='bar', ax=axes[1], color=['red', 'green', 'blue', 'orange'])
axes[1].set_title("Statystyki KNN (ważonego) dla różnych miar dystansu", fontsize=14)
axes[1].set_ylabel("Wartości", fontsize=12)
axes[1].set_xlabel("Miary odległości", fontsize=12)

plt.tight_layout()
plt.savefig('knn_stats_comparison.png') 
plt.close()

