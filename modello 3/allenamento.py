import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

Posizione_dataset = 'f1_dataset.csv'


def carica_dati(path):
    df = pd.read_csv(path)

    features = ['Indice_Performance_Team', 'Stint', 'Vita_Gomma',
                     'Temperatura_Pista', 'Temperatura_Aria', 'Benzina_Stimata']

    df = df.dropna(subset=features + ['Ratio_Tempo_Giro', 'Mescola'])

    #Isolare le gare per gruppi ci permette di non dare al modello riferimenti su tutte le gare e verificare le reali capacitá predittive del modello
    df['Gruppo_Gara'] = df['Anno'].astype(str) + "_" + df['Circuito']

    #facciamo l encoding delle variabili categoriche
    df['Mescola_Codificata'] = df['Mescola'].astype('category').cat.codes
    df = pd.get_dummies(df, columns=['Circuito'], drop_first=False)

    feature_col = features + ['Mescola_Codificata'] + \
                   [c for c in df.columns if c.startswith('Circuito_')]

    return df[feature_col], df['Ratio_Tempo_Giro'], df['Gruppo_Gara']


def valuta_modelli(X_tv, y_tv, X_test, y_test):
    models = {
        'SVR': make_pipeline(StandardScaler(), SVR(C=10.0, epsilon=0.05)),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=8,
                                              random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05,
                                                      max_depth=5, random_state=42)
    }

    gkf = GroupKFold(n_splits=4)
    results = {}

    for name, model in models.items():
        print(f"\n[+] {name}")

        #cross-validation per gruppi
        oof = np.zeros(len(y_tv))
        for tr_idx, val_idx in gkf.split(X_tv, y_tv, gruppi):
            model.fit(X_tv.iloc[tr_idx], y_tv.iloc[tr_idx])
            oof[val_idx] = model.predict(X_tv.iloc[val_idx])


        model.fit(X_tv, y_tv)
        pred = model.predict(X_test)

        results[name] = {
            'r2_val': r2_score(y_tv, oof),
            'r2_test': r2_score(y_test, pred),
            'mae_test': mean_absolute_error(y_test, pred),
            'rmse_test': np.sqrt(mean_squared_error(y_test, pred)),
            'pred_test': pred,
            'model': model
        }

        print(f"    Val  R²: {results[name]['r2_val']:.4f}")
        print(f"    Test R²: {results[name]['r2_test']:.4f} | "
              f"MAE: {results[name]['mae_test']:.5f} | RMSE: {results[name]['rmse_test']:.5f}")

    return results


def grafici(X_test, y_test, results):
    best = max(results, key=lambda k: results[k]['r2_test'])


    # 1 il grafico di confronto del valore r^2
    names = sorted(results, key=lambda k: results[k]['r2_test'], reverse=True)
    plt.figure(figsize=(10, 5))
    plt.bar(names, [results[n]['r2_test'] for n in names],
            color='royalblue', edgecolor='black', alpha=0.8)
    plt.title("Comparazione modelli (R²)", fontweight='bold')
    plt.ylabel("R² Score")
    plt.grid(True, linestyle=':', alpha=0.5, axis='y')
    plt.tight_layout()
    plt.show()

    # 2 il grafico di importanza delle feature(utile per comprendere se il modello migliore ha compreso la relazione tra le feature che portano al valore target)
    modello = results[best]['model']
    importances = modello.feature_importances_
    idx = np.argsort(importances)[::-1][:10]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[idx], y=X_test.columns[idx],
                palette='plasma', hue=X_test.columns[idx], legend=False)
    plt.title(f'Top 10 Feature ({best})', fontweight='bold')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y, groups = carica_dati(Posizione_dataset)

    #isoliamo un gruppo pari al 20% come test set, in modo da validare le reali prestazioni del modello su dati mai visti
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tv_idx, test_idx = next(gss.split(X, y, groups))

    X_tv, X_test = X.iloc[tv_idx], X.iloc[test_idx]
    y_tv, y_test = y.iloc[tv_idx], y.iloc[test_idx]
    gruppi = groups.iloc[tv_idx]

    print(f"Train/Val: {len(X_tv)} righe ({gruppi.nunique()} GP) | "
          f"Test: {len(X_test)} righe ({groups.iloc[test_idx].nunique()} GP)")

    results = valuta_modelli(X_tv, y_tv, X_test, y_test)
    grafici(X_test, y_test, results)