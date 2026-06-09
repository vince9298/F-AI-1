import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix

#definiamo le feature e dividiamo il dataset in training set e test set in base all'anno

features = ['Percentuale_acceleratore_premuto', 'Numero_cambi_marcia',
            'Numero_frenate', 'Percentuale_freno', 'Percentuale_acc_parzializzato']

df = pd.read_csv('dataset_circuiti_giri.csv')
train, test = df[df['Year'] == 2023], df[df['Year'] == 2024]

x_train, y_train = train[features], train['Track']
x_test, y_test = test[features], test['Track']

print(f"Giri Train (2023): {len(train)} | Giri Test (2024): {len(test)}\n")

#facciamo lo scaling dei dati per i modelli di regressione logistica e SVM

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#inizializziamo i modelli scelti

modelli = {
    "Regressione Logistica": LogisticRegression(max_iter=1000),
    "SVM": SVC(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}


migliore_nome, migliore_acc, miglior_modello = "", 0, None
risultati = {}

for nome, modello in modelli.items():
    print(f"--- Addestramento {nome} ---")
    modello.fit(x_train_scaled, y_train)
    y_pred = modello.predict(x_test_scaled)

    #calcoliamo le metriche utili al confronto e alla validazione dei modelli
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)


    risultati[nome] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1': f1}
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")


    if acc > migliore_acc:
        migliore_acc, migliore_nome, miglior_modello = acc, nome, modello


    # se il modello ha un coefficente di importanza delle feature allora calcoliamo il peso di ogni feature

    importances = None
    if hasattr(modello, 'feature_importances_'):
        importances = modello.feature_importances_
    elif hasattr(modello, 'coef_'):
        importances = np.abs(modello.coef_).mean(axis=0)

    if importances is not None:
        df_imp = pd.DataFrame({'Feature': features, 'Importanza': importances})
        df_imp = df_imp.sort_values(by='Importanza', ascending=True)

        plt.figure(figsize=(10, 5))
        plt.barh(df_imp['Feature'], df_imp['Importanza'], color='steelblue')
        plt.title(f"Importanza delle Feature - {nome}")
        plt.xlabel("Peso")
        plt.tight_layout()

        nome_file = f'importanza_feature_{nome.replace(" ", "_")}.png'
        plt.savefig(nome_file)
        plt.show()
        print(f"  -> Grafico salvato: {nome_file}\n")
    else:
        print("  -> Grafico importanza feature non disponibile\n")




df_risultati = pd.DataFrame(risultati).T

#grafico confronto metriche standard
ax = df_risultati.plot(kind='bar', figsize=(12, 6), ylim=(0, 1.25))
plt.title("Confronto Modelli")
plt.ylabel("Punteggio")
plt.savefig('confronto_modelli.png')
plt.xticks(rotation=0)


#per il modello migliore stampiamo a schermo tutte le metriche esatte per ogni circuito
y_pred_migliore = miglior_modello.predict(x_test_scaled)

print("--- Report Completo Modello Migliore ---")
print(classification_report(y_test, y_pred_migliore, zero_division=0))


#stampa matrice di confusione a schermo
classi = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred_migliore, labels=classi, normalize='true')

plt.figure(figsize=(10, 8))
sns.heatmap(cm, cmap='Blues', xticklabels=classi, yticklabels=classi, vmin=0, vmax=1)
plt.title(f"Matrice di Confusione - {migliore_nome}")
plt.xlabel("Predetto")
plt.ylabel("Reale")
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.show()