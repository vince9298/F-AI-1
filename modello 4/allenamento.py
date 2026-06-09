import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, classification_report, confusion_matrix, precision_recall_fscore_support


print("Caricamento dei dati da 'f1_dataset.csv'...")
df = pd.read_csv('f1_dataset.csv')

df['Weekend_Gara'] = df['Anno'].astype(str) + "_" + df['Circuito']

def assegna_classe_passo(ratio):
    if ratio < 1.03:
        return 0
    elif ratio <= 1.07:
        return 1
    else:
        return 2

df['Classe_Passo'] = df['Ratio_Tempo_Giro'].apply(assegna_classe_passo)

df['Mescola'] = df['Mescola'].map({'Soft': 0, 'Medium': 1, 'Hard': 2})

df = pd.get_dummies(df, columns=['Circuito'])

colonne_circuito = [col for col in df.columns if col.startswith('Circuito_')]
colonne_finali_modello = ['Indice_Performance_Team', 'Vita_Gomma', 'Temperatura_Pista',
                          'Temperatura_Aria', 'Benzina_Stimata', 'Mescola'] + colonne_circuito

X_totale = df[colonne_finali_modello].copy()
Y_totale = df['Classe_Passo'].values
Gruppi_totali = df['Weekend_Gara'].values

colonne_numeriche = ['Indice_Performance_Team', 'Vita_Gomma', 'Temperatura_Pista', 'Temperatura_Aria', 'Benzina_Stimata']

#dividiamo i dati in training set e test set, divisi giá per weekend di gara
divisione_iniziale = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
indici_addestramento, indici_test = next(divisione_iniziale.split(X_totale, Y_totale, groups=Gruppi_totali))

X_addestramento = X_totale.iloc[indici_addestramento]
Y_addestramento = Y_totale[indici_addestramento]
Gruppi_addestramento = Gruppi_totali[indici_addestramento]

X_test = X_totale.iloc[indici_test]
Y_test_veri = Y_totale[indici_test]

modello_classico = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)
modello_ssl = SelfTrainingClassifier(modello_classico, criterion='threshold', threshold=0.90)


print("\nInizio addestramento e mascheramento (85% dei dati nascosti)...")

risultati_classico = []
risultati_ssl = []
valutatore_fold = GroupKFold(n_splits=4)

for round_n, (indici_train_fold, indici_val_fold) in enumerate(valutatore_fold.split(X_addestramento, Y_addestramento, groups=Gruppi_addestramento)):

    X_train_f = X_addestramento.iloc[indici_train_fold].copy()
    Y_train_veri_f = Y_addestramento[indici_train_fold].copy()

    X_val_f = X_addestramento.iloc[indici_val_fold].copy()
    Y_val_veri_f = Y_addestramento[indici_val_fold]

    scaler = StandardScaler()
    X_train_f[colonne_numeriche] = scaler.fit_transform(X_train_f[colonne_numeriche])
    X_val_f[colonne_numeriche] = scaler.transform(X_val_f[colonne_numeriche])

    np.random.seed(42 + round_n)
    numero_righe_studio = len(Y_train_veri_f)

    # togliamo l'etichetta dall 85% dei dati
    indici_da_nascondere = np.random.choice(numero_righe_studio, size=int(numero_righe_studio * 0.85), replace=False)

    Y_train_mascherati = Y_train_veri_f.copy()
    Y_train_mascherati[indici_da_nascondere] = -1

    solo_righe_etichettate = (Y_train_mascherati != -1)

    modello_classico.fit(X_train_f[solo_righe_etichettate], Y_train_mascherati[solo_righe_etichettate])
    modello_ssl.fit(X_train_f, Y_train_mascherati)

    predizioni_classico = modello_classico.predict(X_val_f)
    predizioni_ssl = modello_ssl.predict(X_val_f)

    risultati_classico.append(f1_score(Y_val_veri_f, predizioni_classico, average='macro'))
    risultati_ssl.append(f1_score(Y_val_veri_f, predizioni_ssl, average='macro'))

print(f"F1 Score Medio Classico: {np.mean(risultati_classico):.4f}")
print(f"F1 Score Medio SSL     : {np.mean(risultati_ssl):.4f}")

#fase di test sui dati tenuti da parte in precedenza
print(" TEST FINALE")


scaler_finale = StandardScaler()
X_addestramento_scalato = X_addestramento.copy()
X_test_scalato = X_test.copy()

X_addestramento_scalato[colonne_numeriche] = scaler_finale.fit_transform(X_addestramento[colonne_numeriche])
X_test_scalato[colonne_numeriche] = scaler_finale.transform(X_test[colonne_numeriche])

np.random.seed(42)
totale_righe = len(Y_addestramento)
indici_finali_nascosti = np.random.choice(totale_righe, size=int(totale_righe * 0.85), replace=False)

Y_add_mascherato = Y_addestramento.copy()
Y_add_mascherato[indici_finali_nascosti] = -1
solo_etichette_finali = (Y_add_mascherato != -1)

modello_classico.fit(X_addestramento_scalato[solo_etichette_finali], Y_add_mascherato[solo_etichette_finali])
modello_ssl.fit(X_addestramento_scalato, Y_add_mascherato)

test_pred_classico = modello_classico.predict(X_test_scalato)
test_pred_ssl = modello_ssl.predict(X_test_scalato)

print(f"MODELLO 1 (Supervisionato puro su 15% dati):")
print(f" - Accuratezza: {accuracy_score(Y_test_veri, test_pred_classico):.4f}")
print(f" - F1 Macro   : {f1_score(Y_test_veri, test_pred_classico, average='macro'):.4f}")
print(f" - Errore MAE : {mean_absolute_error(Y_test_veri, test_pred_classico):.4f}")

print(f"\nMODELLO 2 (Semi-Supervisionato SSL):")
print(f" - Accuratezza: {accuracy_score(Y_test_veri, test_pred_ssl):.4f}")
print(f" - F1 Macro   : {f1_score(Y_test_veri, test_pred_ssl, average='macro'):.4f}")
print(f" - Errore MAE : {mean_absolute_error(Y_test_veri, test_pred_ssl):.4f}")

delta_f1 = f1_score(Y_test_veri, test_pred_ssl, average='macro') - f1_score(Y_test_veri, test_pred_classico, average='macro')



print(" VALIDAZIONI E GRAFICI")



nomi_classi = ['Spinto (0)', 'Gestione (1)', 'Degrado (2)']
print("REPORT MODELLO 1 (Supervisionato):")
print(classification_report(Y_test_veri, test_pred_classico, target_names=nomi_classi, zero_division=0))

print("\nREPORT MODELLO 2 (Self-Training):")
print(classification_report(Y_test_veri, test_pred_ssl, target_names=nomi_classi, zero_division=0))

# matrici di confusione
fig, assi = plt.subplots(1, 2, figsize=(14, 6))

cm_classico = confusion_matrix(Y_test_veri, test_pred_classico)
cm_ssl = confusion_matrix(Y_test_veri, test_pred_ssl)

sns.heatmap(cm_classico, annot=True, fmt='d', ax=assi[0],
            xticklabels=nomi_classi, yticklabels=nomi_classi)
assi[0].set_title('Matrice di Confusione: Supervisionato')
assi[0].set_ylabel('Etichetta Reale')
assi[0].set_xlabel('Etichetta Predetta')

sns.heatmap(cm_ssl, annot=True, fmt='d', cmap='Greens', ax=assi[1],
            xticklabels=nomi_classi, yticklabels=nomi_classi)
assi[1].set_title('Matrice di Confusione: SSL')
assi[1].set_ylabel('Etichetta Reale')
assi[1].set_xlabel('Etichetta Predetta')

plt.tight_layout()
plt.show()

p_c, r_c, f_c, _ = precision_recall_fscore_support(Y_test_veri, test_pred_classico, average='macro', zero_division=0)
p_s, r_s, f_s, _ = precision_recall_fscore_support(Y_test_veri, test_pred_ssl, average='macro', zero_division=0)
x = np.arange(4)
plt.figure(figsize=(9, 5))
plt.bar(x - 0.15, [accuracy_score(Y_test_veri, test_pred_classico), p_c, r_c, f_c], 0.3, label='Supervisionato', color='steelblue')
plt.bar(x + 0.15, [accuracy_score(Y_test_veri, test_pred_ssl), p_s, r_s, f_s], 0.3, label='SSL', color='mediumseagreen')
plt.ylim(0.65, 0.85)
plt.xticks(x, ['Accuracy', 'Precision', 'Recall', 'F1-Score'])
plt.legend()
plt.tight_layout()
plt.show()