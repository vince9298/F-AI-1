import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

percorso_dataset = 'f1_dataset.csv'

df = pd.read_csv(percorso_dataset)

#la mescola va convertita in un valore numerico
df['Mescola_Codificata'] = df['Mescola'].astype('category').cat.codes

colonne = ['Ratio_Tempo_Giro', 'Indice_Performance_Team', 'Stint',
           'Vita_Gomma', 'Mescola_Codificata', 'Temperatura_Pista',
           'Temperatura_Aria', 'Benzina_Stimata']

corr_matrix = df[colonne].corr()

plt.figure(figsize=(12, 8))


sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    linewidths=0.5
)

plt.title('Matrice di Correlazione')

plt.show()