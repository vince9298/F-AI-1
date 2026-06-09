import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dati.csv')

numerical_df = df.drop(columns=['Track'])

correlation_matrix = numerical_df.corr()

print(correlation_matrix)
#creiamo la matrice di correlazione
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matrice di Correlazione')

plt.savefig('matrice_correlazione.png')