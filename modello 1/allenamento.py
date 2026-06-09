import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def main():

    try:
        df = pd.read_csv('dati.csv')
    except FileNotFoundError:
        print("Errore: 'dati.csv' non trovato. Assicurati di aver eseguito lo script di estrazione.")
        return

    #definiamo le feature, scaliamo i dati e applichiamo la PCA

    features = [
        'Percentuale_acceleratore_premuto',
        'Numero_cambi_marcia',
        'Percentuale_freno',
        'Percentuale_acc_parzializzato'
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]

    print("\nAddestramento algoritmo di clustering")
    n_clusters = 4

    #inizializziamo il modello k means
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
    df['Cluster_ID'] = kmeans.fit_predict(X_pca)

    plt.figure(figsize=(14, 8))
    colori = [
        '#E63946',
        '#1D3557',
        '#F4A261',
        '#2A9D8F',
        '#E9C46A'
    ]

    #impostiamo il grafico, quindi impostiamo gli assi, i punti dei centroidi per vedere dove cade e togliamo dal nome degli eventi Grand Prix

    for cluster_id in range(n_clusters):
        mask = df['Cluster_ID'] == cluster_id
        plt.scatter(df.loc[mask, 'PC1'],
                    df.loc[mask, 'PC2'],
                    c=colori[cluster_id], s=200, edgecolor='black', alpha=0.85,
                    label=f"Cluster {cluster_id}")


    centroidi = kmeans.cluster_centers_
    plt.scatter(centroidi[:, 0], centroidi[:, 1], color='black', marker='X', s=300,
                edgecolor='white', linewidths=2, label='Centroidi', zorder=10)



    for i, row in df.iterrows():
        nome = str(row['Track']).replace(" Grand Prix", "")
        plt.annotate(nome, (row['PC1'], row['PC2']),
                     fontsize=8, alpha=0.7, xytext=(5, 5), textcoords='offset points')


    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.4)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.4)

    plt.xlabel('PC1', fontsize=12, fontweight='bold')
    plt.ylabel('PC2', fontsize=12, fontweight='bold')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(title="Cluster Identificati", loc='best')
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()