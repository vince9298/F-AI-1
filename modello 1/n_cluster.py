import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():
    try:
        df = pd.read_csv('dati.csv')
    except FileNotFoundError:
        print("Errore: 'dati.csv' non trovato.")
        return

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


    wcss = []
    silhouette_coefficients = []

    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X_pca)


        wcss.append(kmeans.inertia_)


        score = silhouette_score(X_pca, kmeans.labels_)
        silhouette_coefficients.append(score)


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


    #creiamo i due grafici che ci aiuteranno nella scelta del numero di cluster

    ax1.plot(k_range, wcss)
    ax1.set_xlabel('Numero di Cluster (K)')
    ax1.set_ylabel('WCSS (Inerzia)')
    ax1.set_title('Metodo dell\'Elbow')
    ax1.set_xticks(k_range)
    ax1.grid(True, linestyle=':', alpha=0.6)


    ax2.plot(k_range, silhouette_coefficients)
    ax2.set_xlabel('Numero di Cluster (K)')
    ax2.set_ylabel('Silhouette Score Medio')
    ax2.set_title('Analisi della Silhouette')
    ax2.set_xticks(k_range)
    ax2.grid(True, linestyle=':', alpha=0.6)


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()