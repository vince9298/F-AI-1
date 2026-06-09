# F-AI — Analisi e Machine Learning su dati Formula 1

Progetto di Machine Learning applicato alla Formula 1. A partire dai dati ufficiali di gara e
qualifica (tempi sul giro, telemetria, meteo) vengono costruiti **quattro modelli indipendenti**,
ognuno dedicato a una domanda diversa: dal raggruppamento dei circuiti per stile di guida fino
alla previsione del passo di gara.

> 📄 Una **relazione** dettagliata con metodologia, risultati e discussione verrà aggiunta a breve.

---

## ⚙️ L'estrazione dei dati (parte più complessa del progetto)

Il dataset **non è un file pronto all'uso**: viene costruito da zero interrogando l'API
[FastF1](https://docs.fastf1.dev/), ed è di gran lunga la fase più onerosa dell'intero progetto.

Ogni script `estrazione.py` deve:

- **scaricare sessione per sessione** (Qualifica e/o Gara) per intere stagioni, scorrendo l'intero
  calendario di ogni anno ed escludendo i test;
- **collegare la telemetria giro per giro** (acceleratore, freno, marce, distanza): è il collo di
  bottiglia in termini di tempo, perché ogni singolo giro richiede una richiesta dedicata;
- **pulire e filtrare il rumore**: vengono scartate le gare bagnate (mescole `INTERMEDIATE`/`WET`),
  i giri sotto regime di bandiera (`TrackStatus != 1`), i giri di pit-in/pit-out, gli outlier oltre
  soglia (regola del 107% e varianti) e i giri con dati di telemetria corrotti o sensori difettosi;
- **gestire casi particolari** (es. la qualifica corrotta di Las Vegas, ricondotta all'anno 2024);
- **costruire feature derivate** non presenti nei dati grezzi: indice di performance del team
  (rapporto col tempo della pole), stima del carburante residuo in base al giro, normalizzazione
  dei cambi marcia per chilometro per rendere confrontabili piste di lunghezza diversa, ratio del
  tempo sul giro rispetto a un benchmark di gara, ecc.

### 🗂️ La cache è obbligatoria (e molto onerosa)
FastF1 salva i dati scaricati in una cartella `cache/` (diversi **GB**). Senza cache:
- ogni esecuzione riscaricherebbe tutto, con tempi lunghissimi;
- non è garantito che l'estrazione vada a buon fine.

⚠️ **Importante sui tempi e sui limiti dell'API:** i dati **vanno scaricati integralmente** da
FastF1 e il download è **molto pesante** sia in termini di tempo che di dimensione. Con troppe
richieste ravvicinate si rischia di **superare il limite di chiamate dell'API** (rate limit): in
questi casi l'estrazione va eseguita **in più sessioni**, riprendendola in un secondo momento.
Grazie alla cache i dati già scaricati non vengono richiesti di nuovo, quindi a ogni ripresa si
scarica solo la parte mancante fino a completare l'intero dataset.

La cartella `cache/` è **esclusa dal repository** (`.gitignore`) per dimensione: al primo avvio
verrà rigenerata automaticamente. **I file `.csv` già presenti nelle cartelle dei modelli sono il
risultato di queste estrazioni**, quindi gli script di training si possono eseguire direttamente
senza ripetere l'estrazione.

---

## 📁 Struttura del progetto

```
.
├── modello 1/      Clustering dei circuiti per stile di guida (non supervisionato)
├── modello 2/      Classificazione del circuito dallo stile di guida
├── modello 3/      Regressione del passo di gara
├── modello 4/      Passo di gara: supervisionato vs semi-supervisionato
├── dataset esempio.py   Esempio minimo di estrazione con FastF1
└── esempio.csv          Output d'esempio
```

Ogni cartella contiene tipicamente: `estrazione.py` (costruzione del dataset), il `.csv` generato,
`allenamento.py` (addestramento e valutazione) ed eventuali grafici prodotti.

---

## 🧩 I modelli

### Modello 1 — Clustering dei circuiti *(non supervisionato)*
Raggruppa i circuiti in base allo **stile di guida** richiesto, estratto dalla telemetria del giro
più veloce di qualifica (stagione 2025). Pipeline: standardizzazione → **PCA** (2 componenti) →
**K-Means**. Lo script `n_cluster.py` aiuta a scegliere il numero di cluster (metodo dell'Elbow +
analisi della Silhouette).

### Modello 2 — Classificazione del circuito *(supervisionato)*
Riconosce **su quale circuito** si sta guidando a partire dalle sole feature di stile di guida (top
giri per pilota). Addestramento sul 2023, test sul 2024. Confronto tra **Regressione Logistica**,
**SVM** e **Random Forest**, con grafici di importanza delle feature e matrice di confusione.

### Modello 3 — Regressione del passo di gara *(supervisionato)*
Prevede il **ratio del tempo sul giro** in funzione di gomma, stint, temperature, carburante stimato
e performance del team. Confronto tra **SVR**, **Random Forest** e **Gradient Boosting**, con
validazione **GroupKFold per gara** per misurare le reali capacità predittive su gare mai viste.

### Modello 4 — Supervisionato vs Semi-supervisionato
Classifica il passo di gara in 3 classi (*Spinto / Gestione / Degrado*) e confronta un **Random
Forest supervisionato** con un approccio **semi-supervisionato (Self-Training)**, nascondendo l'85%
delle etichette per simulare uno scenario con pochi dati annotati.

---

## 🚀 Come eseguire

```bash
# 1. ambiente virtuale
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 2. dipendenze
pip install fastf1 pandas numpy scikit-learn matplotlib seaborn

# 3. (opzionale) ri-estrarre i dati — richiede tempo e crea la cache
cd "modello 3"
python estrazione.py

# 4. addestramento e valutazione
python allenamento.py
```

> I dataset `.csv` sono già inclusi: il passo 3 è necessario solo se vuoi rigenerare i dati da zero.

---

## 🛠️ Stack tecnologico
**Python**, **FastF1**, **pandas**, **NumPy**, **scikit-learn**, **matplotlib**, **seaborn**.
