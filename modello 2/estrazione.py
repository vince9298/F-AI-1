import fastf1
import pandas as pd
#pip install fastf1

#colleghiamo le cache in modo da non dover scaricare i dati ogni volta
#SENZA CACHE non é assicurato che il codice venga runnato in maniera corretta, per quanto pesanti sono fondamentali per la buona riuscita delle estrazioni
#L'ESTRAZIONE DEI DATI ATTRAVERSO QUESTA PAGINA RICHIEDE TEMPO A CAUSA DEL COLLEGAMENTO AD OGNI GIRO DEI DATI TELEMETRICI

#colleghiamo la cartella di cache in modo da avere i dati senza doverli riscaricare ogni volta

fastf1.Cache.enable_cache('../cache')

"""
    Definiamo alcune variabili utili ai fini del preprocessamento e all'estrazione dei dati 
    1) Gli anni di estrazione dei dati
    2) Gli X giri piú veloci che vogliamo estrarre
    3) La soglia sopra il cui un giro viene scartato perché contenente un'anomalia evidente (107%)
"""

ANNI = [2023, 2024]
TOP_X_GIRI = 5
SOGLIA_PACE = 1.07


def estrai_dataset_circuiti():
    dati = []

    for anno in ANNI:

        #prendiamo tutti gli eventi del calendario per l'anno selezionato

        calendario = fastf1.get_event_schedule(anno)
        gare = calendario[calendario['EventFormat'] != 'testing']

        for _, evento in gare.iterrows():
            pista = evento['EventName']
            numero_round = evento['RoundNumber']

            print(f"[{anno}] {pista} ...", end=' ')

            try:
                session = fastf1.get_session(anno, numero_round, 'R')
                session.load(telemetry=True, weather=False, messages=False)

                #dopo aver isolato i giri della gara li filtriamo ed eliminiamo i giri con problemi evidenti o con outlier meteorologici

                laps = session.laps
                filtro = laps[
                    laps['IsAccurate'] &
                    laps['PitInTime'].isna() & laps['PitOutTime'].isna() &
                    (laps['TrackStatus'] == '1') &
                    (~laps['Compound'].isin(['INTERMEDIATE', 'WET'])) &
                    (laps['LapNumber'] > 1)
                ]

                #saltiamo la gara se tutti i giri sono anomali(es. gara bagnata)

                if filtro.empty:
                    print("0 giri (nessun giro pulito)")
                    continue


                sec = filtro['LapTime'].dt.total_seconds()
                best = filtro.groupby('Driver')['LapTime'].transform('min').dt.total_seconds()
                filtro = filtro[sec <= best * SOGLIA_PACE]


                piloti = list(filtro['Driver'].unique())

                righe = 0
                for drv in piloti:
                    giri_drv = filtro[filtro['Driver'] == drv]
                    idx_top = giri_drv['LapTime'].sort_values().index[:TOP_X_GIRI]

                    #isoliamo i top giri di ogni pilota e per tutti i giri migliori dei piloti colleghiamo la telemetria e le feature scelte

                    for idx in idx_top:
                        lap = filtro.loc[idx]
                        try:
                            tel = lap.get_telemetry()
                        except Exception:
                            continue

                        lung = tel['Distance'].max()

                        #isoliamo e rimuoviamo i giri con problemi ai sensori

                        if pd.isna(lung) or lung < 2000:
                            continue

                        thr = tel['Throttle']
                        brk = tel['Brake'] > 0
                        dati.append({
                            'Track': pista,
                            'Year': anno,
                            'Percentuale_acceleratore_premuto': (thr > 95).mean(),
                            'Numero_cambi_marcia': ((tel['nGear'] != tel['nGear'].shift(1)).sum()) * 1000 / lung,
                            'Numero_frenate': ((brk.astype(int).diff() == 1).sum()) * 1000 / lung,
                            'Percentuale_freno': brk.mean(),
                            'Percentuale_acc_parzializzato': ((thr > 0) & (thr < 65)).mean(),
                        })
                        righe += 1

                print(f"{righe} giri")

            except Exception as e:
                print(f"ERRORE: {e}")

    return dati


def main():
    dati = estrai_dataset_circuiti()
    if not dati:
        print("\nNessun dato estratto. Controlla connessione e cache.")
        return

    df = pd.DataFrame(dati).dropna()

    #dato che alleniamo sul 2023 e validiamo sul 2024, teniamo le piste che hanno giri validi in entrambi gli anni(saltano un paio di piste)

    conteggi = df.groupby(['Track', 'Year']).size().unstack(fill_value=0)
    for a in ANNI:
        if a not in conteggi.columns:
            conteggi[a] = 0
    validi = conteggi[(conteggi[2023] >= 10) &
                      (conteggi[2024] >= 10)].index
    print("\nGiri per circuito e anno:")
    print(conteggi)
    df = df[df['Track'].isin(validi)]

    df.to_csv('dataset_circuiti_giri.csv', index=False)
    print(f"\nSalvato 'dataset_circuiti_giri.csv'  |  righe: {len(df)}  |  circuiti: {df['Track'].nunique()}")


if __name__ == "__main__":
    main()