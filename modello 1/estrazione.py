import fastf1
import pandas as pd
#pip install fastf1

#colleghiamo le cache in modo da non dover scaricare i dati ogni volta
#SENZA CACHE non é assicurato che il codice venga runnato in maniera corretta, per quanto pesanti sono fondamentali per la buona riuscita delle estrazioni

fastf1.Cache.enable_cache('../cache')

def estrai_dataset_circuiti():

    anno_calendario = 2025
    calendario = fastf1.get_event_schedule(anno_calendario)
    gare = calendario[calendario['EventFormat'] != 'testing']

    #preso il calendario completo degli eventi, estraiamo il giro piú veloce di qualifica e le relative feature telemetriche

    dati = []

    for index, evento in gare.iterrows():
        pista = evento['EventName']
        numero_round = evento['RoundNumber']

        #per la quali di Las Vegas corrotta, l anno di estrazione diventa il 2024
        if pista == "Las Vegas Grand Prix":
            anno_riferimento = 2024
            identificatore_sessione = "Las Vegas Grand Prix"
        else:
            anno_riferimento = anno_calendario
            identificatore_sessione = numero_round

        print(f"Elaborazione: {pista} (Anno Riferimento: {anno_riferimento})...")

        try:

            session = fastf1.get_session(anno_riferimento, identificatore_sessione, 'Q')
            session.load(telemetry=True, weather=False, messages=False)


            fastest_lap = session.laps.pick_fastest()

            if pd.isna(fastest_lap['LapTime']):
                print(f"  -> Nessun tempo valido per {pista}, salto.")
                continue


            telemetria = fastest_lap.get_telemetry()
            len_circuito = telemetria['Distance'].max()

            pct_acc_premuto = (telemetria['Throttle'] > 95).mean()
            pct_acc_parzializzato = ((telemetria['Throttle'] > 0) & (telemetria['Throttle'] < 65)).mean()
            numero_cambi_marcia_per_chilometro = ((telemetria['nGear'] != telemetria['nGear'].shift(1)).sum()) * 1000 / len_circuito #questa feature é stata normalizzata per renderla affidabile in piste sia corte che lunghe
            pct_freno = (telemetria['Brake'] > 0).mean()

            dati.append({
                'Track': pista,
                'Percentuale_acceleratore_premuto': pct_acc_premuto,
                'Numero_cambi_marcia': numero_cambi_marcia_per_chilometro,
                'Percentuale_freno': pct_freno,
                'Percentuale_acc_parzializzato': pct_acc_parzializzato
            })

        except Exception as e:
            print(f"  -> Errore durante l'estrazione per {pista}: {e}")

    return dati
def main():
    dati = estrai_dataset_circuiti()

    if dati:
        df = pd.DataFrame(dati)
        df.to_csv('dati.csv', index=False)
        print("\nEstrazione completata con successo. Dataset salvato in 'dati.csv'")
    else:
        print("\nNessun dato estratto. Controlla la connessione e la cache.")


if __name__ == "__main__":
    main()