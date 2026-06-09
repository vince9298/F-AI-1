import pandas as pd
import numpy as np
import fastf1


fastf1.Cache.enable_cache('../cache')


def estazione_dati(anno, round_gp, nome_gp):

    try:
        sessione_g = fastf1.get_session(anno, round_gp, 'R')
        sessione_g.load(laps=True, telemetry=False, weather=True)

        sessione_q = fastf1.get_session(anno, round_gp, 'Q')
        sessione_q.load(laps=True, telemetry=False, weather=False)
    except Exception:
        return None

    #eliminiamo a prescindere le gare sul bagnato(avremmo troppo rumore sui dati se le includessimo)
    mescole_bagnato = ['INTERMEDIATE', 'WET']
    qualifica_bagnata = sessione_q.laps['Compound'].isin(mescole_bagnato).any()
    gara_bagnata = sessione_g.laps['Compound'].isin(mescole_bagnato).any()

    if qualifica_bagnata or gara_bagnata:
        print(f" Saltato (Sessione Bagnata)")
        return None

    #dalla qualifica estraiamo le performance del team
    giri_q = sessione_q.laps.copy()
    giri_q['LapTimeSeconds'] = giri_q['LapTime'].dt.total_seconds()
    giri_q = giri_q.dropna(subset=['LapTimeSeconds'])

    if giri_q.empty:
        return None

    tempo_pole = giri_q['LapTimeSeconds'].min()
    miglior_q_team = giri_q.groupby('Team')['LapTimeSeconds'].min().reset_index()
    miglior_q_team['Team_Base_Pace'] = miglior_q_team['LapTimeSeconds'] / tempo_pole
    dizionario_passo_team = dict(zip(miglior_q_team['Team'], miglior_q_team['Team_Base_Pace']))

    #rimuoviamo gli outlier
    giri_g = sessione_g.laps.copy()
    giri_g['LapTimeSeconds'] = giri_g['LapTime'].dt.total_seconds()
    giri_g = giri_g[giri_g['LapNumber'] > 2]
    giri_g = giri_g[giri_g['TrackStatus'] == '1']
    giri_g = giri_g.dropna(subset=['LapTimeSeconds', 'PitOutTime', 'PitInTime'], how='all')
    giri_g = giri_g[giri_g['LapTimeSeconds'] > 0]

    #integriamo la temperatura dell asfalto e dell aria
    try:
        dati_meteo = giri_g.get_weather_data().reset_index(drop=True)
        giri_g = giri_g.reset_index(drop=True)
        giri_g['TrackTemp'] = dati_meteo['TrackTemp']
        giri_g['AirTemp'] = dati_meteo['AirTemp']
    except Exception:
        giri_g['TrackTemp'] = np.nan
        giri_g['AirTemp'] = np.nan

    #filtro per i giri troppo lenti prima dell'estrazione del target
    giri_g['Grid_Lap_Median'] = giri_g.groupby('LapNumber')['LapTimeSeconds'].transform('median')
    giri_filtrati = giri_g[giri_g['LapTimeSeconds'] <= (giri_g['Grid_Lap_Median'] * 1.07)].copy()

    if giri_filtrati.empty:
        return None


    giri_filtrati['Year'] = anno
    giri_filtrati['Circuit'] = nome_gp

    # estraiamo la variabile target normalizzata
    benchmark_gara = giri_filtrati['LapTimeSeconds'].quantile(0.05)
    giri_filtrati['LapTime_Ratio'] = giri_filtrati['LapTimeSeconds'] / benchmark_gara

    #se il giro é troppo lento andiamo ad estrarlo(abbassare questa soglia migliorerá il modello nelle predizioni sul passo di gara, ma abbassa ancora la soglia di rilevamento)
    giri_filtrati = giri_filtrati[giri_filtrati['LapTime_Ratio'] <= 1.12]

    if giri_filtrati.empty:
        return None

    giri_filtrati['Team_Base_Pace'] = giri_filtrati['Team'].map(dizionario_passo_team)
    giri_filtrati['Team_Base_Pace'] = giri_filtrati['Team_Base_Pace'].fillna(giri_filtrati['Team_Base_Pace'].max())

    giri_totali_gara = giri_g['LapNumber'].max()

    #stimiamo il consumo di carburante in base al numero di giri
    giri_filtrati['EstimatedFuel'] = giri_filtrati['LapNumber'].apply(
        lambda x: 100.0 - ((x - 1) * (98.0 / giri_totali_gara))
    )

    # Mappiamo tutte le variabili per chiarezza
    mappatura = {
        'Year': 'Anno',
        'Circuit': 'Circuito',
        'Driver': 'Pilota',
        'Team': 'Team',
        'Team_Base_Pace': 'Indice_Performance_Team',
        'Stint': 'Stint',
        'Compound': 'Mescola',
        'TyreLife': 'Vita_Gomma',
        'TrackTemp': 'Temperatura_Pista',
        'AirTemp': 'Temperatura_Aria',
        'EstimatedFuel': 'Benzina_Stimata',
        'LapTimeSeconds': 'Tempo_Giro_Secondi',
        'LapTime_Ratio': 'Ratio_Tempo_Giro'
    }

    final_df = giri_filtrati[list(mappatura.keys())].copy()
    final_df = final_df.rename(columns=mappatura)
    return final_df


def run_mass_extraction(start_year=2022, end_year=2025):
    all_laps_dataset = []

    for year in range(start_year, end_year + 1):
        print(f"\n--- Elaborazione Stagione {year} ---")

        schedule = fastf1.get_event_schedule(year)
        races = schedule[schedule['EventFormat'] != 'testing']

        for _, row in races.iterrows():
            gp_round = row['RoundNumber']
            gp_nome = row['EventName']

            if gp_round == 0:
                continue

            print(f"[ESTRAZIONE] Round {gp_round}: {gp_nome}...", end="", flush=True)

            df_gp = estazione_dati(year, gp_round, gp_nome)

            if df_gp is not None and not df_gp.empty:
                all_laps_dataset.append(df_gp)
                print(f" Fatto! (Giri validi: {len(df_gp)})")



    if not all_laps_dataset:
        print("Nessun dato estratto.")
        return

    dataset = pd.concat(all_laps_dataset, ignore_index=True)
    dataset.to_csv('f1_dataset.csv', index=False)

    print(f"\nDATASET COMPILATO CON SUCCESSO!")
    print(f"➡ File salvato")
    print(f"➡ Righe totali nel dataset: {len(dataset)}")


if __name__ == "__main__":
    run_mass_extraction(start_year=2022, end_year=2025)