import fastf1
import pandas as pd

fastf1.Cache.enable_cache('cache')

def estrai_dataset():
    session = fastf1.get_session(2024, 'Monza', 'R')

    session.load(laps=True, telemetry=False, weather=False)

    session = session.laps.copy()

    giri = session.copy()

    return giri

def main():
    giri = estrai_dataset()

    df = pd.DataFrame(giri)

    df.to_csv('esempio.csv')




if __name__ == "__main__":
     main()