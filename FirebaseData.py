import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import json

def export_rtdb_to_csv(database_url: str, output_file: str, certificate_path: str):
    
    cred = credentials.Certificate(certificate_path)
    
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred, {
            'databaseURL': database_url
        })

    ref = db.reference("/DHT11/Temperature")
    raw_dataTemperature = ref.get()

    ref2 = db.reference("/DHT11/Humidity")
    raw_dataHumidity = ref2.get()

    ref3 = db.reference("/DHT11/Time")
    raw_dataTime = ref3.get()

    # Normalize data from the three nodes and align lengths
    # If all three are dicts: align by common keys (preserve order of temperature dict)
    if isinstance(raw_dataTemperature, dict) and isinstance(raw_dataHumidity, dict) and isinstance(raw_dataTime, dict):
        list_temp = list(raw_dataTemperature.values())
        list_hum = list(raw_dataHumidity.values())
        list_time = list(raw_dataTime.values())
        len_temp = len(list_temp)
        len_hum = len(list_hum)
        len_time = len(list_time)
        min_len = min(len_temp, len_hum, len_time)
        if min_len == 0:
            df = pd.DataFrame(columns=["Temperature", "Humidity", "Time"])
        else:
            df = pd.DataFrame({
                "Temperature": list_temp[:min_len],
                "Humidity": list_hum[:min_len],
                "Time": list_time[:min_len]
            })
    else:
        # Fallback: convert each node to a list (dict -> values()) and truncate to smallest length
        def to_list(x):
            if x is None:
                return []
            if isinstance(x, dict):
                return list(x.values())
            if isinstance(x, (list, tuple)):
                return list(x)
            return [x]

        l_temp = to_list(raw_dataTemperature)
        l_hum = to_list(raw_dataHumidity)
        l_time = to_list(raw_dataTime)

        min_len = min(len(l_temp), len(l_hum), len(l_time))
        if min_len == 0:
            df = pd.DataFrame(columns=["Temperature", "Humidity", "Time"])
        else:
            df = pd.DataFrame({
                "Temperature": l_temp[:min_len],
                "Humidity": l_hum[:min_len],
                "Time": l_time[:min_len]
            })

    # Try to parse Time column if present
    if "Time" in df.columns:
        try:
            df["Time"] = pd.to_datetime(df["Time"])
        except Exception:
            pass

    df.to_csv(output_file, index=False)
    print(f"Successfully exported to {output_file}")

if __name__ == "__main__":
    DB_URL = "https://ptta-weather-463d9-default-rtdb.firebaseio.com/"
    CERT_PATH = "./mycertificate.json"
    export_rtdb_to_csv(DB_URL, "temp_data.csv", certificate_path=CERT_PATH)