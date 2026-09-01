import os
import time
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

def generate_monthly_chunks(start_date_str, end_date_str):
    """Generates monthly date ranges to keep server payloads small."""
    start = datetime.strptime(start_date_str, "%Y-%m-%d")
    end = datetime.strptime(end_date_str, "%Y-%m-%d")
    
    chunks = []
    current_start = start
    while current_start <= end:
        # Move forward roughly 30 days
        current_end = current_start + timedelta(days=30)
        if current_end > end:
            current_end = end
        chunks.append((current_start.strftime("%Y-%m-%d"), current_end.strftime("%Y-%m-%d")))
        current_start = current_end + timedelta(days=1)
    return chunks

def collect_wind_data_onshore():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "Mozilla/5.0 GermanyWeatherBulletproof/3.0"})

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    
    # Representative German reference cities
    cities = {
        "Lower_Saxony": {"lat": 52.6367, "lon": 9.8451},
        "Schleswig-Holstein": {"lat": 54.2194, "lon": 9.6961},
        "Brandenburg": {"lat":  52.4125, "lon":  12.5316},
        "North Rhine-Westphalia": {"lat": 51.4332, "lon":7.6616}
    }
    
    today_str = datetime.today().strftime('%Y-%m-%d')
    chunks = generate_monthly_chunks("2023-01-01", today_str)
    
    # Dictionary to hold final DataFrames for each city
    city_dfs = {}
    
    for city_name, coords in cities.items():
        print(f"\n=== Processing City Grid: {city_name} ===")
        city_chunks = []
        
        for start_date, end_date in chunks:
            print(f"  Fetching {start_date} to {end_date}...")
            
            params = {
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "start_date": start_date,
                "end_date": end_date,
                "models": "icon_eu",
                "hourly": ["temperature_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
                "timezone": "GMT"
            }
            
            try:
                response = session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                # Check for standard error pages wrapped in 200 OK
                if "application/json" not in response.headers.get("Content-Type", "").lower():
                    print(f"    -> Skipped (Non-JSON payload returned)")
                    continue
                    
                data = response.json()
                
                if "hourly" in data:
                    m60 = data["hourly"]
                    chunk_df = pd.DataFrame({
                        "timestamp_gmt": pd.to_datetime(m60["time"]),
                        f"temp_{city_name}": m60["temperature_2m"],
                        f"wind_{city_name}": m60["wind_speed_10m"],
                        f"dir_{city_name}": m60["wind_direction_10m"],
                        f"gust_{city_name}": m60["wind_gusts_10m"]
                    })
                    chunk_df.set_index("timestamp_gmt", inplace=True)
                    city_chunks.append(chunk_df)
                
                # Tiny rest to avoid hitting API rate limits
                time.sleep(1.5)
                
            except Exception as e:
                print(f"    -> Block error ({start_date} to {end_date}): {e}")
                continue
        
        if city_chunks:
            # Combine all monthly chunks for this city
            city_dfs[city_name] = pd.concat(city_chunks, axis=0).sort_index()
            print(f"Finished {city_name}. Total intervals: {len(city_dfs[city_name])}")
            
    if len(city_dfs) == len(cities):
        print("\n=== Merging and Calculating Generalized Germany Averages ===")
        # Merge all cities together side-by-side
        combined_df = pd.concat(city_dfs.values(), axis=1)
        # Calculate the mathematical average across the columns
        final_df = pd.DataFrame(index=combined_df.index)
        final_df["onshore_temperature_2m_c"] = combined_df[[c for c in combined_df.columns if "temp" in c]].mean(axis=1)
        final_df["onshore_wind_speed_10m_kmh"] = combined_df[[c for c in combined_df.columns if "wind" in c]].mean(axis=1)
        final_df["onshore_wind_direction_10m_deg"] = combined_df[[c for c in combined_df.columns if "dir" in c]].mean(axis=1)
        final_df["onshore_wind_gusts_10m_kmh"] = combined_df[[c for c in combined_df.columns if "gust" in c]].mean(axis=1)
        final_df.reset_index(inplace=True)
        
        # Save output file
        output_file = "data/germany_onshore_hourly.csv"
        final_df.to_csv(output_file, index=False)
        run_weather_ingestion_quality_assurance(final_df)
        print(f"\nProcessing Complete!")
        print(f"File stored safely at: {output_file}")
        print(f"Total rows compiled: {len(final_df)}")
        print(final_df.head(3))
    else:
        print("\nError: Could not retrieve data for all 4 target regions. Partial data compilation aborted.")



def collect_wind_data_offshore():
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "Mozilla/5.0 GermanyWeatherBulletproof/3.0"})

    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    
    # Representative German reference cities
    sea = {
        "North_Sea": {"lat": 54.6738, "lon":6.7029},
        "Baltic_Sea": {"lat": 54.3414, "lon":12.1812}
        
    }
    
    today_str = datetime.today().strftime('%Y-%m-%d')
    chunks = generate_monthly_chunks("2023-01-01", today_str)
    
    # Dictionary to hold final DataFrames for each city
    sea_dfs = {}
    
    for sea_name, coords in sea.items():
        print(f"\n=== Processing City Grid: {sea_name} ===")
        sea_chunks = []
        
        for start_date, end_date in chunks:
            print(f"  Fetching {start_date} to {end_date}...")
            
            params = {
                "latitude": coords["lat"],
                "longitude": coords["lon"],
                "start_date": start_date,
                "end_date": end_date,
                "models": "icon_eu",
                "hourly": ["temperature_2m", "wind_speed_10m", "wind_direction_10m", "wind_gusts_10m"],
                "timezone": "GMT"
            }
            
            try:
                response = session.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                # Check for standard error pages wrapped in 200 OK
                if "application/json" not in response.headers.get("Content-Type", "").lower():
                    print(f"    -> Skipped (Non-JSON payload returned)")
                    continue
                    
                data = response.json()
                
                if "hourly" in data:
                    hourly_data = data["hourly"]
                    chunk_df = pd.DataFrame({
                        "timestamp_gmt": pd.to_datetime(hourly_data["time"]),
                        f"temp_{sea_name}": hourly_data["temperature_2m"],
                        f"wind_{sea_name}": hourly_data["wind_speed_10m"],
                        f"dir_{sea_name}": hourly_data["wind_direction_10m"],
                        f"gust_{sea_name}": hourly_data["wind_gusts_10m"]
                    })
                    chunk_df.set_index("timestamp_gmt", inplace=True)
                    sea_chunks.append(chunk_df)
                
                # Tiny rest to avoid hitting API rate limits
                time.sleep(1.5)
                
            except Exception as e:
                print(f"    -> Block error ({start_date} to {end_date}): {e}")
                continue
        
        if sea_chunks:
            # Combine all monthly chunks for this city
            sea_dfs[sea_name] = pd.concat(sea_chunks, axis=0).sort_index()
            print(f"Finished {sea_name}. Total intervals: {len(sea_dfs[sea_name])}")
            
    if len(sea_dfs) == len(sea):
        print("\n=== Merging and Calculating Generalized Germany Averages ===")
        # Merge all cities together side-by-side
        combined_df = pd.concat(sea_dfs.values(), axis=1)
        w_ns = 0.833
        w_bs = 0.167
        # Calculate the mathematical average across the columns
        final_df = pd.DataFrame(index=combined_df.index)
        print(combined_df.columns)
        final_df['offshore_wind_speed'] = (combined_df['wind_North_Sea'] * w_ns) + (combined_df['wind_Baltic_Sea'] * w_bs)
        final_df['offshore_wind_gusts'] = (combined_df['gust_North_Sea'] * w_ns) + (combined_df['gust_Baltic_Sea'] * w_bs)
        final_df['offshore_temperature'] = (combined_df['temp_North_Sea'] * w_ns) + (combined_df['temp_Baltic_Sea'] * w_bs)
        rad_ns = np.radians(combined_df['dir_North_Sea'])
        rad_bs = np.radians(combined_df['dir_Baltic_Sea'])
        sin_component = (np.sin(rad_ns) * w_ns) + (np.sin(rad_bs) * w_bs)
        cos_component = (np.cos(rad_ns) * w_ns) + (np.cos(rad_bs) * w_bs)
    
    # Reconstruct angles back to 0-360 degrees natively
        final_df['offshore_wind_direction'] = np.degrees(np.arctan2(sin_component, cos_component)) % 360    
        final_df.reset_index(inplace=True)
        
        # Save output file
        output_file = "data/germany_offshore_hourly.csv"
        final_df.to_csv(output_file, index=False)
        
        print(f"\nProcessing Complete!")
        print(f"File stored safely at: {output_file}")
        print(f"Total rows compiled: {len(final_df)}")
        print(final_df.head(3))
    else:
        print("\nError: Could not retrieve data for all 4 target regions. Partial data compilation aborted.")
    return


def run_weather_ingestion_quality_assurance(onshore_csv_path: str) -> tuple:
    """
    Executes structural validation and spatial vector aggregation checks 
    across onshore and marine weather streams, eliminating linear angular averages.
    """
    print("\n=========================================================================")
    print("➡️   INITIALIZING METEOROLOGICAL INGESTION QUALITY ASSURANCE (QA) LAYER  ")
    print("=========================================================================")
    
    # -----------------------------------------------------------------
    # CHECK 1: ONSHORE VECTOR SYNTHESIS CORRECTION (TRIGONOMETRIC MIX)
    # -----------------------------------------------------------------
    if not os.path.exists(onshore_csv_path):
        print(f"[⚠️] Onshore target file missing or not yet generated at: {onshore_csv_path}")
        return None, None

    print("[+] Executing Trigonometric Direction Check over Onshore Node Frames...")
    # Load your raw saved data rows to verify and rebuild direction indicators
    df_onshore = pd.read_csv(onshore_csv_path)
    df_onshore['timestamp_gmt'] = pd.to_datetime(df_onshore['timestamp_gmt']).dt.tz_localize(None)
    
    # Reindex onto a perfect unbroken reference baseline to catch network chunk drops
    perfect_timeline = pd.date_range(start=df_onshore['timestamp_gmt'].min(), end=df_onshore['timestamp_gmt'].max(), freq='h')
    df_onshore.set_index('timestamp_gmt', inplace=True)
    df_onshore = df_onshore.reindex(perfect_timeline)
    df_onshore.index.name = 'timestamp_gmt'
    df_onshore.reset_index(inplace=True)
    
    # Interpolate numeric gaps left behind by any missed 30-day loop exceptions
    num_cols = df_onshore.select_dtypes(include=[np.number]).columns
    df_onshore[num_cols] = df_onshore[num_cols].interpolate(method='linear', limit_direction='both')

    # If the file was written using raw city vectors before averages, we recalculate directions safely
    # Checking for specific column keys
    dir_cols = [c for c in df_onshore.columns if 'dir_' in c]
    
    if dir_cols:
        print(f" -> Found {len(dir_cols)} spatial direction components. Converting to vector sines/cosines...")
        sin_total = np.zeros(len(df_onshore))
        cos_total = np.zeros(len(df_onshore))
        
        for col in dir_cols:
            rad_angles = np.radians(df_onshore[col])
            sin_total += np.sin(rad_angles)
            cos_total += np.cos(rad_angles)
            
        # Reconstruct clean average angles (0 to 360) completely bypassing the linear average flaw
        df_onshore['onshore_wind_direction_10m_deg'] = np.degrees(np.arctan2(sin_total, cos_total)) % 360
        
    # Clean up names to match feature engineering expectations
    df_onshore.rename(columns={'timestamp_gmt': 'Datetime_UTC'}, inplace=True)
    df_onshore.to_csv(onshore_csv_path, index=False)
    print(f" Onshore file verified and updated safely at: {onshore_csv_path}")

    return df_onshore
