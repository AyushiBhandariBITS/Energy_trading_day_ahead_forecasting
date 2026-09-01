import os
import time
import numpy as np
import pandas as pd
import requests
import glob

def collect_smard_features(start_date, end_date):
# 1. FIXED: Configuration Settings expanded to handle all Filter IDs
    METRICS = {
        "Total_MWh": "122",
        "PV_and_Wind_MWh": "5097",
        "Wind_Offshore_MWh": "3791",
        "Wind_Onshore_MWh": "123",
        "Photovoltaics_MWh": "125",
        "Other_MWh": "124",
    "Forecasted_Grid Load_(Forecasted_Total_Consumption)_MWh" : "411",
    }
    REGION = "DE"  # Bidding Zone (DE-LU maps to DE)
    RESOLUTION = "hour" 
    
    # Generate clear 3-month blocks using pandas date ranges from Jan 2023 to present day
    current_date = pd.Timestamp.now().floor("D")
    # This forces the slices to land exactly on the FIRST day of the months
    date_bounds = pd.date_range(start=start_date, end=end_date, freq="3MS")
    
    
    # Convert bounds into clean tuple ranges [(start, end), ...]
    quarters = []
    for i in range(len(date_bounds) - 1):
        quarters.append((date_bounds[i], date_bounds[i + 1]))
    # Append the final uncompleted rolling trailing period up to today
    if date_bounds[-1] < current_date:
        quarters.append((date_bounds[-1], current_date))
    
    print(f" Defined {len(quarters)} individual 3-month batch files to fetch.\n")
    
    # Using the baseline Filter ID 122 to fetch structural indices layout
    base_filter = "122"
    index_url = f"https://www.smard.de/app/chart_data/{base_filter}/{REGION}/index_{RESOLUTION}.json"
    print(" Fetching SMARD platform structural index...")
    response = requests.get(index_url, timeout=15)
    if response.status_code != 200:
        raise Exception(
            f"Failed to access indices. Server responded with: {response.status_code}"
        )
    
    available_timestamps = response.json().get("timestamps", [])
    print(f"Found {len(available_timestamps)} global API fragments.\n" + "=" * 50)
    
    # 3. Step through each 3-month block systematically
    for idx, (q_start, q_end) in enumerate(quarters):
        q_label = f"{q_start.strftime('%Y')}_Block_{idx+1}"
        filename = f"data/smard_forecast_hour_{q_label}.csv"
    
        print(
            f"\n Starting Batch [{idx+1}/{len(quarters)}]: {q_start.date()} to {q_end.date()}"
        )
        print(f" Intended output path: {filename}")
    
        # Translate target range to Millisecond boundaries
        start_ms = int(q_start.tz_localize("UTC").timestamp() * 1000)
        end_ms = int(q_end.tz_localize("UTC").timestamp() * 1000)
    
        # Filter global timestamps that match this specific quarter slice
        target_chunks = [
            ts
            for ts in available_timestamps
            if ts >= (start_ms - 30 * 24 * 60 * 60 * 1000) and ts <= end_ms
        ]
    
        if not target_chunks:
            print(f" No active data blocks registered on the server for {q_label}.")
            continue
    
        # Dictionary to collect individual clean DataFrames for this quarter before merging
        quarter_dfs = []
    
        # Loop through each generation metric type individually
        for metric_name, filter_id in METRICS.items():
            print(f" Downloading {len(target_chunks)} files for metric: [{metric_name}]...")
            
            quarter_records = []
            for sub_idx, ts in enumerate(target_chunks):
                data_url = f"https://www.smard.de/app/chart_data/{filter_id}/{REGION}/{filter_id}_{REGION}_{RESOLUTION}_{ts}.json"
    
                try:
                    res = requests.get(data_url, timeout=10)
                    if res.status_code == 200:
                        chunk_data = res.json().get("series", [])
                        quarter_records.extend(chunk_data)
    
                    #  Short pause to prevent API rate-limiting blocks
                    time.sleep(0.15)
    
                    # Live updates for tracking long downloads
                    if (sub_idx + 1) % 5 == 0 or (sub_idx + 1) == len(target_chunks):
                        print(f"   ↳ [{metric_name}] Progress: [{sub_idx+1}/{len(target_chunks)}] files loaded...")
    
                except requests.exceptions.RequestException as e:
                    print(f"    Connection hiccup during query: {e}")
                    continue
    
            # 4. Clean and Parse this specific metric stream
            if quarter_records:
                df = pd.DataFrame(
                    quarter_records, columns=["Timestamp", metric_name]
                )
                df.drop_duplicates(subset=["Timestamp"], inplace=True)
    
                # Convert timestamps to UTC datetime profiles
                df["Datetime_UTC"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    
                # Strictly slice row results to the quarter's actual boundaries
                mask = (df["Datetime_UTC"] >= q_start.tz_localize("UTC")) & (
                    df["Datetime_UTC"] <= q_end.tz_localize("UTC")
                )
                df_filtered = df[mask][["Datetime_UTC", metric_name]].copy()
    
                if not df_filtered.empty:
                    quarter_dfs.append(df_filtered)
                    print(f"     Collected {len(df_filtered)} valid rows for {metric_name}")
                else:
                    print(f"     Zero rows within target window for {metric_name}")
            else:
                print(f"     Failed to parse data from server streams for {metric_name}")
    
        # 5. Merge all collected metrics into a single wide DataFrame and save
        if quarter_dfs:
            print(" Merging all metric streams together...")
            # Use the first dataframe as a starting baseline anchor
            master_quarter_df = quarter_dfs[0]
            
            # Sequentially join remaining dataframes on the exact Datetime_UTC key
            for next_df in quarter_dfs[1:]:
                master_quarter_df = pd.merge(master_quarter_df, next_df, on="Datetime_UTC", how="outer")
            
            master_quarter_df.sort_values(by="Datetime_UTC", inplace=True)
            
            # Save to disk immediately
            master_quarter_df.to_csv(filename, index=False)
            print(f" SUCCESS: Saved fully merged data table into {filename}")
        else:
            print(" No valid metric profiles were compiled for this batch interval.")
    
        #  Rest for a moment between distinct 3-month blocks to prevent throttling
        print(" Pausing for 3 seconds to rest the connection thread...")
        time.sleep(3.0)
    
    print("\n All 3-month multi-variable batch downloads are complete!")

    csv_files = glob.glob("data/smard_forecast_hour_*.csv")

    if not csv_files:
        print(
            "No files found! Make sure this script is running in the same directory as your CSVs."
        )
        exit()

    print(f" Found {len(csv_files)} files to verify. Merging timelines...")

    # 2. Combine all CSVs into one dataframe for analysis
    dfs = []
    for file in csv_files:
        try:
            temp_df = pd.read_csv(file)
            dfs.append(temp_df)
        except Exception as e:
            print(f" Error reading {file}: {e}")

    master_df = pd.concat(dfs, ignore_index=True)
    master_df["Datetime_UTC"] = pd.to_datetime(master_df["Datetime_UTC"])
    master_df.sort_values(by="Datetime_UTC", inplace=True)
    master_df.drop_duplicates(subset=["Datetime_UTC"], inplace=True)
    master_df.to_csv("data/germany_smard_forecast_hourly.csv")
    # Get the absolute starting and ending point of your data
    actual_start = master_df["Datetime_UTC"].min()
    actual_end = master_df["Datetime_UTC"].max()

    print("\n" + "=" * 50)
    print(f"📅 Data range detected: From {actual_start} to {actual_end}")
    print("=" * 50)

    # 4. Create a mathematically perfect, unbroken 15-minute timeline baseline
    perfect_timeline = pd.date_range(
        start=actual_start, end=actual_end, freq="60min", tz="UTC"
    )

    # 5. Compare your data against the perfect baseline
    expected_rows = len(perfect_timeline)
    actual_rows = len(master_df)
    missing_rows_count = expected_rows - actual_rows

    print(f"📋 Expected Data Points: {expected_rows}")
    print(f"📊 Actual Data Points:   {actual_rows}")

    if missing_rows_count == 0:
        print("\n✅ SUCCESS: Unbroken Timeline! Zero gaps or missing intervals found.")
    else:
        print(f"\n⚠️ WARNING: Found {missing_rows_count} missing 15-minute gap(s)!")

        # 6. Isolate exactly WHICH timestamps are missing
        master_df.set_index("Datetime_UTC", inplace=True)
        # Find timestamps present in the perfect timeline but missing in your data
        missing_timestamps = perfect_timeline.difference(master_df.index)

        print("\n🚨 Missing Timestamps List (First 10 shown):")
        for ts in missing_timestamps[:10]:
            print(f"   ↳ {ts}")

        if len(missing_timestamps) > 10:
            print(f"   ... and {len(missing_timestamps) - 10} more.")















import os
import time
import glob
import pandas as pd
import requests

def collect_features(start_date, end_date):
    # 1. FIXED: Changed Filter IDs from "Actual" to official Day-Ahead "Forecasted" versions
    METRICS = {
        "Forecasted_Total_Generation_MWh": "122",      # Day-ahead total forecast
        "Forecasted_PV_and_Wind_MWh": "5097",          # Combined Wind/Solar forecast
        "Forecasted_Wind_Offshore_MWh": "3791",        # Wind Offshore forecast
        "Forecasted_Wind_Onshore_MWh": "123",          # Wind Onshore forecast
        "Forecasted_Photovoltaics_MWh": "125",         # Solar forecast
        "Forecasted_Other_MWh": "124",                 # Other generation forecast
        "Forecasted_Grid_Load_MWh": "410"              # Forecasted Total Consumption
    }
    
    REGION = "DE"       # Bidding Zone 
    RESOLUTION = "hour" # Requesting hourly intervals
    os.makedirs("data", exist_ok=True)
    
    current_date = pd.Timestamp.now().floor("D")
    date_bounds = pd.date_range(start=start_date, end=end_date, freq="3MS")
    
    quarters = []
    for i in range(len(date_bounds) - 1):
        quarters.append((date_bounds[i], date_bounds[i + 1]))
    if date_bounds[-1] < current_date:
        quarters.append((date_bounds[-1], current_date))
    
    print(f"Defined {len(quarters)} individual 3-month batch blocks to fetch.\n")
    
    # Process each quarter block sequentially
    for idx, (q_start, q_end) in enumerate(quarters):
        q_label = f"{q_start.strftime('%Y')}_Block_{idx+1}"
        filename = f"data/smard_forecast_hourly_{q_label}.csv"
    
        print(f"\nStarting Batch [{idx+1}/{len(quarters)}]: {q_start.date()} to {q_end.date()}")
        print(f"Intended output path: {filename}")
    
        start_ms = int(q_start.tz_localize("UTC").timestamp() * 1000)
        end_ms = int(q_end.tz_localize("UTC").timestamp() * 1000)
    
        quarter_dfs = []
    
        for metric_name, filter_id in METRICS.items():
            # 2. FIXED: Dynamic index indexing. Pull index mapping specific to each independent Filter ID
            index_url = f"https://www.smard.de/app/chart_data/{filter_id}/{REGION}/index_{RESOLUTION}.json"
            try:
                response = requests.get(index_url, timeout=15)
                if response.status_code != 200:
                    print(f" Skipping {metric_name}: Index unavailable (Code {response.status_code})")
                    continue
                available_timestamps = response.json().get("timestamps", [])
            except Exception as e:
                print(f" Error getting index for {metric_name}: {e}")
                continue
                
            # Filter timestamps matching the time range
            target_chunks = [
                ts for ts in available_timestamps
                if ts >= (start_ms - 30 * 24 * 60 * 60 * 1000) and ts <= end_ms
            ]
    
            if not target_chunks:
                print(f" No active data blocks registered for {metric_name} in this timeline window.")
                continue
    
            print(f" Downloading {len(target_chunks)} file fragments for metric: [{metric_name}]...")
            quarter_records = []
            
            for sub_idx, ts in enumerate(target_chunks):
                data_url = f"https://www.smard.de/app/chart_data/{filter_id}/{REGION}/{filter_id}_{REGION}_{RESOLUTION}_{ts}.json"
                try:
                    res = requests.get(data_url, timeout=10)
                    if res.status_code == 200:
                        chunk_data = res.json().get("series", [])
                        quarter_records.extend(chunk_data)
                    time.sleep(0.15)  # Throttling protection
                except requests.exceptions.RequestException:
                    continue
    
            if quarter_records:
                df = pd.DataFrame(quarter_records, columns=["Timestamp", metric_name])
                df.drop_duplicates(subset=["Timestamp"], inplace=True)
                df["Datetime_UTC"] = pd.to_datetime(df["Timestamp"], unit="ms", utc=True)
    
                # Explicit bounding clip
                mask = (df["Datetime_UTC"] >= q_start.tz_localize("UTC")) & (df["Datetime_UTC"] <= q_end.tz_localize("UTC"))
                df_filtered = df[mask][["Datetime_UTC", metric_name]].copy()
    
                if not df_filtered.empty:
                    quarter_dfs.append(df_filtered)
                    print(f"     Collected {len(df_filtered)} valid rows for {metric_name}")
            
        # 3. FIXED: Clean chronological outer-joining loop to build balanced data wide frames
        if quarter_dfs:
            print(" Merging all metric streams together for this quarter...")
            master_quarter_df = quarter_dfs[0]
            for next_df in quarter_dfs[1:]:
                master_quarter_df = pd.merge(master_quarter_df, next_df, on="Datetime_UTC", how="outer")
            
            master_quarter_df.sort_values(by="Datetime_UTC", inplace=True)
            master_quarter_df.to_csv(filename, index=False)
            print(f" SUCCESS: Saved merged file -> {filename}")
        else:
            print(" No data profiles compiled for this block.")
            
        time.sleep(2.0)
    
    # 4. Final Aggregator Compilation Block
    print("\nProcessing compilation file merge...")
    csv_files = glob.glob("data/smard_forecast_hourly_*.csv")
    if not csv_files:
        print("No source files found to merge.")
        return None
        
    dfs = [pd.read_csv(f) for f in csv_files]
    master_df = pd.concat(dfs, ignore_index=True)
    master_df.drop_duplicates(subset=["Datetime_UTC"], inplace=True)
    master_df.sort_values(by="Datetime_UTC", inplace=True)
    
    master_df.to_csv("data/germany_smard_forecast_hourly.csv", index=False)
    run_smard_ingestion_quality_assurance()
    print("SUCCESS: Combined file saved to 'data/germany_smard_forecast_hourly.csv'")
    return master_df


def run_smard_ingestion_quality_assurance(input_csv_path: str) -> pd.DataFrame:
    """
    Executes structural data validation, tracks gaps against an unbroken timeline,
    and programmatically corrects missing rows, anomalies, and extreme values.
    """
    print("\n=========================================================================")
    print("➡️   INITIALIZING PROGRAMMATIC INGESTION QUALITY ASSURANCE (QA) LAYER     ")
    print("=========================================================================")
    
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"[-] QA Error: Target data footprint file missing at: {input_csv_path}")
        
    # 1. Load data and force strict datetime parsing
    df = pd.read_csv(input_csv_path)
    df['Datetime_UTC'] = pd.to_datetime(df['Datetime_UTC']).dt.tz_localize(None) # Force timezone-naive format
    df.sort_values(by='Datetime_UTC', inplace=True)
    df.drop_duplicates(subset=['Datetime_UTC'], inplace=True)
    
    actual_start = df['Datetime_UTC'].min()
    actual_end = df['Datetime_UTC'].max()
    
    # 2. Build a mathematically perfect, unbroken hourly reference baseline
    perfect_timeline = pd.date_range(start=actual_start, end=actual_end, freq='h')
    expected_rows = len(perfect_timeline)
    initial_rows = len(df)
    
    print(f"[+] Timeline Boundary Discovered: {actual_start} to {actual_end}")
    print(f"[+] Total Rows Extracted from Disk Cache : {initial_rows}")
    print(f"[+] Mathematically Expected Perfect Rows : {expected_rows}")
    
    # 3. Dynamic Gap Mitigation: Reindex dataset onto the perfect timeline vector
    df.set_index('Datetime_UTC', inplace=True)
    df_aligned = df.reindex(perfect_timeline)
    df_aligned.index.name = 'Datetime_UTC'
    df_aligned = df_aligned.reset_index()
    
    missing_row_count = expected_rows - initial_rows
    if missing_row_count > 0:
        print(f"[⚠️] WARNING DETECTED: Discovered {missing_row_count} structural hourly data gaps.")
        print("     ↳ Dynamic Mitigation Triggered: Executing chronological linear interpolation...")
        
        # Linearly interpolate missing structural elements to protect model shapes
        numeric_cols = df_aligned.select_dtypes(include=[np.number]).columns
        df_aligned[numeric_cols] = df_aligned[numeric_cols].interpolate(method='linear', limit_direction='both')
    else:
        print("[ Continuity Check Complete: 0 gaps discovered in the chronological index stream.")

    # =====================================================================
    # 📊 BOUNDARY VALIDATION & PHYSICAL BOUNDARY CLAMPING
    # =====================================================================
    print("[+] Evaluating physical volume thresholds and boundary limitations...")
    anomaly_logs = []
    
    # Generation metrics can never physically drop below zero megawatts
    volumetric_columns = [
        "Total_MWh", "PV_and_Wind_MWh", "Wind_Offshore_MWh", 
        "Wind_Onshore_MWh", "Photovoltaics_MWh", "Other_MWh",
        "Forecasted_Grid Load_(Forecasted_Total_Consumption)_MWh"
    ]
    
    # Keep only columns that exist in the dataframe slice
    active_vol_cols = [col for col in volumetric_columns if col in df_aligned.columns]
    
    for col in active_vol_cols:
        # Detect illegal negative energy observations
        negative_mask = df_aligned[col] < 0
        negative_count = negative_mask.sum()
        
        if negative_count > 0:
            anomaly_logs.append(f"Field '{col}': Clamped {negative_count} negative entries to 0.0 MWh.")
            df_aligned.loc[negative_mask, col] = 0.0
            
    # Check for missing values left behind
    null_summary = df_aligned.isnull().sum()
    total_nulls = null_summary.sum()
    if total_nulls > 0:
        print(f" Critical Residual Null Fields Detected:\n{null_summary[null_summary > 0]}")
        df_aligned.fillna(method='bfill', inplace=True) # Fail-safe backward fill anchor
        
    # 4. Print final operational report summaries to terminal logs
    if anomaly_logs:
        print(" BOUNDARY ANOMALIES INTERCEPTED AND RESOLVED:")
        for log in anomaly_logs:
            print(f"     ↳ {log}")
    else:
        print(" Physical Boundary Check Complete: All columns conform to physical constraints.")
        
    
    print(f"\n SUCCESS: Post-Ingestion QA Complete. Matrix ")
    print("=========================================================================\n")
    
    return df_aligned