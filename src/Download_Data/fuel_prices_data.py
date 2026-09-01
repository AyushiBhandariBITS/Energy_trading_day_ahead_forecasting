import pandas as pd
import yfinance as yf

import os
import pandas as pd
import numpy as np

def run_fuel_ingestion_quality_assurance(input_csv_path: str, output_csv_path: str) -> pd.DataFrame:
    """
    Executes structural data validation on commodity close datasets,
    preserves weekend gaps using forward-fills, and standardizes date vectors.
    """
    print("\n=========================================================================")
    print("➡️   INITIALIZING COMMODITY PRICE INGESTION QUALITY ASSURANCE (QA) LAYER  ")
    print("=========================================================================")
    
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"[-] QA Error: Target fuel data file missing at: {input_csv_path}")
        
    # 1. Ingest raw financial array matrix
    df = pd.read_csv(input_csv_path)
    
    # Isolate and convert the text date components safely into timezone-naive dates
    # Splits the string '2026-08-31 00:00:00 CEST' to grab just the raw calendar day
    df['clean_date'] = pd.to_datetime(df['Date'].str.split(' ').str[0]).dt.date
    df.sort_values(by='clean_date', inplace=True)
    df.drop_duplicates(subset=['clean_date'], inplace=True)
    
    actual_start = df['clean_date'].min()
    actual_end = df['clean_date'].max()
    
    print(f"[+] Commodity Trading Calendar Detected: {actual_start} to {actual_end}")
    print(f"[+] Total Raw Business Days Ingested  : {len(df)}")
    
    # 2. Build a perfect, continuous calendar range including ALL Saturdays and Sundays
    perfect_calendar = pd.date_range(start=actual_start, end=actual_end, freq='D').date
    
    # Reindex onto the complete calendar timeline using an outer join alignment framework
    calendar_df = pd.DataFrame({'clean_date': perfect_calendar})
    df_aligned = pd.merge(calendar_df, df, on='clean_date', how='left')
    
    # 3. Dynamic Forward-Fill Buffer: Carry Friday's closing prices over the weekend gaps
    fuel_columns = [col for col in df_aligned.columns if 'close_' in col]
    
    initial_null_count = df_aligned[fuel_columns].isnull().sum().sum()
    print(f"[⚠️] Detected {initial_null_count} empty rows due to weekend exchange closures and holidays.")
    print("     ↳ Dynamic Mitigation Triggered: Executing chronological forward-fill (`ffill`)...")
    
    # Forward-fill weekend gaps, then backward-fill any initial boundary gaps
    df_aligned[fuel_columns] = df_aligned[fuel_columns].ffill().bfill()
    
    # 4. Final Validation: Ensure zero null values remain in the matrix
    final_null_count = df_aligned[fuel_columns].isnull().sum().sum()
    if final_null_count == 0:
        print("[✅] Continuity Check Complete: All weekend and holiday gaps successfully patched.")
    else:
        print(f"[❌] Critical QA Defect: {final_null_count} missing entries remain unresolved.")
        
    # 5. Format to clean output and export
    # Re-standardize column naming styles for your feature engineering pipelines
    df_aligned['Date'] = pd.to_datetime(df_aligned['clean_date']).dt.strftime('%Y-%m-%d')
    final_output_cols = ['Date'] + fuel_columns
    df_final = df_aligned[final_output_cols].copy()
    
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df_final.to_csv(output_csv_path, index=False)
    
    print(f" SUCCESS: Fuel Ingestion QA Complete. Verified matrix saved at: '{output_csv_path}'")
    print("=========================================================================\n")
    
    return df_final
