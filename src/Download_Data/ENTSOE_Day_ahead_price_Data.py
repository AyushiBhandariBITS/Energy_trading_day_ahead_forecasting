
import time
import os
import numpy as np
import pandas as pd
from entsoe import EntsoePandasClient

def collect_day_ahead_price(start_date, end_date):
    client = EntsoePandasClient(api_key= "YOUR ENTSOE KEY")

    # Data Timeline
    start_total = pd.Timestamp(start_date, tz='Europe/Berlin')
    end_total = pd.Timestamp(end_date, tz='Europe/Berlin')
    country_code = 'DE_LU'
    
    # Generate 90-day chunks (approx. 3-month batches)
    date_range = pd.date_range(start=start_total, end=end_total, freq='30D')
    if date_range[-1] < end_total:
        date_range = date_range.append(pd.DatetimeIndex([end_total]))
    
    all_batches = []
    
    print("Starting batch downloads...")
    
    # Loop through chunks to avoid overloading the server and missing any data points
    for i in range(len(date_range) - 1):
        start_chunk = date_range[i]
        end_chunk = date_range[i+1]
        
        print(f"Downloading batch {i+1}: From {start_chunk.date()} to {end_chunk.date()}...")
            # --- FIX: Retry Loop with Exponential Backoff ---
        max_retries = 5
        retry_delay = 5  # Start with a 5-second wait
        success = False
        prices_chunk = None  
        for attempt in range(max_retries):          
            try:
                # Try fetching with explicit 60min resolution configuration
                prices_chunk = client.query_day_ahead_prices(
                    country_code, 
                    start=start_chunk, 
                    end=end_chunk,
                )
                break  # Break out of retry loop if successful
                
            except Exception as e:
                print(f"  [Attempt {attempt + 1}/{max_retries}] Error: {e}")
                if attempt < max_retries - 1:
                    print(f"  Server busy (503) or error encountered. Waiting {retry_delay} seconds before retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Double the wait time for the next attempt
                else:
                    print(f"  Batch {i+1} permanently failed after {max_retries} attempts.")

        if not success or prices_chunk is None:
            continue 

        try:
            # Convert series data to dataframe structures safely
            if isinstance(prices_chunk, pd.Series):
                df_chunk = prices_chunk.to_frame(name='Price_EUR_MWh')
            else:
                df_chunk = prices_chunk.iloc[:, [0]].copy()
                df_chunk.columns = ['Price_EUR_MWh']
                
            # Standardize all records to a 60-minute grid
            df_chunk = df_chunk.resample('60min').ffill()
            all_batches.append(df_chunk)
            
        except Exception as e:
            print(f"Error processing batch {i+1} data: {e}")
        
        # Consistent baseline pause between clean requests
        time.sleep(2)
    
    # Combine all downloaded chunks together
    if all_batches:
        print("\nMerging all batches and formatting final file...")
        final_df = pd.concat(all_batches)
        
        # Remove any minor overlap rows caused by the slicing boundaries
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        
        # Extract the timestamp into a standalone column
        final_df = final_df.reset_index()
        final_df.columns = ['DateTime', 'Price_EUR_MWh']
        
        # Ensure target directory exists before saving
        os.makedirs('data', exist_ok=True)
        
        # Save directly to a local CSV file
        final_df.to_csv('data/germany_prices_hourly_day_ahead.csv', index=False)
        
        try:
            run_extraction_qa('data/germany_prices_hourly_day_ahead.csv')
        except NameError:
            print("Notice: run_extraction_qa function not defined in this scope, skipping QA check.")
            
        print("Success! Data successfully saved to 'data/germany_prices_hourly_day_ahead.csv'")
        print(final_df.head(10))
    else:
        print("No data was retrieved. Please check your API token or connection settings.")


def run_extraction_qa(csv_path: str = 'data/germany_prices_hourly_day_ahead.csv', output_dir: str = 'outputs'):
    """
    Executes a strict QA pipeline on the extracted 60-minute ENTSO-E price data.
    Generates a formal text artifact in the outputs/ directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    report = ["====================================================",
              "       ENTSO-E EXTRACTION QA AUDIT REPORT           ",
              "===================================================="]
    
    # Check 1: File Existence
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"QA Failed: Target file '{csv_path}' does not exist.")
        
    df = pd.read_csv(csv_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    
    # Check 2: Row and Shape Properties
    report.append(f"\n[1. DATASET COMPACTNESS]")
    report.append(f" Total rows extracted: {len(df)}")
    report.append(f" Total columns: {list(df.columns)}")
    
    # Check 3: Completeness & NaN Detection
    report.append(f"\n[2. MISSING VALUES & COMPLETENESS]")
    nan_count = df['Price_EUR_MWh'].isnull().sum()
    report.append(f" Missing price records (NaNs): {nan_count}")
    
    # Check 4: 60-Minute Grid Continuity (Critical for ENTSO-E)
    report.append(f"\n[3. TIMELINE CONTINUITY & CHRONOLOGY]")
    min_time = df['DateTime'].min()
    max_time = df['DateTime'].max()
    report.append(f" Extracted Time Horizon: {min_time} to {max_time}")
    
    # Construct a theoretical perfect 60-minute grid
    perfect_grid = pd.date_range(start=min_time, end=max_time, freq='60min')
    report.append(f" Expected theoretical rows in timeline: {len(perfect_grid)}")
    
    grid_gap = len(perfect_grid) - len(df['DateTime'].unique())
    report.append(f" Missing chronological 60-min intervals: {grid_gap}")
    
    # Check 5: Duplicate Timestamps
    report.append(f"\n[4. DUPLICATE TILES]")
    dup_count = df.duplicated(subset=['DateTime']).sum()
    report.append(f" Found duplicate timestamps: {dup_count}")

    # Check 6: Extreme Value and Outlier Sanity Checks
    report.append(f"\n[5. EXTREME PRICE MARKET BOUNDARIES]")
    max_p = df['Price_EUR_MWh'].max()
    min_p = df['Price_EUR_MWh'].min()
    report.append(f" Maximum observed price: {max_p:.2f} EUR/MWh")
    report.append(f" Minimum observed price: {min_p:.2f} EUR/MWh")
    
    # Flag values outside standard regulatory caps
    extreme_spikes = (df['Price_EUR_MWh'] > 500).sum()
    extreme_plunges = (df['Price_EUR_MWh'] < -600).sum()
    report.append(f" Instances exceeding €500/MWh (Spikes): {extreme_spikes}")
    report.append(f" Instances below -€600/MWh (Deep Negatives): {extreme_plunges}")
    
    # Compile text and write file
    report_text = "\n".join(report)
    output_path = os.path.join(output_dir, "extraction_qa_results.txt")
    with open(output_path, "w") as f:
        f.write(report_text)
        
    print(f"\n[QA Audit] Verification complete. Report written to: '{output_path}'")
    
    # Check for immediate pipeline failures
    if nan_count > 0 or grid_gap > 0 or dup_count > 0:
        print("QA WARNING: Gaps, duplicates, or missing cells detected in raw data. Check logs.")
    else:
        print(" QA PASS: 60-minute time series grid is continuous, clean, and complete.")