import os
import pandas as pd
from openai import OpenAI  # Standard OpenAI client mapped to OpenRouter
from tqdm import tqdm  # Visual progress bar for batch loops

def run_programmatic_ai_desk_analyst(df_test_row, ml_forecasted_price):
    """Parses grid snapshot, executes OpenRouter LLM request, returns response and prompt text."""
    # 1. Extract parameters safely with precise structural fallbacks
    timestamp = str(df_test_row.get('Datetime_UTC', 'Current Interval'))
    gas_price = float(df_test_row.get('gas_1d_lag', 0.0))
    carbon_price = float(df_test_row.get('Carbon_1d_lag', 0.0))
    load_mwh = float(df_test_row.get('Forecasted_Grid Load_(Forecasted_Total_Consumption)_MWh', 0.0))
    solar_mwh = float(df_test_row.get('Photovoltaics_MWh', 0.0))
    
    wind_mwh = max(0.0, float(df_test_row.get('PV_and_Wind_MWh', 0.0)) - solar_mwh)
    renew_ratio = float(df_test_row.get('renewable_penetration_ratio', 0.0))
    lag_24h = float(df_test_row.get('price_lag_24h', 0.0))
    
    client = OpenAI(
        base_url="https://openrouter.ai",
        api_key="YOUR_OPENAI_API_KEY"  
    )
    
    # 3. Construct prompt
    prompt_content = f"""
    You are an AI Energy Trader interpreting tomorrow's German (DE) day-ahead auction results.
    Analyze these physical grid telemetry snapshots and provide trade direction instructions.

    GRID TELEMETRY VARIABLES FOR TIMESTAMP [{timestamp}]:
    - System Forecasted Total Load: {load_mwh:.2f} MWh
    - Expected Solar PV Generation: {solar_mwh:.2f} MWh
    - Expected Total Wind Fleet Generation: {wind_mwh:.2f} MWh
    - Total Renewable Penetration Ratio: {renew_ratio * 100:.1f}%
    - Yesterday's Base Settlement Price: €{lag_24h:.2f}/MWh
    - Input Commodity Costs: Gas (TTF) = €{gas_price:.2f}/MWh, Carbon (EUA) = €{carbon_price:.2f}/t

    OUR ENSEMBLE MACHINE LEARNING QUANT PRICE FORECAST:
    - LightGBM + CatBoost Predicted Price: €{ml_forecasted_price:.2f}/MWh

    YOUR STRATEGIC TASK:
    Provide a concise, 3-sentence executive trading commentary:
    Sentence 1: State the core pricing direction risk based on the grid margin balance (e.g., severe midday negative price risk or steep evening conventional ramps).
    Sentence 2: Give 1 explicit physical event that INVALIDATES our ML forecast model (e.g., a real-time wind forecast miss over 15%, or sudden solar economic curtailments under the 'Solarspitzengesetz').
    Sentence 3: Provide tactical trading guidance for positioning a Day-Ahead-to-curve arbitrage spread view.
    """

    # 4. Programmatic execution block via OpenRouter
    # 4. Programmatic execution block via OpenRouter
    try:
        response = client.chat.completions.create(
            model='openai/gpt-4o-mini',  
            messages=[
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.15,
            max_tokens=150
        )
        
        # Guard against unexpected string injections or malformed structures
        if isinstance(response, str):
            llm_response_text = response.strip()
        elif hasattr(response, 'choices') and response.choices:
            llm_response_text = response.choices[0].message.content.strip()
        elif isinstance(response, dict) and 'choices' in response:
            llm_response_text = response['choices'][0]['message']['content'].strip()
        else:
            llm_response_text = f"[UNEXPECTED API STRUCTURE]: {str(response)}"

    except Exception as e:
        # Capture the exact error details into your dataset logs for tracking
        llm_response_text = f"[ERROR ENCOUNTERED]: {str(e)}"


def batch_process_predictions(csv_input_path, csv_output_path, prediction_column_name):
    df = pd.read_csv(csv_input_path)
    ai_log_timeline = (df['Datetime_UTC'] >= '2026-08-20 00:00:00')
    ai_log = df.loc[ai_log_timeline].copy()

    commentaries = []
    ai_log_records = []
    
    print(f"🚀 Processing {len(ai_log)} rows from '{csv_input_path}'...")
    
    for index, row in tqdm(ai_log.iterrows(), total=len(ai_log)):
        ml_price = float(row.get(prediction_column_name, 0.0))
        result = run_programmatic_ai_desk_analyst(row, ml_price)
        
        commentaries.append(result["Logged_Response"])
        
        log_entry = {
            "index": index,
            "timestamp_utc": str(row.get('Datetime_UTC', 'Unknown')),
            "ml_predicted_price": ml_price,
            "prompt_sent": result["Logged_Prompt"],
            "response_received": result["Logged_Response"]
        }
        ai_log_records.append(log_entry)
    
    ai_log['AI_Trader_Commentary'] = commentaries
    ai_log.to_csv(csv_output_path, index=False)
    print(f"💾 Augmented data saved successfully to: {csv_output_path}")


