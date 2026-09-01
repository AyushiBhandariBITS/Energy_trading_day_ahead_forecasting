import os
import pandas as pd
import numpy as np

def translate_predictions_to_proper_curve(predictions_df: pd.DataFrame) -> tuple:

    df = predictions_df.copy()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Map every single hour to its specific tradable ISO week string (e.g., '2026-W36')
    df['ISO_Week'] = df['datetime'].dt.strftime('%Y-W%V')
    # Map hourly rows to distinct calendar months (e.g., '2026-09')
    df['Month_Period'] = df['datetime'].dt.to_period('M').astype(str)
    
    # Define masks for Peakload calculations (Mon-Fri, 08:00 to 20:00)
    weekdays_mask = df['day_of_week'].between(0, 4)
    peak_hours_mask = df['hour'].between(8, 19)
    
    # --- WEEKLY AGGREGATION ---
    weekly_base = df.groupby('ISO_Week')['y_pred'].mean().reset_index().rename(columns={'y_pred': 'Expected_Baseload'})
    weekly_peak = df[weekdays_mask & peak_hours_mask].groupby('ISO_Week')['y_pred'].mean().reset_index().rename(columns={'y_pred': 'Expected_Peakload'})
    weekly_df = pd.merge(weekly_base, weekly_peak, on='ISO_Week', how='left')
    
    # --- MONTHLY AGGREGATION ---
    monthly_base = df.groupby('Month_Period')['y_pred'].mean().reset_index().rename(columns={'y_pred': 'Expected_Baseload'})
    monthly_peak = df[weekdays_mask & peak_hours_mask].groupby('Month_Period')['y_pred'].mean().reset_index().rename(columns={'y_pred': 'Expected_Peakload'})
    monthly_df = pd.merge(monthly_base, monthly_peak, on='Month_Period', how='left')
    
    return weekly_df, monthly_df


def generate_trading_signals(weekly_forecasts: pd.DataFrame, current_market_curves: dict, threshold: float = 2.0) -> pd.DataFrame:

    print("Deriving arbitrage positions from aggregated curves...")
    summary_df = weekly_forecasts.copy()
    
    # Map traded financial curve prices to our dataframe
    summary_df['Current_Curve_Price'] = summary_df['ISO_Week'].map(current_market_curves)
    
    # Backfill with a realistic placeholder if an explicit week isn't defined in our input map
    default_market_price = np.mean(list(current_market_curves.values())) if current_market_curves else 85.0
    summary_df['Current_Curve_Price'] = summary_df['Current_Curve_Price'].fillna(default_market_price)
    
    # Calculate Arbitrage Spreads & Positions
    summary_df['Arbitrage_Premium'] = summary_df['Expected_Baseload'] - summary_df['Current_Curve_Price']
    
    def derive_signal(row):
        if row['Arbitrage_Premium'] > threshold:
            return "LONG PROMPT CURVE"
        elif row['Arbitrage_Premium'] < -threshold:
            return "SHORT PROMPT CURVE"
        else:
            return "NEUTRAL / NO-TRADE"
            
    summary_df['Recommended_Position'] = summary_df.apply(derive_signal, axis=1)
    
    # Export structured results to outputs directory to fulfill submission requirements
    os.makedirs('outputs', exist_ok=True)
    summary_df.to_csv("outputs/prompt_curve_trading_signals.csv", index=False)
    
    # Write a quick text log summarizing execution results
    with open("outputs/prompt_curve_summary.txt", "w") as f:
        f.write("=== DAY-AHEAD SPOT TO PROMPT CURVE TRANSLATION VIEW ===\n\n")
        for _, row in summary_df.iterrows():
            f.write(f"Delivery Period: {row['ISO_Week']}\n")
            f.write(f" -> Expected Spot Baseload : {row['Expected_Baseload']:.2f} EUR/MWh\n")
            f.write(f" -> Expected Spot Peakload : {row['Expected_Peakload']:.2f} EUR/MWh\n")
            f.write(f" -> Current Traded Curve   : {row['Current_Curve_Price']:.2f} EUR/MWh\n")
            f.write(f" -> Calculated Premium     : {row['Arbitrage_Premium']:.2f} EUR/MWh\n")
            f.write(f" -> RECOMMENDED ACTION     : {row['Recommended_Position']}\n")
            f.write("-" * 50 + "\n")
            
    print(" Successfully generated trading translation views in outputs/")
    return summary_df
