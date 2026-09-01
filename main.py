
from src.Download_Data.ENTSOE_Day_ahead_price_Data import *
from src.Download_Data.SMARD_features_Data import collect_smard_features
from src.Download_Data.wind_data_collection import *
from src.Download_Data.fuel_prices_data import *

from src.Data_preprocessing.raw_data_collection import compiled_data
from src.Data_preprocessing.Feature_engineering import feature_engineering

from src.models.ridge_regression import linear_ridge_regression
from src.models.base_model_lightgbm import train_lightgbm_model
from src.models.lasso_with_lighbgm import *
from src.models.ensemble_model_best_version import *

from src.prompt_curve_trading import *

from src.ai_integration import *

import pandas as pd
import joblib
from tabulate import tabulate


#1. Ingest Data

start_date_time="2023-01-01 00:00:00"
start_date="2023-01-01"
end_date_time="2026-09-02 00:00:00"
end_date="2026-09-02"

#collect_day_ahead_price(start_date_time,end_date_time)
#collect_smard_features(start_date_time,end_date_time)
#collect_wind_data_onshore()
#collect_wind_data_offshore(start_date,end_date)
#collect_fuel_prices(start_date,end_date)

#compiled_data()
#raw_data=pd.read_csv("data/raw_data_compiled.csv")
#df = feature_engineering(raw_data)

df = pd.read_csv("data/clean_feature_engineered_data.csv")
df=df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0','Carbon', 'Coal', 'Gas' ])


#2. correlation finding and graphs for visualization
#visualization_analysis(df)

train_mask = (df['Datetime_UTC'] >= '2024-01-01 00:00:00') & (df['Datetime_UTC'] <= '2025-06-30 23:00:00')
val_mask   = (df['Datetime_UTC'] >= '2025-07-01 00:00:00') & (df['Datetime_UTC'] <= '2025-09-30 23:00:00')
test_mask  = (df['Datetime_UTC'] >= '2025-10-01 00:00:00')


#BASELINE MODELS

#3. Train Test Validate baseline models- Linear ridge regression and LightBGM
print("Base Model 1: Linear Ridge Regression\n\n")
linear_ridge_model,linear_ridge_scaler,LRR_predictions_df,LRR_val_rmse,LRR_val_mae,LRR_test_rmse,LRR_test_mae = linear_ridge_regression(df,train_mask,val_mask,test_mask)
joblib.dump(linear_ridge_model, 'models/linear_ridge_model.pkl')

print("Base Model 2: LightGBM Regressor\n\n")
LGBM_model, LGBM_predictions_df,LGBM_val_rmse, LGBM_val_mae,LGBM_test_rmse, LGBM_test_mae= train_lightgbm_model(df,train_mask,val_mask,test_mask)
joblib.dump(LGBM_model, 'models/LGBM_model.pkl')

print("Base Model 3: LightGBM Regressor TRAINED ON LASSO Selected Features\n\n")
LGBM_with_Lasso_model, LGBM_with_Lasso_predictions_df,LGBM_with_Lasso_val_rmse, LGBM_with_Lasso_val_mae,LGBM_with_Lasso_test_rmse, LGBM_with_Lasso_test_mae=train_lighbgm_with_lasso_selected_features(df,train_mask,val_mask,test_mask)
joblib.dump(LGBM_with_Lasso_model, 'models/LGBM_with_Lasso_model.pkl')

#4. Train Test Validate LightBGM+CatBoost Emsemble Model
#FINAL MODEL
print("Improved model: LightGBM and Catboost Ensemble model")
Final_model,final_predictions_df, ens_val_rmse,  ens_val_mae, ens_test_rmse, ens_test_mae =improved_ensemble_model(df,train_mask,val_mask,test_mask)
joblib.dump(Final_model, 'models/FINAL_LGBM_with_Catboost_model.pkl')


#VALIDATION AND TEST SCORES COMBINED DATA
metrics_data = {
    "Model_Name": [
        "Linear Ridge Regression",
        "Time-Series LightGBM (All Features)",
        "LightGBM with LASSO Selection",
        "Hybrid LGBM + CatBoost Ensemble (FINAL)"
    ],
    "Validation_RMSE": [LRR_val_rmse, LGBM_val_rmse, LGBM_with_Lasso_val_rmse, ens_val_rmse],
    "Validation_MAE": [LRR_val_mae, LGBM_val_mae, LGBM_with_Lasso_val_mae, ens_val_mae],
    "Test_RMSE": [LRR_test_rmse, LGBM_test_rmse, LGBM_with_Lasso_test_rmse, ens_test_rmse],
    "Test_MAE": [LRR_test_mae, LGBM_test_mae, LGBM_with_Lasso_test_mae, ens_test_mae]
}

# 2. Build the structured Pandas DataFrame and round float values for professional tracking
df_metrics = pd.DataFrame(metrics_data)
for col in df_metrics.columns:
    if df_metrics[col].dtype == 'float64':
        df_metrics[col] = df_metrics[col].round(2)

# 3. Print a polished scannable table to your execution terminal logs
print("\n=========================================================================")
print("             MANDATORY CASE STUDY PERFORMANCE BENCHMARK INDEX            ")
print("=========================================================================")
# Use tabulate if available, fallback to standard pandas printing if not
try:
    print(tabulate(df_metrics, headers='keys', tablefmt='psql', showindex=False))
except ImportError:
    print(df_metrics.to_string(index=False))
print("=========================================================================\n")

# 4. Save the finalized tabular matrix strictly as a CSV tracking checkpoint
os.makedirs("outputs", exist_ok=True)
csv_output_path = "outputs/model_performance_benchmarks.csv"
df_metrics.to_csv(csv_output_path, index=False)

print(f" Export Complete: Benchmarks archived at local disk footprint: '{csv_output_path}'")

predictions_df =pd.read_csv("predictions.csv")
#Prompt curve trading Translation

#PROMPT TRADE CURVE
weekly_forecasts, monthly_forecasts = translate_predictions_to_proper_curve(predictions_df)

# 2. Provide a dictionary of traded prompt forward prices for the target delivery weeks.
# In live environments, these values tie directly to an EEX or Trayport data feed.
CURRENT_MARKET_CURVE_PRICES = {
    "2026-W36": 82.10,  # EUR/MWh
    "2026-W37": 89.45,  # EUR/MWh
    "2026-W38": 85.00   # EUR/MWh
}

# Set execution trigger hurdles (e.g., mispricing must cross 2.50 EUR/MWh)
EXECUTION_HURDLE_EUR = 2.50

signals_df = generate_trading_signals(
    weekly_forecasts=weekly_forecasts,
    current_market_curves=CURRENT_MARKET_CURVE_PRICES,
    threshold=EXECUTION_HURDLE_EUR
)

# Stream structural findings directly to terminal trace logs
print("\n=== SYSTEM EXECUTION TRACE LOGS ===")
for idx, row in signals_df.iterrows():
    print(f"[{row['ISO_Week']}] Premium: {row['Arbitrage_Premium']:+6.2f} EUR/MWh -> Trigger: {row['Recommended_Position']}")
    
    

#AI INTEGRATION- USING OPEN-AI
#Done only from 20th August 2026- 29th August 2026 due to lack of tokens


#INPUT_CSV = "ai_logs/ai_pointer_collection.csv"
#OUTPUT_CSV = "ai_logs/ai_pointer_collection_with_insight.csv"
#
#PREDICTION_COL = "ensemble_pred" 
#
## Execute batch job
#batch_process_predictions(INPUT_CSV, OUTPUT_CSV,  PREDICTION_COL)
