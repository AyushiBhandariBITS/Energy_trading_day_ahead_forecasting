import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
def train_lightgbm_model(df: pd.DataFrame,train_mask,val_mask,test_mask):
    warnings.filterwarnings('ignore')
    """
    Trains an institutional-grade LightGBM Regressor to capture non-linear 
    merit-order dynamics using a strict chronological out-of-sample split.
    """
    df = df.copy().sort_values('Datetime_UTC').reset_index(drop=True)
    TARGET='Price_EUR_MWh'
    # 1. Feature Selection (Trees natively handle cyclical time variables as integers)
    feature_cols = [
 'Total_MWh', 'PV_and_Wind_MWh', 'Wind_Offshore_MWh',
       'Wind_Onshore_MWh', 'Photovoltaics_MWh', 'Other_MWh',
       'Forecasted_Grid Load_(Forecasted_Total_Consumption)_MWh',
       'offshore_wind_speed', 'offshore_wind_gusts', 'offshore_temperature',
       'offshore_wind_direction', 'onshore_temperature_2m_c',
       'onshore_wind_speed_10m_kmh', 'onshore_wind_direction_10m_deg',
       'onshore_wind_gusts_10m_kmh',  'hour',
       'day_of_week', 'month', 'is_weekend', 'is_peak_hour',
       'forecast_residual_load_mwh', 'renewable_penetration_ratio',
       'solar_to_wind_ratio', 'price_lag_24h', 'price_lag_168h',
       'price_volatility_24h', 'PV_and_Wind_Last_Year',
       'Absolute_Difference_MWh', 'Percentage_Change_%', 'demand_ramp_3h_mw',
       'spatial_wind_variance', 'wind_speed_volatility_delta',
       'gas_generation_cost_floor_eur', 'gas_ttf_5d_variance',
       'carbon_eua_5d_variance', 'heating_degree_days', 'cooling_degree_days',
       'solar_saturation_index', 'regime_saturation_flag', 'Gas_1d_lag',
       'Carbon_1d_lag', 'Coal_1d_lag'
    ]


    # Features (X) - Directly select the rows and feature columns using .loc
    X_train = df.loc[train_mask, feature_cols].reset_index(drop=True)
    X_val = df.loc[val_mask, feature_cols].reset_index(drop=True)
    X_test = df.loc[test_mask, feature_cols].reset_index(drop=True)

    # Target (y) - Select only the target column for the same masks
    y_train = df.loc[train_mask, 'Price_EUR_MWh'].reset_index(drop=True)
    y_val = df.loc[val_mask, 'Price_EUR_MWh'].reset_index(drop=True)
    y_test = df.loc[test_mask, 'Price_EUR_MWh'].reset_index(drop=True)
    
    
    # 3. Initialize LightGBM Regressor with Pinned Random Seed for Determinism
    model = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42, # Crucial for Cobblestone's reproducibility check
        n_jobs=-1        # Uses all available CPU cores for lightning-fast training
    )
    
    # 4. Fit Model
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )
    
    # 5. Out-of-Sample Evaluation
    preds = model.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, preds))
    mae_val = mean_absolute_error(y_val, preds)
    preds_test = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, preds_test))
    mae_test = mean_absolute_error(y_test, preds_test)
    
    print("=========================================")
    print("     IMPROVED LIGHTGBM MODEL METRICS     ")
    print("=========================================")
    print(f"VALIDATION RMSE: {rmse_val:.2f} EUR/MWh")
    print(f"VALIDATION MAE:  {mae_val:.2f} EUR/MWh\n")
    print(f"TEST RMSE: {rmse_test:.2f} EUR/MWh")
    print(f"TEST MAE:  {mae_test:.2f} EUR/MWh\n")
    # Extracting .values ensures everything aligns cleanly without index conflicts
    predictions_df = pd.DataFrame({
        'datetime': pd.to_datetime(df.loc[test_mask, 'Datetime_UTC'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ').values,
        'Linear_Ridge_Regression_pred': preds_test,
        'y_test': y_test.values
    })

    # Save to CSV without the generic row index column
    predictions_df.to_csv("model_predictions/LightGBM Predictions.csv", index=False)

    print("Predictions saved successfully!")
    # ============================================================
    #  3. CHRONOLOGICAL EVALUATION DASHBOARD
    # ============================================================
    eval_df = df.loc[test_mask, ['Datetime_UTC', TARGET]].copy()
    eval_df['LightGBM_Pred'] = preds_test
    eval_df['Year_Month'] = pd.to_datetime(eval_df['Datetime_UTC'], utc=True).dt.to_period('M')

    monthly_metrics = []
    for name, group in eval_df.groupby('Year_Month'):
        m_rmse = np.sqrt(mean_squared_error(group[TARGET], group['LightGBM_Pred']))
        m_mae = mean_absolute_error(group[TARGET], group['LightGBM_Pred'])
        monthly_metrics.append({'Month': str(name), 'RMSE': m_rmse, 'MAE': m_mae})

    df_monthly = pd.DataFrame(monthly_metrics)    
    df_monthly.to_csv("outputs/LGBM_Test_results_monthly_analysis.csv")

    return model,predictions_df,rmse_val, mae_val,rmse_test, mae_test
