import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
# Set seeds for absolute reproducibility (Cobblestone Rule)
np.random.seed(42)

# =====================================================================
#  AUTOMATED FEATURE SELECTION DESK
# =====================================================================
def lasso_feature_selection(X_train: pd.DataFrame, y_train: pd.Series,X_test: pd.DataFrame, X_val) -> tuple:
    """
    Uses cross-validated Lasso regularization to mathematically evaluate,
    rank, and select predictive features, dropping uninformative variance.
    """
    print("[+] Executing automated feature selection via Lasso CV...")
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Fit LassoCV to identify zero-weight coefficients
    lasso = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=2000)
    lasso.fit(X_train_scaled, y_train)
    
    # Extract features where coefficient is mathematically non-zero
    importance = np.abs(lasso.coef_)
    selected_mask = importance > 1e-4
    selected_features = X_train.columns[selected_mask].tolist()
    
    # Fail-safe: if Lasso drops everything, retain top 5 fundamental drivers
    if not selected_features:
        selected_features = ['forecast_residual_load_mwh', 'renewable_penetration_ratio', 'price_lag_24h', 'heating_degree_days', 'demand_ramp_3h_mw']
        
    print(f" Feature Selection Complete. Retained {len(selected_features)}/{X_train.shape[1]} features.")
    print(f" Dropped features: {list(set(X_train.columns) - set(selected_features))}\n")
    
    return X_train[selected_features], X_val[selected_features],X_test[selected_features], selected_features


# =====================================================================
# 3. TIME-SERIES ARCHITECTURE (LIGHTGBM)
# =====================================================================
def lightgbm_with_lasso_features(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Trains a high-density LightGBM Regressor optimized to map cyclical 
    time vectors and sudden structural grid regime spikes.
    """
    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )
    
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    return model,preds, rmse, mae

# =====================================================================
#    PIPELINE EXECUTION MASTER ORCHESTRATOR
# =====================================================================
def train_lighbgm_with_lasso_selected_features(df: pd.DataFrame,train_mask,val_mask,test_mask):
    warnings.filterwarnings('ignore')
    """
    Executes automated feature filtering, runs models across chronological splits,
    and returns production metrics alongside predictions.csv rows.
    """
    df = df.copy().sort_values('Datetime_UTC').reset_index(drop=True)
    
    # Define complete potential feature universe
    raw_features = [
 'Total_MWh', 'PV_and_Wind_MWh', 'Wind_Offshore_MWh',
       'Wind_Onshore_MWh', 'Photovoltaics_MWh', 'Other_MWh',
       'Forecasted_Grid Load_(Forecasted_Total_Consumption)_MWh',
       'offshore_wind_speed', 'offshore_wind_gusts', 'offshore_temperature',
       'offshore_wind_direction', 'onshore_temperature_2m_c',
       'onshore_wind_speed_10m_kmh', 'onshore_wind_direction_10m_deg',
       'onshore_wind_gusts_10m_kmh', 'hour',
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
    TARGET="Price_EUR_MWh"
    # Verify all columns exist in the incoming dataframe slice
    feature_space = [col for col in raw_features if col in df.columns]

    
    df_train = df[train_mask].reset_index(drop=True)
    df_val = df[val_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    X_raw_train = df_train[raw_features]
    X_raw_test = df_test[raw_features]
    X_raw_val = df_val[raw_features]
    y_train = df_train['Price_EUR_MWh']
    y_test = df_test['Price_EUR_MWh']
    y_val = df_val['Price_EUR_MWh']

    X_train,X_val, X_test, active_features = lasso_feature_selection(X_raw_train, y_train, X_raw_test,X_raw_val)

    
    # Train Architectures
    print("[+] Training Advanced Deep Learning MLP...")
    
    print("[+] Training Advanced Time-Series Booster (LightGBM)...")
    ts_model,ts_preds, ts_val_rmse, ts_val_mae = lightgbm_with_lasso_features(X_train, y_train, X_val, y_val)
    ts_test_preds = ts_model.predict(X_test)
    ts_test_rmse = np.sqrt(mean_squared_error(y_test, ts_test_preds))
    ts_test_mae = mean_absolute_error(y_test,ts_test_preds)
    
    print("\n=========================================")
    print("  LGBM WITH LASSO PERFORMANCE REPORT        ")
    print("=========================================")
    print(f"Time-Series LGBM  -> VALIDATION RMSE: {ts_val_rmse:.2f} | MAE: {ts_val_mae:.2f}\n")
    print(f"Time-Series LGBM  -> TEST RMSE: {ts_test_rmse:.2f} | MAE: {ts_test_mae:.2f}\n")


    # Extracting .values ensures everything aligns cleanly without index conflicts
    predictions_df = pd.DataFrame({
        'datetime': pd.to_datetime(df.loc[test_mask, 'Datetime_UTC'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ').values,
        'LightGBM_With_Lasso_pred': ts_test_preds,
        'y_test': y_test.values
    })

    # Save to CSV without the generic row index column
    predictions_df.to_csv("model_predictions/LightGBM_With_Lasso Predictions.csv", index=False)

    print("Predictions saved successfully!")
    # ============================================================
    #  3. CHRONOLOGICAL EVALUATION DASHBOARD
    # ============================================================
    eval_df = df.loc[test_mask, ['Datetime_UTC', TARGET]].copy()
    eval_df['LightGBM_With_Lasso_Pred'] = ts_test_preds
    eval_df['Year_Month'] = pd.to_datetime(eval_df['Datetime_UTC'], utc=True).dt.to_period('M')

    monthly_metrics = []
    for name, group in eval_df.groupby('Year_Month'):
        m_rmse = np.sqrt(mean_squared_error(group[TARGET], group['LightGBM_With_Lasso_Pred']))
        m_mae = mean_absolute_error(group[TARGET], group['LightGBM_With_Lasso_Pred'])
        monthly_metrics.append({'Month': str(name), 'RMSE': m_rmse, 'MAE': m_mae})

    df_monthly = pd.DataFrame(monthly_metrics)
    df_monthly.to_csv("outputs/LGBM_with_lasso_Test_results_monthly_analysis.csv")

    return ts_model, predictions_df,ts_val_rmse, ts_val_mae,ts_test_rmse, ts_test_mae