import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
def improved_ensemble_model(df,train_mask,val_mask,test_mask):

    warnings.filterwarnings('ignore')
    # 1. Feature Selection Mapping

    FEATURES =[
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
    TARGET = 'Price_EUR_MWh'
    
    # Ensure proper datetime parsing
    df = df.copy().sort_values('Datetime_UTC').reset_index(drop=True)
    
    X_train, y_train = df.loc[train_mask, FEATURES], df.loc[train_mask, TARGET]
    X_val, y_val     = df.loc[val_mask, FEATURES], df.loc[val_mask, TARGET]
    X_test, y_test   = df.loc[test_mask, FEATURES], df.loc[test_mask, TARGET]
    
    # 3. Base Model 1: LightGBM
    lgb_model = LGBMRegressor(
        objective='regression_l1', 
        n_estimators=3000,
        learning_rate=0.03,
        num_leaves=63,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training LightGBM Engine...")
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[])
    
    # 4. Base Model 2: CatBoost
    cat_model = CatBoostRegressor(
        loss_function='MAE',
        iterations=3000,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=0 # Suppress internal epoch printing for cleaner final dashboard
    )
    
    print("Training CatBoost Engine...\n")
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=150)
    
    # ==========================================
    #  METRICS ANALYSIS GENERATION
    # ==========================================
    
    # Generate Base Model Predictions for both Validation and Test splits
    val_preds_lgb = lgb_model.predict(X_val)
    val_preds_cat = cat_model.predict(X_val)
    X_meta_val = np.column_stack((val_preds_lgb, val_preds_cat))
    
    test_preds_lgb = lgb_model.predict(X_test)
    test_preds_cat = cat_model.predict(X_test)
    X_meta_test = np.column_stack((test_preds_lgb, test_preds_cat))
    
    # Train Stacking Meta-Learner (Blender) on Validation Set
    meta_learner = Ridge(alpha=1.0, positive=True)
    meta_learner.fit(X_meta_val, y_val)
    
    # Generate Final Blended Predictions
    final_val_preds = meta_learner.predict(X_meta_val)
    final_test_preds = meta_learner.predict(X_meta_test)
    
    # Helper function to compute performance scores
    def compute_scores(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae
    
    # Compute metrics for all variations
    lgb_val_rmse, lgb_val_mae = compute_scores(y_val, val_preds_lgb)
    lgb_test_rmse, lgb_test_mae = compute_scores(y_test, test_preds_lgb)
    
    cat_val_rmse, cat_val_mae = compute_scores(y_val, val_preds_cat)
    cat_test_rmse, cat_test_mae = compute_scores(y_test, test_preds_cat)
    
    ens_val_rmse, ens_val_mae = compute_scores(y_val, final_val_preds)
    ens_test_rmse, ens_test_mae = compute_scores(y_test, final_test_preds)
    
    # ==========================================
    #  FINAL COMPARATIVE PERFORMANCE DASHBOARD
    # ==========================================
    print("=" * 60)
    print(f"{'MODEL ARCHITECTURE':<25} | {'VALIDATION SCORE':<15} | {'TEST SCORE':<15}")
    print(f"{'':<25} | {'(RMSE / MAE)':<15} | {'(RMSE / MAE)':<15}")
    print("=" * 60)
    print(f"{'1. LightGBM Base':<25} | {lgb_val_rmse:>6.2f} / {lgb_val_mae:<6.2f} | {lgb_test_rmse:>6.2f} / {lgb_test_mae:<6.2f}")
    print(f"{'2. CatBoost Base':<25} | {cat_val_rmse:>6.2f} / {cat_val_mae:<6.2f} | {cat_test_rmse:>6.2f} / {cat_test_mae:<6.2f}")
    print("-" * 60)
    print(f"{' Stacked Meta-Ensemble':<25} | \033[1m{ens_val_rmse:>6.2f} / {ens_val_mae:<6.2f}\033[0m | \033[1m{ens_test_rmse:>6.2f} / {ens_test_mae:<6.2f}\033[0m")
    print("=" * 60)

    
    predictions_df = df.loc[test_mask, ['Datetime_UTC', TARGET]].copy()
    predictions_df['ensemble_pred'] = final_test_preds
    predictions_df['Year_Month'] = pd.to_datetime(predictions_df['Datetime_UTC'], utc=True).dt.to_period('M')

    ai_df=df.loc[test_mask,['Datetime_UTC','Gas_1d_lag','Carbon_1d_lag','Forecasted_Grid Load_(Forecasted_Total_Consumption)_MWh', 'Photovoltaics_MWh','PV_and_Wind_MWh','renewable_penetration_ratio','price_lag_24h',TARGET]]
    ai_df['ensemble_pred'] = final_test_preds

    ai_df.to_csv("ai_logs/ai_pointer_collection.csv")



    monthly_metrics = []
    for name, group in predictions_df.groupby('Year_Month'):
        m_rmse = np.sqrt(mean_squared_error(group[TARGET], group['ensemble_pred']))
        m_mae = mean_absolute_error(group[TARGET], group['ensemble_pred'])
        monthly_metrics.append({'Month': str(name), 'RMSE': m_rmse, 'MAE': m_mae})

    df_monthly = pd.DataFrame(monthly_metrics)
    print(df_monthly)
    predictions_df = pd.DataFrame({
  'datetime': pd.to_datetime(df.loc[test_mask, 'Datetime_UTC'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ').values,
          
        'y_pred': final_test_preds,
        'y_test': y_test.values
    })

    # Save to CSV without the generic row index column
    predictions_df.to_csv("model_predictions/Ensemble_predictions.csv", index=False)
    predictions_df.to_csv("predictions.csv", index=False)

    print("Predictions saved successfully!")
        # ============================================================
    #   CHRONOLOGICAL EVALUATION DASHBOARD
    # ============================================================
    eval_df = df.loc[test_mask, ['Datetime_UTC', TARGET]].copy()
    eval_df['LightGBM_Pred'] = final_test_preds
    eval_df['Year_Month'] = pd.to_datetime(eval_df['Datetime_UTC'], utc=True).dt.to_period('M')

    monthly_metrics = []
    for name, group in eval_df.groupby('Year_Month'):
        m_rmse = np.sqrt(mean_squared_error(group[TARGET], group['LightGBM_Pred']))
        m_mae = mean_absolute_error(group[TARGET], group['LightGBM_Pred'])
        monthly_metrics.append({'Month': str(name), 'RMSE': m_rmse, 'MAE': m_mae})

    df_monthly = pd.DataFrame(monthly_metrics)    
    df_monthly.to_csv("outputs/Final_model_results_monthly_analysis.csv")
    return meta_learner,predictions_df, ens_val_rmse,  ens_val_mae, ens_test_rmse, ens_test_mae