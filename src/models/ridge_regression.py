import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings

def linear_ridge_regression(df: pd.DataFrame,train_mask,val_mask,test_mask):

    warnings.filterwarnings('ignore')

    # Ensure proper datetime format
    df['Datetime_UTC'] = pd.to_datetime(df['Datetime_UTC'])

    FEATURES = [
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
    TARGET = 'Price_EUR_MWh'

    # Chronological Splitting
    X_train_raw = df.loc[train_mask, FEATURES]
    y_train     = df.loc[train_mask, TARGET]
    X_val_raw   = df.loc[val_mask, FEATURES]
    y_val       = df.loc[val_mask, TARGET]
    X_test_raw  = df.loc[test_mask, FEATURES]
    y_test      = df.loc[test_mask, TARGET]

    # ============================================================
    #  LINEAR PIPELINE PREPARATION (IMPUTE + SCALE)
    # ============================================================
    # Linear models crash or fail with NaNs. Impute missing spots with training medians.
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp   = imputer.transform(X_val_raw)
    X_test_imp  = imputer.transform(X_test_raw)

    # Scale features so that weights represent direct global feature importance
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled   = scaler.transform(X_val_imp)
    X_test_scaled  = scaler.transform(X_test_imp)

    # ============================================================
    # RIDGE MODEL TRAINING
    # ============================================================
    print("Training Linear Ridge Regression Baseline Engine...")
    # alpha=10.0 provides minor L2 shrinkage to prevent extreme collinear weights
    ridge_model = Ridge(alpha=10.0, random_state=42)
    ridge_model.fit(X_train_scaled, y_train)

    # Generate baseline vectors
    val_preds = ridge_model.predict(X_val_scaled)
    test_preds = ridge_model.predict(X_test_scaled)

    # Helper function to compute performance scores
    def compute_scores(y_true, y_pred):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        return rmse, mae

    val_rmse, val_mae = compute_scores(y_val, val_preds)
    test_rmse, test_mae = compute_scores(y_test, test_preds)

    # ============================================================
    #  SAVE TEST RESULTS & PREDICTIONS
    # ============================================================
    # Extracting .values ensures everything aligns cleanly without index conflicts
    predictions_df = pd.DataFrame({
        'datetime': pd.to_datetime(df.loc[test_mask, 'Datetime_UTC'], utc=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ').values,
        'Linear_Ridge_Regression_pred': test_preds,
        'y_test': y_test.values
    })

    # Save to CSV without the generic row index column
    predictions_df.to_csv("outputs/Linear Ridge Regression Predictions.csv", index=False)

    print("Predictions saved successfully!")
    # ============================================================
    #  3. CHRONOLOGICAL EVALUATION DASHBOARD
    # ============================================================
    eval_df = df.loc[test_mask, ['Datetime_UTC', TARGET]].copy()
    eval_df['Ridge_Pred'] = test_preds
    eval_df['Year_Month'] = pd.to_datetime(eval_df['Datetime_UTC'], utc=True).dt.to_period('M')

    monthly_metrics = []
    for name, group in eval_df.groupby('Year_Month'):
        m_rmse = np.sqrt(mean_squared_error(group[TARGET], group['Ridge_Pred']))
        m_mae = mean_absolute_error(group[TARGET], group['Ridge_Pred'])
        monthly_metrics.append({'Month': str(name), 'RMSE': m_rmse, 'MAE': m_mae})

    df_monthly = pd.DataFrame(monthly_metrics)    
    df_monthly.to_csv("model_predictions/Linear_Ridge_Regression_results_monthly_analysis.csv")

    print("\n" + "=" * 60)
    print(f" LINEAR RIDGE REGRESSION SUMMARY PERFORMANCE")
    print("=" * 60)
    print(f"Validation Target Fit: RMSE = {val_rmse:.2f}  |  MAE = {val_mae:.2f}")
    print(f"Total Test Target Fit: RMSE = {test_rmse:.2f}  |  MAE = {test_mae:.2f}")
    print("=" * 60)

    print(f"\n{'TEST MONTH':<15} | {'RMSE':<15} | {'MAE':<15}")
    print("-" * 60)
    for _, row in df_monthly.iterrows():
        print(f"{row['Month']:<15} | {row['RMSE']:>13.2f} | {row['MAE']:>13.2f}")
    print("=" * 60)

    # ============================================================
    #   COEFFICIENT WEIGHT DISTRIBUTION REPORT
    # ============================================================
    importance_df = pd.DataFrame({
        'Feature': FEATURES,
        'Coefficient_Weight': ridge_model.coef_
    })
    # Rank features by absolute magnitude of impact
    importance_df['Absolute_Weight'] = importance_df['Coefficient_Weight'].abs()
    importance_df = importance_df.sort_values(by='Absolute_Weight', ascending=False).drop(columns=['Absolute_Weight'])

    print("\n" + " LINEAR COEFFICIENT FEATURE IMPORTANCE MATRIX")
    print(" (Ranked from highest magnitude to lowest magnitude)")
    print("=" * 60)
    print(f"{'FEATURE NAME':<40} | {'COEFFICIENT WEIGHT':<15}")
    print("-" * 60)
    for _, row in importance_df.iterrows():
        print(f"{row['Feature']:<40} | {row['Coefficient_Weight']:>18.4f}")
    print("=" * 60)
    print(f"Model Intercept (Base Pricing Floor Constant): {ridge_model.intercept_:.4f} EUR/MWh")

    importance_df.to_csv("outputs/Regression_coefficient_features.csv")
    return ridge_model,scaler,predictions_df,val_rmse, val_mae,test_rmse, test_mae
