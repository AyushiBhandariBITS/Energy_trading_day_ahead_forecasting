import pandas as pd
def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers advanced market and temporal features from raw SMARD data
    without creating look-ahead bias or data leakage.
    """
    # Ensure DataFrame is sorted chronologically
    df = df.copy()
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.sort_values('DateTime').reset_index(drop=True)
    
    # ==========================================
    # 1. TEMPORAL & CALENDAR FEATURES
    # ==========================================
    df['hour'] = df['DateTime'].dt.hour
    df['day_of_week'] = df['DateTime'].dt.dayofweek
    df['month'] = df['DateTime'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # German peak pricing hours: 8-12 and 17-21
    df['is_peak_hour'] = df['hour'].isin([8,9,10,11,12,17,18,19,20,21]).astype(int)
    
    # ==========================================
    # 2. FUNDAMENTAL MARKET COUPLING FEATURES
    # ==========================================
    # Core Residual Load Metric
    df['forecast_residual_load_mwh'] = (
        df['Forecasted_Grid Load_(Forecasted_Total_Consumption)_MWh'] - df['PV_and_Wind_MWh']
    )
    
    # Renewable Generation Penetration Ratio
    # Avoid division by zero by adding a tiny epsilon
    df['renewable_penetration_ratio'] = (
        df['PV_and_Wind_MWh'] / (df['Forecasted_Grid Load_(Forecasted_Total_Consumption)_MWh'] + 1e-5)
    )
    
    # Solar vs Wind Mix Ratio (Tells if supply is stable wind or volatile solar)
    df['solar_to_wind_ratio'] = (
        df['Photovoltaics_MWh'] / (df['Wind_Onshore_MWh'] + df['Wind_Offshore_MWh'] + 1e-5)
    )
    
    # Target prices must ALWAYS be lagged so we don't predict today using today's price
    df['price_lag_24h'] = df['Price_EUR_MWh'].shift(24) # Yesterday's price at this exact hour
    df['price_lag_168h'] = df['Price_EUR_MWh'].shift(168) # Last week's price at this exact hour
    
    # Rolling market volatility over the past 24 hours
    df['price_volatility_24h'] = df['Price_EUR_MWh'].shift(1).rolling(window=24).std()

    last_year_df = df[['DateTime', 'PV_and_Wind_MWh']].copy()
    last_year_df['DateTime'] = df['DateTime'] + pd.DateOffset(years=1)
    last_year_df.rename(columns={'PV_and_Wind_MWh': 'PV_and_Wind_Last_Year'}, inplace=True)

# 3. Merge the historical data back into your main dataframe
    df = pd.merge(df, last_year_df, on='DateTime', how='left')

# 4. Calculate comparisons
    df['Absolute_Difference_MWh'] = df['PV_and_Wind_MWh'] - df['PV_and_Wind_Last_Year']
    df['Percentage_Change_%'] = (df['Absolute_Difference_MWh'] / df['PV_and_Wind_Last_Year']) * 100
    df['demand_ramp_3h_mw'] = df['Forecasted_Grid Load_(Forecasted_Total_Consumption)_MWh'] - df['Forecasted_Grid Load_(Forecasted_Total_Consumption)_MWh'].shift(3)
    # 1. Compute rolling spatial variance across the columns for each row
    wind_cols = ['onshore_wind_speed_10m_kmh', 'offshore_wind_speed'] # Onshore and Offshore
    df['spatial_wind_variance'] = df[wind_cols].var(axis=1).fillna(0)

    # 2. Subtract yesterday's spatial variance at the exact same hour to get the structural Delta
    df['wind_speed_volatility_delta'] = df['spatial_wind_variance'] - df['spatial_wind_variance'].shift(24)
    # Calculate the strict structural economic cost baseline for gas units
    # This acts as a predictive price anchor without using the target variable
    df['gas_generation_cost_floor_eur'] = (df['Gas'].shift(24) / 0.50) + (df['Carbon'].shift(24) * 0.202)
    df['gas_ttf_5d_variance'] = df['Gas'].shift(24).rolling(window=120).var()
    df['carbon_eua_5d_variance'] = df['Carbon'].shift(24).rolling(window=120).var()
    df['heating_degree_days'] = df['onshore_temperature_2m_c'].apply(lambda x: max(0, 15.0 - x))
    df['cooling_degree_days'] = df['onshore_temperature_2m_c'].apply(lambda x: max(0, x - 22.0))
    df['solar_saturation_index'] = ( df['Photovoltaics_MWh'] / (df['Forecasted_Grid Load_(Forecasted_Total_Consumption)_MWh'] + 1e-5))
    df['regime_saturation_flag'] = (df['solar_saturation_index'] >= 0.70).astype(int)

    df['Gas_1d_lag'] = df['Gas'].shift(24)
    df['Carbon_1d_lag'] = df['Carbon'].shift(24)
    df['Coal_1d_lag'] = df['Coal'].shift(24)

    # Drop rows at the very beginning that contain NaN values due to lagging
    df = df.dropna().reset_index(drop=True)
    df.to_csv("data/clean_feature_engineered_data.csv")
    print("Saved.data/clean_feature_engineered_data.csv")
    return df
