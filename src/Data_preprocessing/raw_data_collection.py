import pandas as pd
def compiled_data():    
    Wind_offshore= pd.read_csv(r"data\germany_offshore_hourly.csv")
    Wind_onshore= pd.read_csv(r"data\germany_onshore_hourly.csv")
    SMARD_Data = pd.read_csv(r"data\germany_smard_forecast_hourly.csv")
    fuel_Data = pd.read_csv(r"C:data\germany_fuel_prices.csv")
    Day_ahead_price=pd.read_csv(r"data\germany_hourly_day_ahead_prices.csv")

    Day_ahead_price["Datetime_UTC"] = pd.to_datetime(Day_ahead_price['DateTime'], utc=True).dt.tz_convert('Europe/Berlin')
    SMARD_Data["Datetime_UTC"] = pd.to_datetime(SMARD_Data['Datetime_UTC'], utc=True).dt.tz_convert('Europe/Berlin')

    # 1. Parse Open-Meteo's default UTC timestamps correctly
    Wind_onshore["Datetime_UTC"] = pd.to_datetime(
        Wind_onshore["timestamp_gmt"], utc=True
    )
    # 2. Convert to Germany's local timezone (Europe/Berlin)
    Wind_onshore["Datetime_UTC"] = Wind_onshore["Datetime_UTC"].dt.tz_convert(
        "Europe/Berlin"
    )
    Wind_onshore=Wind_onshore.drop(columns="timestamp_gmt")
    # 1. Parse Open-Meteo's default UTC timestamps correctly
    Wind_offshore["Datetime_UTC"] = pd.to_datetime(
        Wind_offshore["timestamp_gmt"], utc=True
    )
    # 2. Convert to Germany's local timezone (Europe/Berlin)
    Wind_onshore["Datetime_UTC"] = Wind_onshore["Datetime_UTC"].dt.tz_convert(
        "Europe/Berlin"
    )
    Wind_offshore=Wind_offshore.drop(columns="timestamp_gmt")


    Wind_data= pd.merge(Wind_offshore,Wind_onshore, on=['Datetime_UTC'], how="inner")
    master_df = pd.merge(SMARD_Data,Wind_data, on=['Datetime_UTC'], how="inner")
    fuel_Data ["Date"] = pd.to_datetime(fuel_Data["Date"]).dt.date
    master_df["Merge_Date"] = master_df["Datetime_UTC"].dt.date
    master_df = pd.merge(
        master_df, 
        fuel_Data, 
        left_on="Merge_Date", 
        right_on="Date", 
        how="left"
    )

    # 4. Clean up the temporary column used for structural matching
    master_df.drop(columns=["Merge_Date"], inplace=True)

    master_df=pd.merge(Day_ahead_price,master_df,on=['Datetime_UTC'], how="inner")

    master_df=master_df.drop(columns=["Unnamed: 0","Unnamed: 0_y","Unnamed: 0_x"])
    master_df= master_df.dropna()
    print(master_df.info())
    master_df.to_csv("data/raw_data_compiled.csv")