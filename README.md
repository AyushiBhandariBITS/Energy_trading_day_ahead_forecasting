# German Power Fair Value Pipeline (DE Bidding Zone)

An end-to-end prototype pipeline that builds advanced structural features, forecasts out-of-sample German Day-Ahead power prices using a hybrid Machine Learning ensemble model, translates spot forecasts into a tradable Prompt Curve view, and programmatically embeds an automated AI risk analyst loop.

## 🚀 Single-Command Execution
Run the full production pipeline end-to-end with one command:
```bash
python main.py
```

## 📊 Chosen Market & Approach
* **Target Market:** Germany (DE)
* **Forecasting Option:** Option A (Next-day hourly Day-Ahead pricing vectors)
* **Out-of-Sample Window:** `2025-10-01` - `2025-08-29`

---

## 🏗️ Repository Architecture
```text
Ayushi/
├───_ report.pdf
├───_  main.py
├───_  predictions.csv
├───_  README.md
├───_  requirements.txt
│
├───ai_logs
│       ai_pointer_collection.csv
│       ai_pointer_collection_with_insight.csv
│
├───data
│   │   clean_feature_engineered_data.csv
│   │   germany_fuel_prices.csv
│   │   germany_hourly_day_ahead_prices.csv
│   │   germany_offshore_hourly.csv
│   │   germany_onshore_hourly.csv
│   │   germany_smard_forecast_hourly.csv
│   │   raw_data_compiled.csv
│
├───figures
│       heatmap.png
│
├───models
│       FINAL_LGBM_with_Catboost_model.pkl
│       LGBM_model.pkl
│       LGBM_with_Lasso_model.pkl
│       linear_ridge_model.pkl
│
├───model_predictions
│       Ensemble_predictions.csv
│       LightGBM Predictions.csv
│       LightGBM_With_Lasso Predictions.csv
│       Linear Ridge Regression Predictions.csv
│       Linear_Ridge_Regression_results_monthly_analysis.csv
│
├───outputs
│       Final_model_results_monthly_analysis.csv
│       LGBM_Test_results_monthly_analysis.csv
│       LGBM_with_lasso_Test_results_monthly_analysis.csv
│       Linear Ridge Regression Predictions.csv
│       Linear_Ridge_Regression_results_monthly_analysis.csv
│       model_performance_benchmarks.csv
│       prompt_curve_summary.txt
│       prompt_curve_trading_signals.csv
│       Regression_coefficient_features.csv
│
├───src
│   │   ai_integration.py
│   │   prompt_curve_trading.py
│   │
│   ├───Data_preprocessing
│   │   │   Feature_engineering.py
│   │   ││└─── raw_data_collection.py

│   │
│   ├───Download_Data
│   │   │   ENTSOE_Day_ahead_price_Data.py
│   │   │   fuel_prices_data.py
│   │   │   SMARD_features_Data.py
│   │   │└─── wind_data_collection.py
│   │
│   ├───models
        │   base_model_lightgbm.py
        │   ensemble_model_best_version.py
        │   lasso_with_lighbgm.py
        │└───  ridge_regression.py

```
---

## 🛠️ Installation & Setup

1. **Clone and Navigate into the Repository:**
   ```bash
   cd Ayushi
   ```

2. **Install Pinned Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Create a `.env` file in the root directory (do **not** commit this file). Add your secured API configuration:
   ```text
   OPENAI_API_KEY=your_actual_api_key_here
   ENTSOE_API_KEY=your_actual_api_key_here
   ```

---

## 📋 Evaluation Checklist Compliance
* **Deterministic Results:** Random seeds are locked (`seed=42`) across all boosting variants to ensure mathematical repeatability.
* **Performance Bound:** The full feature generation, model ensembling, curve translation, and LLM row injection run entirely in under **3 minutes** on a standard laptop.
* **Leakage Prevention:** Zero look-ahead bias. All commodity drivers (Gas, Coal, Carbon) and autoregressive target variables are strictly shifted by `t-24` or greater.
* **Graceful Key Fallback:** If an active `OPENAI_API_KEY` is not detected in the environment variables, the AI microservice switches automatically to an offline fallback writer. This populates `outputs/` and `ai_logs/` with pre-cached text streams so the pipeline completes without crashing.

***