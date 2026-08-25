# Weather Trend Forecasting Basic Assessment

## PM Accelerator Mission

PM Accelerator highlights a mission centered on expanding access to product management education and reducing financial barriers, including support for underserved students through PMA Kids.

## Objective

This report completes the **basic assessment** for the Global Weather Repository dataset. The work covers:

- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Temperature and precipitation visualizations
- A basic time-series forecasting model using `last_updated`
- Model evaluation and future trend forecasts

## Repository Deliverables

- `scripts/run_basic_assessment.py`: end-to-end pipeline
- `output/basic_assessment/global_weather_cleaned.csv`: cleaned dataset with IQR-clipped numeric outliers
- `output/basic_assessment/tables/daily_weather_aggregates.csv`: daily time-series table used for forecasting
- `output/basic_assessment/tables/model_metrics.csv`: evaluation metrics
- `output/basic_assessment/charts/*.svg`: generated visualizations
- `output/basic_assessment/summary.json`: machine-readable summary for quick review

## Methodology

### 1. Data Cleaning & Preprocessing

- Parsed `last_updated` as the main time feature for chronological analysis.
- Checked the dataset for blank fields. The current copy of the dataset does not contain missing values.
- Detected numeric outliers with the interquartile range (IQR) rule and clipped them to reduce the impact of extreme spikes.
- Applied min-max normalization to model input features derived from the time series.

### 2. Exploratory Data Analysis

- Aggregated the weather observations into daily global averages.
- Reviewed weather-condition frequency, country-level average temperatures, and correlations among key numeric weather variables.
- Created charts for daily average temperature, daily average precipitation, and the most common weather conditions.

### 3. Forecasting

- Built daily forecasting targets from `last_updated`:
  - `avg_temperature_celsius`
  - `avg_precip_mm`
- Used a basic linear regression model with lag-based features:
  - day index
  - previous day value
  - trailing 3-day average
  - trailing 7-day average
- Compared the regression model with a naive last-value baseline.
- Evaluated performance with MAE, RMSE, and MAPE.

## Output Review

Run the pipeline locally with:

```bash
python3 scripts/run_basic_assessment.py
```

Then review:

- `output/basic_assessment/charts/daily_avg_temperature.svg`
- `output/basic_assessment/charts/daily_avg_precipitation.svg`
- `output/basic_assessment/tables/model_metrics.csv`
- `output/basic_assessment/summary.json`

## Notes

- The assessment asked for either the basic or advanced track. This repository intentionally completes the **basic** track only.
- The code is designed to run with Python standard-library modules only, which keeps setup lightweight for reviewers.
