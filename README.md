# Weather Trend Forecasting

This repository completes the **basic assessment** for the PM Accelerator weather forecasting take-home. It analyzes the `GlobalWeatherRepository.csv` dataset, prepares the data for modeling, explores global weather patterns, visualizes temperature and precipitation trends, and builds a simple forecasting workflow based on the `last_updated` field.

## PM Accelerator Mission

PM Accelerator emphasizes expanding access to product management education, reducing financial barriers, and supporting underserved learners through initiatives such as PMA Kids. That mission is reflected in the report deliverable included in this repo.

## Assessment Scope Completed

The basic assessment required:

- Data cleaning and preprocessing
- Exploratory data analysis (EDA)
- Temperature and precipitation visualizations
- A basic forecasting model using `last_updated`
- Model evaluation with multiple metrics
- A report or presentation-style deliverable plus project documentation

This repository includes each of those items.

## Repository Structure

- `GlobalWeatherRepository.csv`: source dataset
- `scripts/run_basic_assessment.py`: end-to-end analysis pipeline
- `report/basic_assessment_report.md`: written deliverable summary
- `output/basic_assessment/`: generated cleaned data, tables, charts, and summary artifacts
- `requirements.txt`: environment note for reviewers
- `SUBMISSION_CHECKLIST.md`: final submission and GitHub reminder checklist

## How To Run

The project uses Python's standard library only.

```bash
python3 scripts/run_basic_assessment.py
```

## Generated Outputs

Running the script produces:

- `output/basic_assessment/global_weather_cleaned.csv`
- `output/basic_assessment/tables/daily_weather_aggregates.csv`
- `output/basic_assessment/tables/model_metrics.csv`
- `output/basic_assessment/charts/daily_avg_temperature.svg`
- `output/basic_assessment/charts/daily_avg_precipitation.svg`
- `output/basic_assessment/charts/top_weather_conditions.svg`
- `output/basic_assessment/summary.json`

## Methodology

### Data Cleaning & Preprocessing

- Parsed `last_updated` into a time-aware daily series.
- Checked for missing values. No blank fields were found in the provided dataset snapshot.
- Applied IQR-based clipping to numeric columns to reduce the influence of outliers.
- Applied min-max normalization to the model input features used during training.

### Exploratory Data Analysis

- Aggregated global observations into daily averages.
- Measured frequency of weather conditions across the dataset.
- Compared country-level average temperatures.
- Computed correlations across key weather variables including temperature, precipitation, humidity, cloud cover, wind, pressure, UV index, and PM2.5.

### Forecasting

Two daily targets were modeled:

- Average temperature in Celsius
- Average precipitation in millimeters

The forecasting pipeline uses lag-aware linear regression features:

- Day index
- Previous-day value
- Rolling 3-day average
- Rolling 7-day average

For context, the project also compares the regression model with a naive last-value baseline.

## Model Evaluation

Current metrics from `output/basic_assessment/tables/model_metrics.csv`:

- Temperature regression: `MAE 0.3631`, `RMSE 0.6104`, `MAPE 1.7255`
- Temperature naive baseline: `MAE 0.3160`, `RMSE 0.6587`, `MAPE 1.4977`
- Precipitation regression: `MAE 0.0013`, `RMSE 0.0016`, `MAPE 11.4344`
- Precipitation naive baseline: `MAE 0.0015`, `RMSE 0.0019`, `MAPE 12.9631`

The baseline remains strong for temperature, while the lagged regression improves precipitation error. That is a reasonable outcome for a basic time-series assessment and is explained in the report.

## Key Files For Review

- [report/basic_assessment_report.md](/Users/rifatchowdhury/Documents/GitHub/Weather-Trend-Forecasting/report/basic_assessment_report.md)
- [output/basic_assessment/summary.json](/Users/rifatchowdhury/Documents/GitHub/Weather-Trend-Forecasting/output/basic_assessment/summary.json)
- [output/basic_assessment/tables/model_metrics.csv](/Users/rifatchowdhury/Documents/GitHub/Weather-Trend-Forecasting/output/basic_assessment/tables/model_metrics.csv)

## Notes

- The repo currently focuses on the **basic** assessment only, not the advanced extension.
- The assessment also asks for a short demo video and GitHub submission link. Those are manual submission steps and are listed in `SUBMISSION_CHECKLIST.md`.
