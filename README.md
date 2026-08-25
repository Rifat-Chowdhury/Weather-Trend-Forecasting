# Weather Trend Forecasting

This project explores the Global Weather Repository, a collection of weather observations from locations around the world. It cleans the data, looks for patterns in temperature and precipitation, and uses recent daily values to make short-term forecasts.

The analysis uses `last_updated` as the time field. Rather than forecasting a single city, it groups observations by day to show overall global trends in the dataset.

## What Is Included

- Data cleaning, including checks for missing values and treatment of extreme numeric values
- Daily temperature and precipitation trend charts
- A look at common weather conditions, country-level temperatures, and relationships between weather variables
- Simple forecasts for average temperature and precipitation
- Model evaluation using MAE, RMSE, and MAPE

## Quick Start

No third-party packages are required. From the project folder, run:

```bash
python3 scripts/run_basic_assessment.py
```

The script reads `GlobalWeatherRepository.csv` and writes the results to `output/basic_assessment/`.

## Results To Review

- `output/basic_assessment/charts/daily_avg_temperature.svg`
- `output/basic_assessment/charts/daily_avg_precipitation.svg`
- `output/basic_assessment/charts/top_weather_conditions.svg`
- `output/basic_assessment/tables/daily_weather_aggregates.csv`
- `output/basic_assessment/tables/model_metrics.csv`
- `output/basic_assessment/summary.json`

The full written discussion is in [basic_assessment_report.md](report/basic_assessment_report.md).

## Approach

### Preparing the Data

The dataset was checked for blank fields; none were found in this version of the file. Numeric outliers were identified with the interquartile range (IQR) method and clipped so unusually large values would not dominate the analysis. The model inputs were scaled with min-max normalization.

### Exploring the Data

The observations were grouped into daily averages. The analysis then compares weather-condition frequency, country-level average temperatures, and correlations between temperature, precipitation, humidity, cloud cover, wind, pressure, UV index, and PM2.5.

### Forecasting

Two daily values are forecast:

- Average temperature in Celsius
- Average precipitation in millimeters

The model is a linear regression that uses the previous day's value and rolling 3-day and 7-day averages. Its results are compared with a simple baseline that uses the previous day's value as the next day's prediction.

## Model Results

| Target | Model | MAE | RMSE | MAPE |
| --- | --- | ---: | ---: | ---: |
| Temperature | Linear regression | 0.3631 | 0.6104 | 1.7255% |
| Temperature | Previous-day baseline | 0.3160 | 0.6587 | 1.4977% |
| Precipitation | Linear regression | 0.0013 | 0.0016 | 11.4344% |
| Precipitation | Previous-day baseline | 0.0015 | 0.0019 | 12.9631% |

The temperature baseline has a lower average error, while the regression model has a lower RMSE. For precipitation, the regression model performs better across all three measures. Since precipitation values are often close to zero, its percentage error should be read with care.

## Project Layout

- `GlobalWeatherRepository.csv` - source data
- `scripts/run_basic_assessment.py` - analysis and forecasting script
- `report/basic_assessment_report.md` - project report
- `output/basic_assessment/` - generated data tables, charts, and summary
- `requirements.txt` - Python environment note

## PM Accelerator

PM Accelerator works to make product management education more accessible, including through PMA Kids, which supports students from underserved communities. More information is available at [pmaccelerator.io](https://www.pmaccelerator.io/).
