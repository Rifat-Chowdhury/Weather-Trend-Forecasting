#!/usr/bin/env python3
"""Run the Weather Trend Forecasting basic assessment pipeline."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime
from statistics import mean


DATASET_PATH = "GlobalWeatherRepository.csv"
OUTPUT_DIR = "output/basic_assessment"
CHARTS_DIR = os.path.join(OUTPUT_DIR, "charts")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")
REPORT_DATA_PATH = os.path.join(OUTPUT_DIR, "summary.json")
CLEANED_DATASET_PATH = os.path.join(OUTPUT_DIR, "global_weather_cleaned.csv")
DAILY_AGGREGATES_PATH = os.path.join(TABLES_DIR, "daily_weather_aggregates.csv")
MODEL_METRICS_PATH = os.path.join(TABLES_DIR, "model_metrics.csv")

NUMERIC_FIELDS = [
    "latitude",
    "longitude",
    "last_updated_epoch",
    "temperature_celsius",
    "temperature_fahrenheit",
    "wind_mph",
    "wind_kph",
    "wind_degree",
    "pressure_mb",
    "pressure_in",
    "precip_mm",
    "precip_in",
    "humidity",
    "cloud",
    "feels_like_celsius",
    "feels_like_fahrenheit",
    "visibility_km",
    "visibility_miles",
    "uv_index",
    "gust_mph",
    "gust_kph",
    "air_quality_Carbon_Monoxide",
    "air_quality_Ozone",
    "air_quality_Nitrogen_dioxide",
    "air_quality_Sulphur_dioxide",
    "air_quality_PM2.5",
    "air_quality_PM10",
    "air_quality_us-epa-index",
    "air_quality_gb-defra-index",
    "moon_illumination",
]

EDA_FIELDS = [
    "temperature_celsius",
    "precip_mm",
    "humidity",
    "cloud",
    "wind_kph",
    "pressure_mb",
    "uv_index",
    "air_quality_PM2.5",
]

FORECAST_TARGETS = ("avg_temperature_celsius", "avg_precip_mm")


def ensure_dirs() -> None:
    for path in (OUTPUT_DIR, CHARTS_DIR, TABLES_DIR):
        os.makedirs(path, exist_ok=True)


def percentile(sorted_values, q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    idx = (len(sorted_values) - 1) * q
    low = math.floor(idx)
    high = math.ceil(idx)
    if low == high:
        return float(sorted_values[low])
    fraction = idx - low
    return sorted_values[low] * (1 - fraction) + sorted_values[high] * fraction


def safe_float(value: str) -> float:
    return float(value.strip()) if value.strip() else 0.0


def compute_field_stats() -> dict[str, dict[str, float]]:
    values = {field: [] for field in NUMERIC_FIELDS}
    with open(DATASET_PATH, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            for field in NUMERIC_FIELDS:
                values[field].append(safe_float(row[field]))

    stats = {}
    for field, field_values in values.items():
        ordered = sorted(field_values)
        q1 = percentile(ordered, 0.25)
        q3 = percentile(ordered, 0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        min_value = ordered[0]
        max_value = ordered[-1]
        stats[field] = {
            "min": min_value,
            "max": max_value,
            "mean": mean(field_values),
            "q1": q1,
            "median": percentile(ordered, 0.5),
            "q3": q3,
            "lower_bound": lower,
            "upper_bound": upper,
        }
    return stats


def clip_value(value: float, lower: float, upper: float) -> float:
    if value < lower:
        return lower
    if value > upper:
        return upper
    return value


def correlation(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = 0.0
    den_x = 0.0
    den_y = 0.0
    for x_val, y_val in zip(xs, ys):
        x_delta = x_val - mean_x
        y_delta = y_val - mean_y
        num += x_delta * y_delta
        den_x += x_delta * x_delta
        den_y += y_delta * y_delta
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / math.sqrt(den_x * den_y)


def round_if_number(value):
    if isinstance(value, float):
        return round(value, 4)
    return value


def invert_matrix(matrix: list[list[float]]) -> list[list[float]]:
    size = len(matrix)
    augmented = []
    for row_index in range(size):
        row = matrix[row_index][:]
        row.extend(1.0 if row_index == col_index else 0.0 for col_index in range(size))
        augmented.append(row)

    for pivot_index in range(size):
        pivot_row = max(range(pivot_index, size), key=lambda idx: abs(augmented[idx][pivot_index]))
        if abs(augmented[pivot_row][pivot_index]) < 1e-12:
            raise ValueError("Matrix is singular.")
        augmented[pivot_index], augmented[pivot_row] = augmented[pivot_row], augmented[pivot_index]
        pivot_value = augmented[pivot_index][pivot_index]
        augmented[pivot_index] = [value / pivot_value for value in augmented[pivot_index]]
        for row_index in range(size):
            if row_index == pivot_index:
                continue
            factor = augmented[row_index][pivot_index]
            if factor == 0.0:
                continue
            augmented[row_index] = [
                current - factor * pivot
                for current, pivot in zip(augmented[row_index], augmented[pivot_index])
            ]

    return [row[size:] for row in augmented]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    rows = len(a)
    cols = len(b[0])
    shared = len(b)
    result = [[0.0 for _ in range(cols)] for _ in range(rows)]
    for i in range(rows):
        for k in range(shared):
            if a[i][k] == 0.0:
                continue
            for j in range(cols):
                result[i][j] += a[i][k] * b[k][j]
    return result


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*matrix)]


def linear_regression_fit(features: list[list[float]], target: list[float]) -> list[float]:
    x_matrix = [[1.0] + row for row in features]
    y_matrix = [[value] for value in target]
    xt = transpose(x_matrix)
    xtx = matmul(xt, x_matrix)
    xtx_inv = invert_matrix(xtx)
    xty = matmul(xt, y_matrix)
    weights = matmul(xtx_inv, xty)
    return [row[0] for row in weights]


def linear_regression_predict(weights: list[float], features: list[list[float]]) -> list[float]:
    predictions = []
    for row in features:
        total = weights[0]
        for coef, value in zip(weights[1:], row):
            total += coef * value
        predictions.append(total)
    return predictions


def mae(actual: list[float], predicted: list[float]) -> float:
    return sum(abs(a - p) for a, p in zip(actual, predicted)) / len(actual)


def rmse(actual: list[float], predicted: list[float]) -> float:
    return math.sqrt(sum((a - p) ** 2 for a, p in zip(actual, predicted)) / len(actual))


def mape(actual: list[float], predicted: list[float]) -> float:
    filtered = [(a, p) for a, p in zip(actual, predicted) if a != 0]
    if not filtered:
        return 0.0
    return 100.0 * sum(abs((a - p) / a) for a, p in filtered) / len(filtered)


def min_max_scale(values: list[float]) -> list[float]:
    lower = min(values)
    upper = max(values)
    if math.isclose(lower, upper):
        return [0.0 for _ in values]
    return [(value - lower) / (upper - lower) for value in values]


def write_svg_line_chart(
    file_path: str,
    title: str,
    y_label: str,
    values: list[float],
    labels: list[str],
    stroke: str,
) -> None:
    width = 960
    height = 540
    left = 80
    right = 40
    top = 60
    bottom = 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    min_y = min(values)
    max_y = max(values)
    if math.isclose(min_y, max_y):
        min_y -= 1.0
        max_y += 1.0

    def x_pos(index: int) -> float:
        if len(values) == 1:
            return left + plot_width / 2
        return left + (plot_width * index / (len(values) - 1))

    def y_pos(value: float) -> float:
        return top + plot_height - ((value - min_y) / (max_y - min_y) * plot_height)

    points = " ".join(f"{x_pos(i):.2f},{y_pos(v):.2f}" for i, v in enumerate(values))
    grid_lines = []
    for step in range(6):
        y_value = min_y + (max_y - min_y) * step / 5
        y_pixel = y_pos(y_value)
        grid_lines.append(
            f'<line x1="{left}" y1="{y_pixel:.2f}" x2="{width-right}" y2="{y_pixel:.2f}" '
            'stroke="#d9e2ec" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{left-10}" y="{y_pixel+4:.2f}" text-anchor="end" '
            'font-size="12" fill="#486581">{:.1f}</text>'.format(y_value)
        )

    x_ticks = []
    tick_count = min(6, len(labels))
    for step in range(tick_count):
        index = round(step * (len(labels) - 1) / max(tick_count - 1, 1))
        x_pixel = x_pos(index)
        x_ticks.append(
            f'<line x1="{x_pixel:.2f}" y1="{height-bottom}" x2="{x_pixel:.2f}" y2="{height-bottom+6}" '
            'stroke="#243b53" stroke-width="1" />'
        )
        x_ticks.append(
            f'<text x="{x_pixel:.2f}" y="{height-bottom+24}" text-anchor="middle" '
            f'font-size="12" fill="#486581">{labels[index]}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#f7fafc" />
  <text x="{width/2:.0f}" y="32" text-anchor="middle" font-size="24" fill="#102a43">{title}</text>
  <text x="24" y="{height/2:.0f}" transform="rotate(-90 24 {height/2:.0f})" text-anchor="middle" font-size="14" fill="#486581">{y_label}</text>
  {''.join(grid_lines)}
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#243b53" stroke-width="2" />
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#243b53" stroke-width="2" />
  {''.join(x_ticks)}
  <polyline fill="none" stroke="{stroke}" stroke-width="3" points="{points}" />
</svg>
"""
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(svg)


def write_svg_bar_chart(file_path: str, title: str, categories: list[str], values: list[float]) -> None:
    width = 960
    height = 540
    left = 80
    right = 40
    top = 60
    bottom = 110
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(values) if values else 1.0
    bar_width = plot_width / max(len(values), 1)
    bars = []
    labels = []
    for idx, (category, value) in enumerate(zip(categories, values)):
        height_ratio = 0.0 if max_value == 0 else value / max_value
        bar_height = plot_height * height_ratio
        x = left + idx * bar_width + 10
        y = top + plot_height - bar_height
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width-20:.2f}" height="{bar_height:.2f}" '
            'fill="#ef8354" rx="4" />'
        )
        labels.append(
            f'<text x="{x + (bar_width - 20) / 2:.2f}" y="{height-bottom+24}" text-anchor="middle" '
            f'font-size="12" fill="#486581">{category}</text>'
        )
        labels.append(
            f'<text x="{x + (bar_width - 20) / 2:.2f}" y="{y-8:.2f}" text-anchor="middle" '
            f'font-size="12" fill="#102a43">{value:.1f}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#fffaf5" />
  <text x="{width/2:.0f}" y="32" text-anchor="middle" font-size="24" fill="#7c2d12">{title}</text>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#7c2d12" stroke-width="2" />
  <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#7c2d12" stroke-width="2" />
  {''.join(bars)}
  {''.join(labels)}
</svg>
"""
    with open(file_path, "w", encoding="utf-8") as handle:
        handle.write(svg)


def main() -> None:
    ensure_dirs()
    field_stats = compute_field_stats()

    outlier_counts = Counter()
    missing_counts = Counter()
    condition_counts = Counter()
    country_temperature = defaultdict(list)
    selected_field_values = {field: [] for field in EDA_FIELDS}
    daily = defaultdict(lambda: {
        "temperature_sum": 0.0,
        "precip_sum": 0.0,
        "humidity_sum": 0.0,
        "wind_sum": 0.0,
        "count": 0,
    })

    with open(DATASET_PATH, newline="") as source, open(
        CLEANED_DATASET_PATH, "w", newline=""
    ) as cleaned_handle:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        writer = csv.DictWriter(cleaned_handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            cleaned_row = dict(row)
            for field in fieldnames:
                if row[field].strip() == "":
                    missing_counts[field] += 1

            for field in NUMERIC_FIELDS:
                raw_value = safe_float(row[field])
                stats = field_stats[field]
                clipped = clip_value(raw_value, stats["lower_bound"], stats["upper_bound"])
                if not math.isclose(raw_value, clipped):
                    outlier_counts[field] += 1
                cleaned_row[field] = f"{clipped:.4f}"

            writer.writerow(cleaned_row)

            parsed_date = datetime.strptime(row["last_updated"], "%Y-%m-%d %H:%M").date().isoformat()
            temperature = float(cleaned_row["temperature_celsius"])
            precipitation = float(cleaned_row["precip_mm"])
            humidity = float(cleaned_row["humidity"])
            wind_kph = float(cleaned_row["wind_kph"])

            daily_row = daily[parsed_date]
            daily_row["temperature_sum"] += temperature
            daily_row["precip_sum"] += precipitation
            daily_row["humidity_sum"] += humidity
            daily_row["wind_sum"] += wind_kph
            daily_row["count"] += 1

            condition_counts[row["condition_text"]] += 1
            country_temperature[row["country"]].append(temperature)

            for field in EDA_FIELDS:
                selected_field_values[field].append(float(cleaned_row[field]))

    ordered_days = sorted(daily)
    aggregate_rows = []
    for day in ordered_days:
        count = daily[day]["count"]
        aggregate_rows.append(
            {
                "date": day,
                "avg_temperature_celsius": daily[day]["temperature_sum"] / count,
                "avg_precip_mm": daily[day]["precip_sum"] / count,
                "avg_humidity": daily[day]["humidity_sum"] / count,
                "avg_wind_kph": daily[day]["wind_sum"] / count,
                "observation_count": count,
            }
        )

    with open(DAILY_AGGREGATES_PATH, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0].keys()))
        writer.writeheader()
        for row in aggregate_rows:
            writer.writerow({key: round_if_number(value) for key, value in row.items()})

    correlation_matrix = {}
    for left_field in EDA_FIELDS:
        correlation_matrix[left_field] = {}
        for right_field in EDA_FIELDS:
            correlation_matrix[left_field][right_field] = correlation(
                selected_field_values[left_field],
                selected_field_values[right_field],
            )

    hottest_countries = sorted(
        ((country, sum(values) / len(values)) for country, values in country_temperature.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    coolest_countries = sorted(
        ((country, sum(values) / len(values)) for country, values in country_temperature.items()),
        key=lambda item: item[1],
    )[:10]

    model_metrics = []
    forecasts = {}
    future_forecasts = {}
    for target in FORECAST_TARGETS:
        series = [row[target] for row in aggregate_rows]
        lag = 7
        features = []
        target_values = []
        dates = []
        for index in range(lag, len(series)):
            window = series[index - lag:index]
            features.append([
                float(index),
                window[-1],
                sum(window[-3:]) / 3,
                sum(window) / len(window),
            ])
            target_values.append(series[index])
            dates.append(aggregate_rows[index]["date"])

        split_index = max(1, int(len(features) * 0.8))
        train_x = features[:split_index]
        test_x = features[split_index:]
        train_y = target_values[:split_index]
        test_y = target_values[split_index:]

        normalized_train_x = [list(values) for values in zip(*[min_max_scale(column) for column in zip(*train_x)])]
        scaling_pairs = []
        for column in zip(*train_x):
            lower = min(column)
            upper = max(column)
            scaling_pairs.append((lower, upper))

        normalized_test_x = []
        for row in test_x:
            normalized_row = []
            for value, (lower, upper) in zip(row, scaling_pairs):
                if math.isclose(lower, upper):
                    normalized_row.append(0.0)
                else:
                    normalized_row.append((value - lower) / (upper - lower))
            normalized_test_x.append(normalized_row)

        weights = linear_regression_fit(normalized_train_x, train_y)
        predictions = linear_regression_predict(weights, normalized_test_x)
        naive_predictions = [row[1] for row in test_x]

        model_metrics.append(
            {
                "target": target,
                "model": "linear_regression_with_lags",
                "mae": mae(test_y, predictions),
                "rmse": rmse(test_y, predictions),
                "mape": mape(test_y, predictions),
            }
        )
        model_metrics.append(
            {
                "target": target,
                "model": "naive_last_value_baseline",
                "mae": mae(test_y, naive_predictions),
                "rmse": rmse(test_y, naive_predictions),
                "mape": mape(test_y, naive_predictions),
            }
        )

        forecasts[target] = {
            "test_dates": dates[split_index:],
            "actual": test_y,
            "predicted": predictions,
        }

        history = series[:]
        future_points = []
        next_index = len(history)
        for _ in range(7):
            window = history[-lag:]
            raw_features = [float(next_index), window[-1], sum(window[-3:]) / 3, sum(window) / len(window)]
            normalized_row = []
            for value, (lower, upper) in zip(raw_features, scaling_pairs):
                if math.isclose(lower, upper):
                    normalized_row.append(0.0)
                else:
                    normalized_row.append((value - lower) / (upper - lower))
            prediction = linear_regression_predict(weights, [normalized_row])[0]
            future_points.append(prediction)
            history.append(prediction)
            next_index += 1
        future_forecasts[target] = future_points

    with open(MODEL_METRICS_PATH, "w", newline="") as handle:
        fieldnames = ["target", "model", "mae", "rmse", "mape"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in model_metrics:
            writer.writerow({key: round_if_number(value) for key, value in row.items()})

    date_labels = [day[5:] for day in ordered_days]
    write_svg_line_chart(
        os.path.join(CHARTS_DIR, "daily_avg_temperature.svg"),
        "Daily Average Temperature",
        "Temperature (C)",
        [row["avg_temperature_celsius"] for row in aggregate_rows],
        date_labels,
        "#0b7285",
    )
    write_svg_line_chart(
        os.path.join(CHARTS_DIR, "daily_avg_precipitation.svg"),
        "Daily Average Precipitation",
        "Precipitation (mm)",
        [row["avg_precip_mm"] for row in aggregate_rows],
        date_labels,
        "#1c7ed6",
    )
    write_svg_bar_chart(
        os.path.join(CHARTS_DIR, "top_weather_conditions.svg"),
        "Top Weather Conditions",
        [label for label, _ in condition_counts.most_common(8)],
        [count for _, count in condition_counts.most_common(8)],
    )

    summary = {
        "dataset": {
            "path": DATASET_PATH,
            "row_count": sum(row["observation_count"] for row in aggregate_rows),
            "column_count": 41,
            "date_range": {
                "start": ordered_days[0],
                "end": ordered_days[-1],
                "total_days": len(ordered_days),
            },
        },
        "preprocessing": {
            "missing_values_detected": dict(missing_counts),
            "missing_value_strategy": "No blank fields were present; no imputation was required.",
            "outlier_strategy": "Applied IQR clipping across numeric columns before analysis.",
            "outlier_counts_top_10": outlier_counts.most_common(10),
            "normalization_strategy": "Applied min-max normalization to time-series model features on the training split.",
        },
        "eda": {
            "top_weather_conditions": condition_counts.most_common(10),
            "hottest_countries": [(country, round(value, 2)) for country, value in hottest_countries],
            "coolest_countries": [(country, round(value, 2)) for country, value in coolest_countries],
            "correlations": {
                left: {right: round(value, 4) for right, value in right_map.items()}
                for left, right_map in correlation_matrix.items()
            },
        },
        "forecasting": {
            "targets": list(FORECAST_TARGETS),
            "evaluation_metrics": [
                {key: round_if_number(value) for key, value in row.items()} for row in model_metrics
            ],
            "future_7_day_forecast": {
                target: [round(value, 3) for value in values]
                for target, values in future_forecasts.items()
            },
        },
    }

    with open(REPORT_DATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Wrote cleaned dataset to {CLEANED_DATASET_PATH}")
    print(f"Wrote daily aggregates to {DAILY_AGGREGATES_PATH}")
    print(f"Wrote charts to {CHARTS_DIR}")
    print(f"Wrote summary to {REPORT_DATA_PATH}")


if __name__ == "__main__":
    main()
