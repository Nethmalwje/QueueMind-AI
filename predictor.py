
"""
AI prediction models for QueueMind.
"""

from datetime import timedelta

import pandas as pd
from prophet import Prophet
from queue_config import PREDICTION_WINDOW, QUEUES
from state_manager import HISTORY_FILE, get_queue_lengths
import random
import numpy as np
import json


def predict_queue_length(current_lengths):
    """
    Predict total queue growth in the prediction window.

    Args:
        current_lengths: Dictionary of queue lengths

    Returns:
        Predicted total queue length
    """
    total_now = sum(current_lengths.values())

    # Simple prediction model (can be replaced with ML model)
    # Add some randomness to simulate AI prediction
    growth_factor = 1 + random.uniform(0.1, 0.3)
    predicted_total = int(total_now * growth_factor)

    return predicted_total


def predict_wait_time(queue_name):
    """
    Predict wait time for a specific queue.

    Args:
        queue_name: Name of the queue

    Returns:
        Predicted wait time in minutes
    """
    # Check if queue_name exists in QUEUES, otherwise use a default
    if queue_name not in QUEUES:
        # Just return a default value if queue doesn't exist
        return 5.0

    queue_config = QUEUES[queue_name]
    avg_service_time = queue_config["avg_service_time"]

    queue_lengths = get_queue_lengths()
    length = queue_lengths.get(queue_name, 0)

    # Add some variability to make predictions more realistic
    variability = random.uniform(0.8, 1.2)
    predicted_time = min(length * avg_service_time *
                         variability, PREDICTION_WINDOW)

    return round(predicted_time, 1)


def predict_queue_trends():
    """Generate queue trend predictions"""
    predictions = {}
    queue_lengths = get_queue_lengths()

    for queue_name in QUEUES:
        current_wait = queue_lengths.get(
            queue_name, 0) * QUEUES[queue_name]["avg_service_time"]

        # Random trend direction
        trend = random.choice(["increasing", "decreasing", "stable"])
        if trend == "increasing":
            future_wait = current_wait * random.uniform(1.2, 1.5)
            time_frame = random.randint(15, 45)
            message = f"Wait time increasing to {round(future_wait)} min in {time_frame} min"
        elif trend == "decreasing":
            future_wait = current_wait * random.uniform(0.5, 0.8)
            time_frame = random.randint(15, 30)
            message = f"Wait time decreasing to {round(future_wait)} min in {time_frame} min"
        else:
            time_frame = random.randint(20, 40)
            message = f"Stable at {round(current_wait)} min for next {time_frame} min"

        predictions[queue_name] = message

    return predictions


def generate_dummy_forecast(latest_count, start_time, duration=30):
    future_times = [start_time + timedelta(minutes=i)
                    for i in range(1, duration + 1)]
    return pd.DataFrame({
        "date": future_times,
        "queue_count": [latest_count] * duration,
        "is_forecast": True
    })


def load_total_queue_history():
    """Load and format historical queue data for Prophet"""
    with open(HISTORY_FILE, 'r') as f:
        history_data = json.load(f)

    rows = []
    for entry in history_data:
        timestamp = pd.to_datetime(entry["timestamp"])
        # Total queue size across all queues
        total = sum(entry["queues"].values())
        rows.append({"ds": timestamp, "y": total})

    df = pd.DataFrame(rows)
    return df


def generate_prophet_forecast(df_hist, duration=30):
    # Fit Prophet
    model = Prophet()
    model.fit(df_hist)

    # Create future dataframe
    future = model.make_future_dataframe(
        periods=duration, freq='min')  # minute-by-minute

    # Predict
    forecast = model.predict(future)

    # Format output
    forecast_df = forecast[["ds", "yhat"]].copy()
    forecast_df.rename(
        columns={"ds": "date", "yhat": "queue_count"}, inplace=True)
    forecast_df["is_forecast"] = forecast_df["date"] > df_hist["ds"].max()
    return forecast_df


def load_queue_histories():
    """
    Load and format historical queue data for each individual queue.
    Returns:
        Dict[str, pd.DataFrame] where each DataFrame has columns ['ds', 'y']
    """
    with open(HISTORY_FILE, 'r') as f:
        history_data = json.load(f)

    queue_data = {queue: [] for queue in QUEUES}

    for entry in history_data:
        timestamp = pd.to_datetime(entry["timestamp"])
        for queue, length in entry["queues"].items():
            if queue in queue_data:
                queue_data[queue].append({"ds": timestamp, "y": length})

    return {q: pd.DataFrame(rows) for q, rows in queue_data.items() if rows}


def generate_prophet_forecast(df_hist, duration=30):
    """
    Given a queue's history DataFrame with columns ['ds', 'y'],
    returns a forecast DataFrame with ['date', 'queue_count', 'is_forecast']
    """
    if len(df_hist) < 10:  # Not enough data to train Prophet
        return pd.DataFrame()

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.3
    )
    model.fit(df_hist)

    future = model.make_future_dataframe(periods=duration, freq='min')
    forecast = model.predict(future)

    forecast_df = forecast[["ds", "yhat"]].copy()
    forecast_df.rename(
        columns={"ds": "date", "yhat": "queue_count"}, inplace=True)
    forecast_df["is_forecast"] = forecast_df["date"] > df_hist["ds"].max()

    return forecast_df


def get_all_queue_forecasts(duration=30):
    """
    Runs Prophet forecast for each queue separately.
    Returns:
        Dict[str, pd.DataFrame]
    """
    queue_histories = load_queue_histories()
    forecasts = {}

    for queue, df_hist in queue_histories.items():
        forecast = generate_prophet_forecast(df_hist, duration=duration)
        if not forecast.empty:
            forecasts[queue] = forecast

    return forecasts
