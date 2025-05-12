
"""
Generate AI-driven recommendations and render queue predictions for QueueMind.
"""
from queue_config import QUEUES
from predictor import generate_prophet_forecast, get_all_queue_forecasts, predict_queue_trends
from state_manager import load_total_queue_history
import random
import re
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
# import openai
# import os


def preprocess_markdown(text):
    """
    Convert Markdown bold syntax (**text**) to HTML <strong> tags, preserving emojis and other text.

    Args:
        text (str): Input string with potential Markdown syntax.

    Returns:
        str: Processed string with HTML tags.
    """
    def replace_bold(match):
        return f"<strong>{match.group(1)}</strong>"

    text = re.sub(r'\*\*([^\*]+)\*\*', replace_bold, text)
    return text


def generate_ai_recommendations(current_queues, predicted_wait_time_30min, efficiency, queue_forecasts=None):
    """
    Generate intelligent recommendations based on current queue status, predictions, and efficiency.

    Args:
        current_queues (dict): Current queue lengths {queue_name: count}.
        predicted_wait_time_30min (float): Predicted wait time in 30 minutes (minutes).
        efficiency (float): Current operational efficiency (%).
        queue_forecasts (dict, optional): Dictionary of queue forecasts {queue_name: DataFrame}.

    Returns:
        list: List of recommendation strings with HTML tags for bold text.
    """
    recommendations = []
    bottleneck_detected = False

    # Organize queues by load level with defensive programming
    high_load_queues = []
    medium_load_queues = []
    low_load_queues = []

    for queue_name, count in current_queues.items():
        capacity = QUEUES.get(queue_name, {}).get("capacity", 1)
        if capacity <= 0:
            capacity = 1

        load_ratio = count / capacity

        if load_ratio > 0.8:
            high_load_queues.append((queue_name, count, capacity))
            bottleneck_detected = True
        elif load_ratio > 0.5:
            medium_load_queues.append((queue_name, count, capacity))
        else:
            low_load_queues.append((queue_name, count, capacity))

    # Calculate queue imbalance metrics safely
    queue_lengths = list(current_queues.values())
    if len(queue_lengths) >= 2:
        imbalance = max(queue_lengths) - min(queue_lengths)
        imbalance_ratio = max(queue_lengths) / (min(queue_lengths) + 1)
    else:
        imbalance = 0
        imbalance_ratio = 1

    # 1. Handle bottlenecks and high load queues
    if high_load_queues:
        worst_queue = max(high_load_queues, key=lambda x: x[1]/x[2])
        queue_name, count, capacity = worst_queue

        current_time = st.session_state.get('current_time', datetime.now())
        appointment_time = current_time + timedelta(hours=2)
        appointment_str = appointment_time.strftime("%I:%M %p")

        recommendations.append(
            f"🚨 **Critical bottleneck detected in {queue_name}** ({count}/{capacity}). "
            f"Consider opening another service point immediately and offering appointments for {appointment_str} onwards."
        )

        if len(high_load_queues) > 1:
            other_queues = [q[0]
                            for q in high_load_queues if q[0] != queue_name]
            recommendations.append(
                f"⚠️ Multiple queues approaching capacity: {', '.join(other_queues)}. "
                f"Implement load balancing measures and consider temporary staff reallocation."
            )

    # 2. Wait time prediction
    if predicted_wait_time_30min > 20:
        recommendations.append(
            f"⏱️ **Alert:** Predicted wait time in 30 minutes will be {predicted_wait_time_30min} minutes. "
            f"Consider sending SMS notifications to customers with appointments after {(st.session_state.get('current_time', datetime.now()) + timedelta(minutes=60)).strftime('%I:%M %p')}."
        )
    elif predicted_wait_time_30min > 10:
        recommendations.append(
            f"⏱️ **Notice:** Wait time expected to increase to {predicted_wait_time_30min} minutes in 30 minutes. "
            f"Prepare additional staff for peak period."
        )

    # 3. Queue imbalance detection
    if imbalance > 5 and imbalance_ratio > 1.5 and (high_load_queues or medium_load_queues):
        source_queue = None
        if high_load_queues:
            source_queue = high_load_queues[0][0]
        elif medium_load_queues:
            source_queue = medium_load_queues[-1][0]

        target_queue = None
        if low_load_queues:
            target_queue = low_load_queues[0][0]
        elif medium_load_queues and len(medium_load_queues) > 1:
            target_queue = min(medium_load_queues, key=lambda x: x[1]/x[2])[0]

        if source_queue and target_queue:
            recommendations.append(
                f"⚖️ **Queue imbalance detected** (difference of {imbalance} customers). "
                f"Consider redirecting customers from {source_queue} to {target_queue}."
            )
        elif source_queue:
            recommendations.append(
                f"⚖️ **Queue imbalance detected** (difference of {imbalance} customers). "
                f"All queues are busy, but {source_queue} is particularly overloaded."
            )

    # 4. Efficiency-based recommendations
    if efficiency < 75:
        available_queues = list(current_queues.keys())
        if available_queues:
            random_queue = random.choice(available_queues)
            recommendations.append(
                f"🔄 **System efficiency is low** ({efficiency}%). "
                f"Check for processing bottlenecks or consider staff training for {random_queue}."
            )

    # 5. Capacity planning recommendation
    if not bottleneck_detected and efficiency > 85 and low_load_queues:
        underused_queues = [q[0] for q in low_load_queues if q[1]/q[2] < 0.3]
        if underused_queues:
            recommendations.append(
                f"💡 **Resource optimization opportunity:** {', '.join(underused_queues)} are underutilized. "
                f"Consider temporary reassignment of 1-2 staff members."
            )

    # 6. Forecast-based insights
    if queue_forecasts:
        for queue_name, forecast_df in queue_forecasts.items():
            if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty or not all(col in forecast_df.columns for col in ['is_forecast', 'queue_count', 'date']):
                continue

            future_df = forecast_df[forecast_df['is_forecast']].copy()
            if future_df.empty:
                continue

            # Get queue config
            capacity = QUEUES.get(queue_name, {}).get("capacity", 1)
            avg_service_time = QUEUES.get(
                queue_name, {}).get("avg_service_time", 1)

            # Calculate wait times and high load duration
            future_df['wait_time'] = future_df['queue_count'] * \
                avg_service_time
            high_load_threshold = 0.8 * capacity
            high_load_times = future_df[future_df['queue_count']
                                        > high_load_threshold]

            # Calculate trend (slope of queue_count over time)
            if len(future_df) >= 2:
                time_diffs = (
                    future_df['date'] - future_df['date'].iloc[0]).dt.total_seconds() / 60.0
                slope, _ = np.polyfit(time_diffs, future_df['queue_count'], 1)
                trend = "increasing" if slope > 0.1 else "decreasing" if slope < -0.1 else "stable"
            else:
                trend = "stable"

            # Generate recommendation if peak wait time or sustained high load is significant
            max_wait = future_df['wait_time'].max()
            max_queue = future_df['queue_count'].max()
            peak_time = future_df.loc[future_df['wait_time'].idxmax(), 'date']
            high_load_duration = len(
                high_load_times) if not high_load_times.empty else 0

            if max_wait > 20 or (max_queue > 0.9 * capacity and high_load_duration > 10):
                urgency = "Critical" if (
                    max_wait > 30 or trend == "increasing") else "High"
                action = "Open an additional service point" if trend == "increasing" else "Monitor closely and prepare staff"
                duration_str = f" for {high_load_duration} minutes" if high_load_duration > 0 else ""

                recommendations.append(
                    f"📊 **{urgency} Predictive Alert for {queue_name}:** "
                    f"Peak wait time of {max_wait:.1f} minutes predicted at {peak_time.strftime('%I:%M %p')} "
                    f"({int(max_queue)} customers, {int(max_queue/capacity*100)}% of capacity, {trend} trend)."
                    f"{duration_str}. {action} to manage customer flow."
                )
            elif max_wait > 10:
                recommendations.append(
                    f"📊 **Moderate Predictive Alert for {queue_name}:** "
                    f"Wait time of {max_wait:.1f} minutes predicted at {peak_time.strftime('%I:%M %p')} "
                    f"({int(max_queue)} customers, {trend} trend). Ensure staff are prepared."
                )

    # 7. Default positive status if no issues
    if not recommendations:
        recommendations.append(
            "✅ All queues are operating optimally. No immediate actions required.")

    # Preprocess each recommendation to convert Markdown to HTML
    return [preprocess_markdown(rec) for rec in recommendations]


def render_queue_predictions(queue_forecasts, current_queues):
    """
    Renders detailed queue prediction messages with bottleneck detection.

    Args:
        queue_forecasts (dict): Dictionary of queue forecasts by queue name
        current_queues (dict): Current queue lengths by queue name

    Returns:
        dict: Dictionary of queue names and prediction messages
    """
    predictions = {}

    for queue_name, forecast_df in queue_forecasts.items():
        if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
            predictions[queue_name] = "No forecast data available."
            continue

        future_df = forecast_df[forecast_df['is_forecast']]
        if future_df.empty:
            predictions[queue_name] = "No future forecast available."
            continue

        current_count = current_queues.get(queue_name, 0)
        capacity = QUEUES.get(queue_name, {}).get("capacity", 1)
        current_ratio = current_count / capacity if capacity > 0 else 0

        max_future = future_df['queue_count'].max()
        min_future = future_df['queue_count'].min()
        max_time = future_df.loc[future_df['queue_count'].idxmax(), 'date']
        trend_direction = "increasing" if max_future > current_count else "decreasing"

        bottleneck_threshold = 0.8 * capacity
        bottleneck_times = future_df[future_df['queue_count']
                                     > bottleneck_threshold]

        if not bottleneck_times.empty:
            first_bottleneck = bottleneck_times.iloc[0]['date']
            time_to_bottleneck = (first_bottleneck - st.session_state.get(
                'current_time', datetime.now())).total_seconds() / 60
            bottleneck_str = f"⚠️ <span style='color:#ff6b6b'>Bottleneck predicted in {int(time_to_bottleneck)} minutes</span> ({first_bottleneck.strftime('%I:%M %p')})"

            if len(bottleneck_times) > 1:
                bottleneck_duration = (
                    bottleneck_times.iloc[-1]['date'] - bottleneck_times.iloc[0]['date']).total_seconds() / 60
                bottleneck_str += f", lasting approximately {int(bottleneck_duration)} minutes"
        else:
            bottleneck_str = "✅ No bottlenecks predicted in the next 30 minutes"

        prediction = f"{bottleneck_str}<br>"
        prediction += f"Current load: {int(current_ratio*100)}% of capacity<br>"
        prediction += f"Trend: {trend_direction.capitalize()}, peak of {int(max_future)} customers at {max_time.strftime('%I:%M %p')}<br>"

        avg_service_time = QUEUES.get(
            queue_name, {}).get("avg_service_time", 1)
        current_wait = current_count * avg_service_time
        peak_wait = max_future * avg_service_time
        prediction += f"Wait times: Current {current_wait:.1f} min → Peak {peak_wait:.1f} min"

        predictions[queue_name] = preprocess_markdown(prediction)

    return predictions


# def send_to_openai_for_summary(recommendations):
#     """
#     Sends a list of recommendations to OpenAI's GPT-4 model and gets a natural language summary.

#     Args:
#         recommendations (list): List of recommendation strings (HTML-tagged or plain).

#     Returns:
#         str: Summary response from GPT-4.
#     """
#     api_key = os.getenv("OPENAI_API_KEY")
#     if not api_key:
#         raise ValueError("OPENAI_API_KEY not found in environment variables.")

#     openai.api_key = api_key

#     plain_recs = [re.sub(r'<[^>]+>', '', rec)
#                   for rec in recommendations]  # Strip HTML
#     prompt = (
#         "You are an operations analyst. Summarize and prioritize these queue management recommendations "
#         "into a concise action plan for the next 30 minutes:\n\n" +
#         "\n".join(f"- {rec}" for rec in plain_recs)
#     )

#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[{"role": "user", "content": prompt}],
#         temperature=0.5,
#         max_tokens=300
#     )

#     return response['choices'][0]['message']['content'].strip()
