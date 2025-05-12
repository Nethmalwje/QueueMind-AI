import random
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
from streamlit_autorefresh import st_autorefresh
import altair as alt


# Import our modules
from queue_config import QUEUES
from state_manager import get_customers_served, load_total_queue_history, simulate_step, get_queue_lengths, get_queue_activity, set_queue_active
from predictor import generate_dummy_forecast, generate_prophet_forecast, predict_queue_length, predict_queue_trends
from queue_simulator import generate_sample_data
from wait_time_predictor import calculate_wait_times
from queue_messages import generate_ai_recommendations, render_queue_predictions, send_to_openai_for_summary
from predictor import get_all_queue_forecasts, predict_wait_time, predict_queue_trends


# ------------------------ STREAMLIT PAGE CONFIG ------------------------
st.set_page_config(
    layout="wide", page_title="QueueMind AI Dashboard", page_icon="🧠")

# ------------------------ CSS ------------------------
st.markdown("""
<style>
    .appview-container .:main .block-container {
        max-width: 1200px !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        margin: auto !important;
    }
    .stApp { background-color: #0c1116; color: white; }
    .header { font-size: 36px !important; font-weight: bold !important; margin-bottom: 20px !important; }
    .metric-card, .queue-card, .chart-card {
        background-color: #1a1f27;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .big-metric { font-size: 42px; font-weight: bold; margin-top: 5px; }
    .orange-text { color: #ff9c41; }
    .green-text { color: #5cb85c; }
    .wait-unit { font-size: 24px; color: #aaa; }
    .status-pill {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
        margin-top: 5px;
        margin-bottom: 8px;
    }
    .high-pill { background-color: #5d3a11; color: #ff9c41; }
    .medium-pill { background-color: #3a3e11; color: #ded03b; }
    .low-pill { background-color: #21381c; color: #5cb85c; }
    .inactive-pill { background-color: #3a3a3a; color: #ccc; }
    .stButton>button {
        background-color: #5cb85c;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 5px;
        margin-top: 10px;
        width: 100%;
    }
    .stButton>button.inactive {
        background-color: #3a3a3a;
        color: #ccc;
    }
    .stButton>button:hover {
        opacity: 0.9;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ------------------------ Initialize Session State ------------------------
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.last_update = time.time()
    st.session_state.efficiency = 85
    st.session_state.total_served = 0
    st.session_state.current_time = datetime.now()
    st.session_state.efficiency_data = np.clip(np.linspace(
        80, 90, 60) + np.random.normal(0, 3, 60), 60, 100)
    st.session_state.times = [(datetime.now(
    ) - timedelta(minutes=60) + timedelta(minutes=i)).strftime("%H:%M") for i in range(60)]
    st.session_state.customers_served = (np.linspace(
        5, 15, 60) + np.random.poisson(2, 60)).astype(int)

# ------------------------ Auto-Refresh Setup ------------------------
st_autorefresh(interval=5000, key="queue_simulator_refresh")

# Update simulation if 12 seconds have passed
current_time = time.time()
if current_time - st.session_state.last_update >= 12:
    simulate_step()
    st.session_state.last_update = current_time
    st.session_state.current_time += timedelta(minutes=1)
    st.session_state.efficiency += random.uniform(-2, 2)
    st.session_state.efficiency = max(60, min(95, st.session_state.efficiency))
    st.session_state.total_served += random.randint(0, 2)

# ------------------------ Get Current Data ------------------------
current_queues = get_queue_lengths()
queue_activity = get_queue_activity()

# Calculate metrics
total_customers = sum(current_queues.values())
predicted_total = predict_queue_length(current_queues)

# Calculate wait times using new module
avg_wait_time, predicted_wait_time, predicted_wait_time_30min = calculate_wait_times(
    current_queues, QUEUES)

# ------------------------ HEADER ------------------------
st.markdown('<div class="header">QueueMind AI</div>', unsafe_allow_html=True)

# ------------------------ METRICS ------------------------
estimated_time = st.session_state.current_time.strftime("%I:%M %p")
df_hist_raw = load_total_queue_history()
total_served = get_customers_served(df_hist_raw)
col1, col2, col3, col4 = st.columns(4)
with col1:
    # print("from dash", avg_wait_time)
    st.markdown(
        f'<div class="metric-card"><div>Current Wait Time</div><div class="big-metric orange-text">{round(avg_wait_time)} <span class="wait-unit">min</span></div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(
        f'<div class="metric-card"><div>Total Customers Served</div><div class="big-metric green-text">{total_served} <span class="wait-unit"></span></div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(
        f'<div class="metric-card"><div>Current Time</div><div class="big-metric green-text">{estimated_time}</div></div>', unsafe_allow_html=True)
with col4:
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=st.session_state.efficiency,
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={'axis': {'range': [0, 100]}, 'bar': {
            'color': "#5cb85c"}, 'bgcolor': "#1a1f27"},
        number={'suffix': "%", 'font': {'color': "#5cb85c", 'size': 30}}
    ))
    fig_gauge.update_layout(
        height=120, paper_bgcolor="#1a1f27", margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True,
                    config={'displayModeBar': False})

# ------------------------ QUEUE STATUS ------------------------


def create_capacity_gauge(current, capacity, color="#5cb85c"):
    percentage = min(100, int((current / capacity) * 100))
    return f"""
    <div style="margin-top: 8px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="flex-grow: 1; background-color: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; overflow: hidden;">
                <div style="width: {percentage}%; background-color: {color}; height: 100%;"></div>
            </div>
            <div style="font-size: 12px; color: #aaa;">{current}/{capacity}</div>
        </div>
    </div>
    """


queue_config = [(name, QUEUES[name]["capacity"]) for name in QUEUES]
queue_cols = st.columns(len(queue_config))

for col, (label, capacity) in zip(queue_cols, queue_config):
    with col:
        count = current_queues.get(label, 0)
        is_active = queue_activity.get(label, True)

        if not is_active:
            status, pill, color, wait = "INACTIVE", "inactive-pill", "#ccc", "— min"
        else:
            wait = f"{round(count * QUEUES[label]['avg_service_time'])} min"
            if count >= 0.75 * capacity:
                status, pill, color = "HIGH", "high-pill", "#ff9c41"
            elif count >= 0.5 * capacity:
                status, pill, color = "MEDIUM", "medium-pill", "#ded03b"
            else:
                status, pill, color = "LOW", "low-pill", "#5cb85c"

        gauge = create_capacity_gauge(count, capacity, color)
        button_text = "Deactivate" if is_active else "Activate"

        col.markdown(f'''
        <div class="queue-card">
            <div>{label}</div>
            <div class="status-pill {pill}">{status}</div>
            <div style="display: flex; align-items: center;">
                <span style="font-size: 24px; font-weight: bold;">{count}</span>
                <span style="margin-left: 5px;">Customers</span>
            </div>
            <div>Wait: {wait}</div>
            {gauge}
    </div>
        ''', unsafe_allow_html=True)

        button_key = f"button_{label}"
        if st.button(button_text, key=button_key, args=(is_active,), help=f"Toggle {label} status"):
            set_queue_active(label, not is_active)
            st.rerun()

# # ------------------------ PREDICTIONS ------------------------

# df_hist = load_total_queue_history()
# df_hist_prophet = df_hist.rename(
#     columns={"ds": "date", "y": "queue_count"})  # for plotting

# if not df_hist.empty:
#     forecast_df = generate_prophet_forecast(df_hist)
#     forecast_df_plot = forecast_df.rename(
#         columns={"date": "date", "queue_count": "queue_count"})
#     df_combined = pd.concat([df_hist_prophet, forecast_df_plot])
# else:
#     df_combined = df_hist_prophet


# def plot_queue_trend(data, height=300, bottleneck_threshold=15):
#     data['date'] = pd.to_datetime(data['date'])
#     historical = data[~data['is_forecast']].copy()
#     forecast = data[data['is_forecast']].copy()

#     if not forecast.empty:
#         transition = forecast.iloc[0]
#         transition_date, transition_val = transition['date'], transition['queue_count']
#         historical = pd.concat(
#             [historical, pd.DataFrame([transition])], ignore_index=True)
#         forecast = pd.concat(
#             [pd.DataFrame([transition]), forecast], ignore_index=True)
#     else:
#         transition = historical.iloc[-1]
#         transition_date, transition_val = transition['date'], transition['queue_count']

#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=historical['date'], y=historical['queue_count'], mode='lines', name='Historical',
#                              line=dict(color='rgba(186, 134, 255, 0.9)',
#                                        shape='spline', width=2),
#                              fill='tozeroy', fillcolor='rgba(186, 134, 255, 0.25)', hoverinfo='skip'))
#     fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['queue_count'], mode='lines', name='Forecast',
#                              line=dict(color='rgba(100, 181, 246, 1)',
#                                        dash='dash', shape='spline', width=2),
#                              fill='tozeroy', fillcolor='rgba(100, 181, 246, 0.25)', hoverinfo='skip'))
#     fig.add_trace(go.Scatter(x=[transition_date], y=[transition_val], mode='markers',
#                              marker=dict(size=8, color='red'), name='Now', hoverinfo='skip'))
#     fig.add_shape(type='line', x0=data['date'].min(), x1=data['date'].max(),
#                   y0=bottleneck_threshold, y1=bottleneck_threshold,
#                   line=dict(color='rgba(255, 0, 0, 0.6)', width=2, dash='dot'))
#     fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
#                       xaxis=dict(title='Time', showgrid=True, tickformat='%H:%M', zeroline=False, color='white',
#                                  gridcolor='rgba(255,255,255,0.05)'),
#                       yaxis=dict(title='Queue Size', showgrid=True, zeroline=False, color='white',
#                                  gridcolor='rgba(255,255,255,0.05)'),
#                       font=dict(color='white'), showlegend=False)
#     return fig


# # trend_cols = st.columns([4, 2])
# # with trend_cols[0]:
# #     st.markdown('<div class="chart-card"><h4 style="color: white;">Queue Trends</h4>',
# #                 unsafe_allow_html=True)
# #     df = generate_sample_data()
# #     st.plotly_chart(plot_queue_trend(df), use_container_width=True,
# #                     config={'displayModeBar': False})
# #     st.markdown('</div>', unsafe_allow_html=True)

# # with trend_cols[1]:
# #     queue_predictions = predict_queue_trends()
# #     render_queue_predictions(queue_predictions)
# now = pd.Timestamp.now()
# past_30min = now - pd.Timedelta(minutes=30)

# df_hist = load_total_queue_history()
# recent_hist = df_hist[df_hist["date"] >= past_30min]

# # If not enough data, show all
# if len(recent_hist) >= 1:
#     df_hist = recent_hist

# # Create forecast
# if not df_hist.empty:
#     forecast_df = generate_dummy_forecast(
#         latest_count=df_hist["queue_count"].iloc[-1],
#         start_time=df_hist["date"].max()
#     )
#     df_combined = pd.concat([df_hist, forecast_df])
# else:
#     df_combined = df_hist

# # Plot
# # st.plotly_chart(plot_queue_trend(df_combined), use_container_width=True,
# #                 config={'displayModeBar': False})
# st.plotly_chart(plot_queue_trend(df_combined),
#                 use_container_width=True, config={'displayModeBar': False})

# ------------------------ PREDICTIONS ------------------------

def generate_prophet_forecast(df_hist):
    from prophet import Prophet
    df_hist = df_hist.copy()
    model = Prophet()
    model.fit(df_hist)

    future = model.make_future_dataframe(
        periods=10, freq='min')  # 10 minutes into the future
    forecast = model.predict(future)

    forecast_df = forecast[['ds', 'yhat']].copy()
    forecast_df.rename(
        columns={"ds": "date", "yhat": "queue_count"}, inplace=True)
    forecast_df['is_forecast'] = True

    return forecast_df[forecast_df['date'] > df_hist['ds'].max()]


df_hist_raw = load_total_queue_history()

# Filter for last 30 minutes
now = pd.Timestamp.now()
past_30min = now - pd.Timedelta(minutes=30)
recent_hist = df_hist_raw[df_hist_raw["date"] >= past_30min]

df_hist = recent_hist if not recent_hist.empty else df_hist_raw

# Rename for Prophet
if not df_hist.empty:
    df_for_prophet = df_hist.rename(columns={"date": "ds", "queue_count": "y"})
    forecast_df = generate_prophet_forecast(df_for_prophet)

    # For plotting
    df_hist_plot = df_for_prophet.rename(
        columns={"ds": "date", "y": "queue_count"})
    df_hist_plot['is_forecast'] = False

    df_combined = pd.concat([df_hist_plot, forecast_df])
else:
    df_combined = pd.DataFrame(columns=["date", "queue_count", "is_forecast"])


def plot_queue_trend(data, height=300, bottleneck_threshold=75):
    data['date'] = pd.to_datetime(data['date'])
    historical = data[~data['is_forecast']].copy()
    forecast = data[data['is_forecast']].copy()

    if not forecast.empty:
        transition = forecast.iloc[0]
        transition_date, transition_val = transition['date'], transition['queue_count']
        historical = pd.concat(
            [historical, pd.DataFrame([transition])], ignore_index=True)
        forecast = pd.concat(
            [pd.DataFrame([transition]), forecast], ignore_index=True)
    else:
        transition = historical.iloc[-1]
        transition_date, transition_val = transition['date'], transition['queue_count']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical['date'], y=historical['queue_count'], mode='lines', name='Historical',
                             line=dict(color='rgba(186, 134, 255, 0.9)',
                                       shape='spline', width=2),
                             fill='tozeroy', fillcolor='rgba(186, 134, 255, 0.25)', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['queue_count'], mode='lines', name='Forecast',
                             line=dict(color='rgba(100, 181, 246, 1)',
                                       dash='dash', shape='spline', width=2),
                             fill='tozeroy', fillcolor='rgba(100, 181, 246, 0.25)', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[transition_date], y=[transition_val], mode='markers',
                             marker=dict(size=8, color='red'), name='Now', hoverinfo='skip'))
    fig.add_shape(type='line', x0=data['date'].min(), x1=data['date'].max(),
                  y0=bottleneck_threshold, y1=bottleneck_threshold,
                  line=dict(color='rgba(255, 0, 0, 0.6)', width=2, dash='dot'))
    fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      xaxis=dict(title='Time', showgrid=True, tickformat='%H:%M', zeroline=False, color='white',
                                 gridcolor='rgba(255,255,255,0.05)'),
                      yaxis=dict(title='Queue Size', showgrid=True, zeroline=False, color='white',
                                 gridcolor='rgba(255,255,255,0.05)'),
                      font=dict(color='white'), showlegend=False)
    return fig


# Plot chart in Streamlit
st.plotly_chart(plot_queue_trend(df_combined),
                use_container_width=True, config={'displayModeBar': False})


# ------------------------ AI RECOMMENDATIONS ------------------------

st.markdown('<div class="header">AI Recommendations</div>',
            unsafe_allow_html=True)

# Get all queue forecasts
forecasts = get_all_queue_forecasts(duration=30)  # 30 minutes into the future

# Generate enhanced recommendations with forecasts
recommendations = generate_ai_recommendations(
    current_queues,
    predicted_wait_time_30min,
    st.session_state.efficiency,
    queue_forecasts=forecasts
)

for rec in recommendations:
    st.markdown(
        f"<div class='metric-card'>{rec}</div>", unsafe_allow_html=True)

    # # Optional GPT-4 summary
    # summary = send_to_openai_for_summary(recommendations)
    # recommendations.append(f"<br><br><strong>AI Summary:</strong><br>{summary}")

# with metrics_cols[1]:
#     if current_time - st.session_state.last_update >= 12:
#         new_served = random.randint(5, 15)
#         st.session_state.customers_served = np.append(
#             st.session_state.customers_served[1:], new_served)

#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=st.session_state.times,
#         y=st.session_state.customers_served,
#         marker_color='rgba(100, 181, 246, 0.8)',
#         name='Customers Served'
#     ))
#     fig.update_layout(
#         title="Customers Served Per Minute",
#         height=250,
#         paper_bgcolor='rgba(0,0,0,0)',
#         plot_bgcolor='rgba(0,0,0,0)',
#         margin=dict(l=20, r=20, t=40, b=20),
#         xaxis=dict(title="Time", showgrid=True,
#                    gridcolor='rgba(255,255,255,0.1)', color='white'),
#         yaxis=dict(title="Customers", showgrid=True,
#                    gridcolor='rgba(255,255,255,0.1)', color='white'),
#         font=dict(color='white')
#     )
#     st.plotly_chart(fig, use_container_width=True,
#                     config={'displayModeBar': False})
