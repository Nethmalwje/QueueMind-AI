
"""
Simulation engine for QueueMind.
"""

from datetime import datetime, timedelta
import random
from queue_config import QUEUES, SIMULATION_PARAMS
from state_manager import get_queue_lengths, get_queue_activity


def run_simulation(queue_name):
    """Run a simulation for a specific queue"""
    # Check if queue exists in config
    if queue_name not in QUEUES:
        # Use default values if queue not found
        avg_service_time = 2.0
    else:
        config = QUEUES[queue_name]
        avg_service_time = config["avg_service_time"]

    arrival_rate = SIMULATION_PARAMS["arrival_rate"]
    duration = SIMULATION_PARAMS["simulation_duration"]

    current_time = datetime.now()
    end_time = current_time + timedelta(minutes=duration)

    customers = []
    wait_times = []

    while current_time < end_time:
        if random.random() < arrival_rate:
            customers.append(current_time)
            service_time = timedelta(minutes=avg_service_time)
            wait_times.append(service_time.total_seconds() / 60)

        current_time += timedelta(minutes=1)

    return customers, wait_times


def generate_sample_data(days=14, forecast_days=5):
    """Generate sample historical and forecast queue data"""
    from datetime import datetime, timedelta
    import numpy as np
    import pandas as pd

    dates = [(datetime.now() - timedelta(days=days) +
             timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days+1)]
    base = np.linspace(30, 50, days+1)
    weekly = 15 * np.sin(np.linspace(0, 6*np.pi, days+1))
    random = np.random.normal(0, 5, days+1)
    queue_counts = np.maximum(base + weekly + random, 5)
    return pd.DataFrame({
        'date': dates,
        'queue_count': queue_counts.astype(int),
        'is_forecast': [i >= (days - forecast_days) for i in range(days+1)]
    })
