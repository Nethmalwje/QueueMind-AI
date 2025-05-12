
"""
Generate 24 hours of queue history data for QueueMind demo.
"""
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Configuration (match queue_config.py)
QUEUES = {
    "Standard Queue 1": {"avg_service_time": 7, "capacity": 20, "active": True},
    "Standard Queue 2": {"avg_service_time": 10, "capacity": 15, "active": True},
    "Express Queue 3": {"avg_service_time": 3, "capacity": 10, "active": True}
}
DATA_DIR = "data"
HISTORY_FILE = os.path.join(DATA_DIR, "queue_history.json")


def ensure_data_dir():
    """Ensure data directory exists."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


def simulate_queue_step(queues, hour):
    """
    Simulate one minute of queue activity with hourly patterns.

    Args:
        queues (dict): Current queue states {queue_name: {"customers": [...]}}.
        hour (int): Hour of the day (0-23).

    Returns:
        dict: Updated queue states.
    """
    # Define arrival rates by time of day
    if 8 <= hour < 10:  # Morning rush
        arrival_prob = 0.9
        max_arrivals = 3
    elif 12 <= hour < 14:  # Lunch peak
        arrival_prob = 0.95
        max_arrivals = 4
    elif 15 <= hour < 17:  # Afternoon
        arrival_prob = 0.7
        max_arrivals = 2
    elif 18 <= hour < 20:  # Evening
        arrival_prob = 0.5
        max_arrivals = 1
    else:  # Off-peak
        arrival_prob = 0.3
        max_arrivals = 1

    departure_prob = 0.4  # Consistent service rate

    for queue_name in QUEUES:
        if not QUEUES[queue_name]["active"]:
            continue
        customers = queues[queue_name]["customers"]
        capacity = QUEUES[queue_name]["capacity"]

        # Arrivals
        if len(customers) < capacity and random.random() < arrival_prob:
            arrivals = random.randint(
                1, min(max_arrivals, capacity - len(customers)))
            for _ in range(arrivals):
                customers.append({"arrived_at": datetime.now().isoformat()})

        # Departures
        if customers and random.random() < departure_prob:
            customers.pop(0)

    return queues


def generate_history():
    """Generate 24 hours of queue history data."""
    start_time = pd.Timestamp("2025-05-08 00:00:00")
    end_time = start_time + timedelta(hours=24)
    interval = timedelta(minutes=1)  # Minutely data

    # Initialize queues
    queues = {name: {"customers": [], "active": QUEUES[name]["active"]}
              for name in QUEUES}
    queues.update({
        "Queue A": {"customers": [], "active": False},
        "Queue B": {"customers": [], "active": False}
    })

    history = []
    current_time = start_time

    while current_time < end_time:
        hour = current_time.hour
        queues = simulate_queue_step(queues, hour)

        # Save snapshot
        history_entry = {
            "timestamp": current_time.isoformat(),
            "queues": {name: len(data["customers"]) for name, data in queues.items()}
        }
        history.append(history_entry)

        current_time += interval

    # Save to queue_history.json
    ensure_data_dir()
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)

    print(f"Generated {len(history)} entries in {HISTORY_FILE}")


if __name__ == "__main__":
    generate_history()
