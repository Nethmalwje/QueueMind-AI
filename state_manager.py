
"""
Manages queue states for the QueueMind application.
"""

import time
import pandas as pd
import streamlit as st
from queue_config import QUEUES
import json
import os
import os
from datetime import datetime
import random

# Directory where queue state data is stored
DATA_DIR = "data"
STATE_FILE = os.path.join(DATA_DIR, "queues.json")

# Ensure these queue names exist in our state
REQUIRED_QUEUES = ["Standard Queue 1", "Standard Queue 2", "Express Queue 3"]

# Initialize state with both configured queues and any required legacy queue names
# DEFAULT_STATE = {
#     name: {"active": True, "customers": []}
#     for name in list(QUEUES.keys()) + ["Queue A", "Queue B"]
# }


def ensure_data_dir():
    """Ensure data directory exists"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)


# def load_queue_states():
#     """Load queue states from file or create default"""
#     ensure_data_dir()
#     if os.path.exists(STATE_FILE):
#         with open(STATE_FILE, 'r') as f:
#             return json.load(f)
#     else:
#         return DEFAULT_STATE.copy()
# def get_default_state():
#     """Build default queue state from queue_config.QUEUES"""
#     return {
#         name: {
#             "active": config["active"],
#             "customers": config.get("customers", [])
#         }
#         for name, config in QUEUES.items()
#     }
def get_default_state():
    """Build default queue state from queue_config.QUEUES"""
    default_state = {}
    for name, config in QUEUES.items():
        default_state[name] = {
            "active": config.get("active", True),
            "customers": config.get("customers", [])
        }
    # # Add legacy queues if needed
    # default_state.update({
    #     "Queue A": {"active": False, "customers": []},
    #     "Queue B": {"active": False, "customers": []}
    # })
    return default_state


# def load_queue_states():
#     """Load queue states from file or fall back to default from queue_config"""
#     ensure_data_dir()
#     if os.path.exists(STATE_FILE):
#         with open(STATE_FILE, 'r') as f:
#             return json.load(f)
#     else:
#         return get_default_state()
def load_queue_states():
    """Load queue states with migration support"""
    ensure_data_dir()

    if not os.path.exists(STATE_FILE):
        return get_default_state()

    with open(STATE_FILE, 'r') as f:
        old_state = json.load(f)

    # Migration logic - merge with current configuration
    new_state = get_default_state()
    for queue_name in old_state:
        if queue_name in new_state:
            # Preserve existing queue data
            new_state[queue_name] = old_state[queue_name]

    return new_state


# def save_queue_states(state):
#     """Save queue states to file"""
#     ensure_data_dir()
#     with open(STATE_FILE, 'w') as f:
#         json.dump(state, f)
HISTORY_FILE = os.path.join(DATA_DIR, "queue_history.json")


def save_queue_states(state):
    """Save current state to queues.json and append to queue_history.json"""
    ensure_data_dir()

    # Save current state
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f)

    # Log queue lengths snapshot to history
    now = datetime.now().isoformat()
    history_entry = {
        "timestamp": now,
        "queues": {name: len(data["customers"]) for name, data in state.items()}
    }

    history_data = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            try:
                history_data = json.load(f)
            except json.JSONDecodeError:
                pass

    history_data.append(history_entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history_data, f, indent=2)


def active_queue_count():
    """
    Count the number of active queues based on load_queue_states.

    Returns:
        int: Number of queues where active is True.
    """
    queue_states = load_queue_states()
    sumcount = sum(1 for queue_data in queue_states.values()
                   if queue_data.get("active", False))
    if sumcount > 0:
        return sumcount
    else:
        return 1


# def simulate_step():
# """Simulate one step of queue activity"""
# now = datetime.now()
# state = load_queue_states()

# # Only simulate for active queues that exist in both config and state
# active_queues = [q for q in state.keys()
#                  if q in QUEUES and state[q]["active"]]

# for name in active_queues:
#     queue = state[name]["customers"]

#     # Add new arrivals (0-2)
#     arrivals = random.randint(0, 2)
#     for _ in range(arrivals):
#         queue.append({"arrived_at": now.isoformat()})

#     # Remove served (0-1)
#     served = random.randint(0, 1)
#     for _ in range(served):
#         if queue:
#             queue.pop(0)

# save_queue_states(state)
# return state
def get_default_state():
    """Build default queue state from queue_config.QUEUES"""
    default_state = {}
    for name, config in QUEUES.items():
        default_state[name] = {
            "active": config.get("active", True),
            "customers": config.get("customers", [])
        }
    # Add legacy queues if needed
    default_state.update({
        "Queue A": {"active": False, "customers": []},
        "Queue B": {"active": False, "customers": []}
    })
    return default_state


def load_queue_states():
    """Load queue states with migration support"""
    ensure_data_dir()

    if not os.path.exists(STATE_FILE):
        return get_default_state()

    with open(STATE_FILE, 'r') as f:
        old_state = json.load(f)

    # Migration logic - merge with current configuration
    new_state = get_default_state()
    for queue_name in old_state:
        if queue_name in new_state:
            # Preserve existing queue data
            new_state[queue_name] = old_state[queue_name]

    return new_state


def simulate_step():
    """Simulate one step of queue activity"""
    now = datetime.now()
    state = load_queue_states()

    # Get all queues from config plus legacy queues
    all_queues = list(QUEUES.keys()) + ["Queue A", "Queue B"]

    for name in all_queues:
        if name not in state:
            continue

        if not state[name]["active"]:
            continue

        queue = state[name]["customers"]

        # Add new arrivals (weighted probability)
        if random.random() < 0.55:  # 70% chance of new arrival
            arrivals = random.randint(0, 2)
            for _ in range(arrivals):
                queue.append({"arrived_at": now.isoformat()})

        # Process service completions
        if queue and random.random() < 0.5:  # 50% chance of service completion
            queue.pop(0)

    save_queue_states(state)
    return state


def get_customers_served(queue_history: pd.DataFrame) -> int:
    """
    Calculate the total number of customers served based on the history of queue counts.

    Args:
        queue_history (pd.DataFrame): DataFrame containing timestamp and total queue counts.

    Returns:
        int: Total number of customers served.
    """
    total_served = 0
    previous_count = 0

    # Iterate through each row in the DataFrame
    for index, row in queue_history.iterrows():
        current_count = row["queue_count"]

        # If this is not the first entry, calculate how many customers were served
        if index > 0:
            if previous_count > current_count:
                total_served += previous_count - current_count

        previous_count = current_count

    return total_served


def get_queue_lengths():
    """Get current length of each queue"""
    state = load_queue_states()
    return {name: len(data["customers"]) for name, data in state.items()}


queue_lengths = get_queue_lengths()


def get_total_queue_length(queue_lengths):
    return sum(queue_lengths.values())


def get_queue_activity():
    """Get active status of each queue"""
    state = load_queue_states()
    return {name: data["active"] for name, data in state.items()}


def get_queue_state(queue_name):
    """Get state of a specific queue"""
    state = load_queue_states()
    return state.get(queue_name, {}).get("active", False)


# def set_queue_active(queue_name, is_active):
#     """Set active status of a queue"""
#     state = load_queue_states()
#     if queue_name in state:
#         state[queue_name]["active"] = is_active
#         save_queue_states(state)
#     return state


def set_queue_active(queue_name, is_active):
    """Set active status of a queue with immediate feedback"""
    state = load_queue_states()
    if queue_name in state:
        state[queue_name]["active"] = is_active
        save_queue_states(state)
        st.session_state.last_state_update = time.time()  # Force refresh
    return state


HISTORY_FILE = "data/queue_history.json"


def load_total_queue_history():
    with open(HISTORY_FILE, 'r') as f:
        raw = json.load(f)

    records = []
    for entry in raw:
        timestamp = pd.to_datetime(entry["timestamp"])
        total_count = sum(entry["queues"].values())
        records.append({
            "date": timestamp,
            "queue_count": total_count,
            "is_forecast": False
        })

    return pd.DataFrame(records)
