
"""
Configuration settings for the QueueMind application.
"""

QUEUES = {
    "Standard Queue 1": {
        "id": "Q1",
        "capacity": 20,
        "avg_service_time": 2.5,  # minutes per person
        "opening_hours": {"start": "08:00", "end": "18:00"},
        "active": False, "customers": []
    },
    "Standard Queue 2": {
        "id": "Q2",
        "capacity": 15,
        "avg_service_time": 3.0,
        "opening_hours": {"start": "09:00", "end": "17:00"},
        "active": True, "customers": []
    },
    "Express Queue 3": {
        "id": "Q3",
        "capacity": 20,
        "avg_service_time": 1.5,
        "opening_hours": {"start": "08:00", "end": "18:00"},
        "active": True, "customers": []
    },
    # Add legacy queue names to ensure compatibility
    "Queue A": {
        "id": "QA",
        "capacity": 20,
        "avg_service_time": 2.5,
        "opening_hours": {"start": "08:00", "end": "18:00"},
        "active": True, "customers": []
    },
    "Queue B": {
        "id": "QB",
        "capacity": 15,
        "avg_service_time": 3.0,
        "opening_hours": {"start": "09:00", "end": "17:00"},
        "active": False, "customers": []
    }
}

SIMULATION_PARAMS = {
    "arrival_rate": 1,  # 0.8 people per minute
    "simulation_duration": 180,  # in minutes
}


PREDICTION_WINDOW = 30  # in minutes

PREDICTION_WINDOW_MINUTES = 30
