# QueueMind 🚦📊

QueueMind is an AI-powered, rule-based queue optimization platform that simulates, analyzes, and predicts queue behavior. It gives actionable recommendations to reduce wait times and optimize staffing.

> ⚠️ **OpenAI integration is commented out** in this version to allow offline, monolithic server deployments without external API dependencies.

---

## 📦 Features

- 🧠 **AI-driven recommendations** using rule-based logic, Prophet forecasting, and simulated data
- 📈 **30-minute queue forecasting** (via [Facebook Prophet](https://facebook.github.io/prophet/))
- 🏥 **Real-time queue monitoring** with imbalance detection and load-based alerts
- 🛠️ **Dynamic recommendations** based on capacity, customer spikes, wait time, and efficiency
- 🗓️ Simulates appointments and peak handling scenarios
- 📊 Visual insights with Streamlit + Plotly dashboards

---

## 🛠️ Tech Stack

| Component         | Technology               |
|------------------|--------------------------|
| Frontend         | Streamlit                |
| Backend Engine   | Python + Rule-Based Logic |
| Forecasting      | Facebook Prophet         |
| Visualization    | Plotly, Streamlit        |
| AI Suggestions   | GPT-4 / Copilot (optional) |
| Deployment Style | Single-file monolith     |

---

## 📂 Project Structure

queuemind/
├── app.py # Main Streamlit app
├── predictor.py # Forecasting with Prophet
├── state_manager.py # Queue simulation + state tracking
├── recommendations.py # Rule-based AI recommendations
├── queue_config.py # Queue and simulation config
├── queue_history.json # Historical queue data
├── queues.json # Live queue state
├── README.md # This file
└── requirements.txt # Dependencies


---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-org/queuemind.git
cd queuemind
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Run the app
```bash
streamlit run app.py
```
## all right, now you are ready to go 
## ⚠️⚠️⚠️ this should work mostly fine..if any issues try stopping the app and run trouble shooting step i mentioned at the end
 
⚙️ Configuration
Define your queues and simulation parameters in queue_config.py. Example:

```python

QUEUES = {
    "Counter A": {"capacity": 10, "avg_service_time": 2},
    "Counter B": {"capacity": 8, "avg_service_time": 3},
}

SIMULATION_SETTINGS = {
    "duration_minutes": 30,
    "arrival_rate_per_minute": 0.8,
}

PREDICTION_WINDOW_MINUTES = 30

```
## 🧠 AI Recommendations
AI recommendations are generated from historical patterns, load thresholds, and heuristics. Example rules:

- If wait time exceeds threshold → suggest adding staff
- If queue inactive but forecast predicts a spike → recommend activation
- If efficiency is low → recommend balancing across queues

## 🧠 OpenAI Integration (Optional)
The function `send_to_openai_for_summary()` uses GPT-4 to summarize recommendations. This is commented out in `recommendations.py` to allow smooth offline usage.

## 📤 Data Files
| File               | Purpose                                      |
|--------------------|----------------------------------------------|
| `queues.json`      | Stores real-time queue state                 |
| `queue_history.json`| Stores historical data for forecasting       |
| `queue_config.py`  | Configures queues and simulation parameters  |

## 📊 Sample Use Cases
- Call centers optimizing agent shifts
- Government service centers handling peak appointments
- Clinics balancing walk-ins vs. appointments
- Retail counters managing unexpected customer spikes

## 📸 Demonstration Link
Include screenshots of the dashboard here (if available):

## 🧠 AI Tools Mentioned
- GPT-4 (optional, disabled by default)
- GitHub Copilot (for developer acceleration)
- Prophet (used for time-series forecasting)
- Rule-Based Engine




## ✅ Status
This version uses rule-based logic + Prophet forecasting for 30-minute predictive planning. With more historical data, it can be extended to forecast hours or days ahead.



## > ⚠️ Troubleshooting

### Issue: Error due to Historical Data

If you encounter any errors, particularly related to the historical data (`queue_history.json`), it is often due to issues with the real-time data being written and processed.

**Solution**: 
1. Delete the following files from your project directory:
   - `queue_history.json`
   - `queues.json`

2. After deleting these files, restart the system, and it will regenerate fresh data for the queues and history.

This should resolve any issues related to corrupted or outdated data in the system.
