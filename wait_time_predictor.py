from state_manager import active_queue_count, get_queue_lengths, get_total_queue_length


def calculate_wait_times(current_queues, queues_config):
    """
    Calculate the current average wait time, predicted wait time (current optimization),
    and predicted wait time in 30 minutes based on expected load.


    Returns:
        tuple: (avg_wait_time, predicted_wait_time, predicted_wait_time_30min) in minutes.
    """
    # Calculate total customers across all queues

    queue_lengths = get_queue_lengths()
    total_customers = get_total_queue_length(queue_lengths)
    # print(total_customers)
    # print(current_queues)

    # Calculate current average wait time
    avg_wait_time = 0
    if total_customers > 0:
        for queue_name, count in current_queues.items():
            # print(count)
            if count > 0:
                queue_config = queues_config[queue_name]
                avg_wait_time = avg_wait_time + (count *
                                                 queue_config["avg_service_time"])
                # print("avg wait", count, avg_wait_time)
        # print("inside if", avg_wait_time)
        avg_wait_time = avg_wait_time / active_queue_count()
    # Predict current optimized wait time (30% reduction from AI optimization)
    predicted_wait_time = max(1, round(avg_wait_time * 0.7))

    # print("after func", avg_wait_time)

    # Predict queue load in 30 minutes (hardcoded: 20% increase in queue lengths)
    predicted_queues = {queue: round(count * 1.2)
                        for queue, count in current_queues.items()}
    predicted_total_customers = sum(predicted_queues.values())

    predicted_wait_time_30min = 0
    if predicted_total_customers > 0:
        for queue_name, count in predicted_queues.items():
            if count > 0:
                queue_config = queues_config[queue_name]
                predicted_wait_time_30min = predicted_wait_time_30min + (
                    count * queue_config["avg_service_time"])
        predicted_wait_time_30min = predicted_wait_time_30min / active_queue_count()
    # Apply 30% reduction for AI optimization in 30 minutes
    predicted_wait_time_30min = max(1, round(predicted_wait_time_30min * 0.7))
    # print("before final", avg_wait_time)
    return avg_wait_time, predicted_wait_time, predicted_wait_time_30min
