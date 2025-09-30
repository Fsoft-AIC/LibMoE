from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Point to the directory (or file) containing your event files.
event_acc = EventAccumulator('/cm/shared/anonymous/moeut_training_code/save/slimpajama_moe_44_no_attmoe_softmax/tensorboard')
event_acc.Reload()  # Load the events

# Extract scalar data for a specific tag, e.g., "loss"
scalars = event_acc.Scalars('train/reg_loss/moe')
for event in scalars:
    print(f"Step: {event.step}, Value: {event.value}, Wall time: {event.wall_time}")
