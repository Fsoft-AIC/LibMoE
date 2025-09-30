from datasets import load_dataset
import os

# Specify the dataset name and version
DATASET_NAME = "cerebras/SlimPajama-627B"

# Directory to save the dataset locally
SAVE_DIR = f"/cm/archive/anonymous/datasets/SlimPajama-627B"

# Create the directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

for split in ["validation", "test", "train"]:
    # Download the dataset with the specified version
    dataset = load_dataset(DATASET_NAME, split=split)

    # Save the dataset locally
    dataset.save_to_disk(SAVE_DIR)
    print(f"Dataset saved to {SAVE_DIR} in its native format.")

print("Dataset downloaded and saved locally.")
