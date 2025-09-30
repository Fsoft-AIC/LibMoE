import shutil
import os
import re

# Source folder path (the directory containing subfolders you want to move)
source_folder = "/cm/shared/anonymous_H102/checkpoints/phi35-siglip224/sft/dynamic_moe"

# Destination path (where you want to move the subfolders)
destination_folder = "/cm/archive/anonymous/checkpoints/clip-vit-large-patch14-336/test_model"

# Regular expression to capture the number at the end of the folder name
pattern = r"checkpoint-(\d+)$"

try:
    # Get a list of all subfolders in the source directory
    subfolders = [f for f in os.listdir(source_folder) if os.path.isdir(os.path.join(source_folder, f))]

    # Sort subfolders based on the numeric part at the end of their names
    subfolders.sort(key=lambda x: int(re.search(pattern, x).group(1)) if re.search(pattern, x) else float('inf'))

    # Leave the last subfolder untouched by excluding it from the iteration
    for subfolder in subfolders[:-1]:
        item_path = os.path.join(source_folder, subfolder)
        
        # Move the subfolder to the destination folder
        shutil.move(item_path, destination_folder)
        print(f"Moved {item_path} to {destination_folder}")
    
    print("All subfolders (except the last one) moved successfully.")
except Exception as e:
    print(f"Error: {e}")
