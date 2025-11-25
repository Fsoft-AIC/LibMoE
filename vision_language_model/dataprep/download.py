import os
from datasets import load_dataset
from tqdm.auto import tqdm
import json


data = load_dataset("lmms-lab/LLaVA-OneVision-Data", split="train", num_proc=200,)

image_folder = "./data/image_onevision"
os.makedirs(image_folder, exist_ok=True)

def process_example(example):
    # save image
    if example["image"] is not None:
        image_path = f"{example['id']}.jpg"

        img = example["image"]
        breakpoint()
        if img.mode != "RGB":
            img = img.convert("RGB")
        img_path = os.path.join(image_folder, image_path)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        img.save(img_path)

        example["image"] = image_path  # Update path in json
    else:
        example["image"] = None

    # Only keep required fields in the map output
    return {
        "id": example["id"],
        "data_source": example["data_source"],
        "image": example["image"],
        "conversations": example["conversations"],
    }

# Now use multi-processing map
converted_data = data.map(
    process_example,
    num_proc=200,
    desc="Processing images + json",
)

# Save to JSON file
converted_list = converted_data.to_list()
os.makedirs("/cm/archive/namnv78/data/jsons", exist_ok=True)

with open("./data/jsons/onevison_single_img.json", "w") as f:
    json.dump(converted_list, f, indent=4, ensure_ascii=False)
