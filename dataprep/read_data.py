import json
from tqdm import tqdm

# Load the original JSON data
with open("/cm/archive/anonymous/data/jsons/onevison_single_img.json", "r") as f:
    data = json.load(f)

# Process each entry and update the 'image' field
for i in tqdm(range(len(data)), desc="Processing"):
    if 'image' in data[i] and data[i]['image'] is not None:
        try:
            data[i]['image'] = data[i]['image']['path']
        except:
            breakpoint()
    else:
        if 'image' in data[i]:
            del data[i]['image'] 

# Save the updated data to a new JSON file
with open("/cm/archive/anonymous/data/jsons/onevision_single_img_standard.json", "w") as f:
    json.dump(data, f, indent=4)

print("Done! JSON file saved.")
