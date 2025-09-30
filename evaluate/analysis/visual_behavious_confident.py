import matplotlib.pyplot as plt
import numpy as np
import ujson as json
from tqdm import tqdm

moe2name = {
    "smoe_cosinegating": "Cosine Router",
    "smoe_sigmoidgating": "Sigmoid Router",
    "hyperrouter": "Hyper Router",
    "smoe_perturbed": "Perturbed Cosine Router",
    "smoe": "SMoE Router"
}


# Function to calculate entropy based on expert selection frequencies
def calculate_entropy(sample_prob, k=2, infor = None):
    # Sort the input probabilities in descending order
    # values_sorted = sorted(sample_prob, reverse=True)    
    p = sample_prob
    
    try:
        # Normalize `p` to ensure the total sums to 1 (valid probability distribution)
        p = [v / sum(p) for v in p]
    except ZeroDivisionError:
        breakpoint()
        print("Error: The sum of probabilities is zero. Please check the input values.")
        # return None
    
    # Calculate entropy using the formula: -sum(p * log2(p))
    entropy = -sum(v * np.log2(v) if v > 0 else 0 for v in p)
    
    return entropy
# List of data paths
path_datas = [
    "/cm/archive/anonymous/toolkitmoe/evaluate/logs/1023_0346_llava_v1.5_mme_llava_model_args_844770", # smoe
    "/cm/archive/anonymous/toolkitmoe/evaluate/logs/1022_2352_llava_v1.5_mme_llava_model_args_5b4c4c",  # cosine
    "/cm/archive/anonymous/toolkitmoe/evaluate/logs/1023_0008_llava_v1.5_mme_llava_model_args_58365d", # perturbed 
    "/cm/archive/anonymous/toolkitmoe/evaluate/logs/1023_0352_llava_v1.5_mme_llava_model_args_982940", # hyper
]

# Dictionary to store the final entropy scores for each stage
score_entropy_moes = {}

# Loop through each path in the dataset
for path in tqdm(path_datas, desc="Processing paths"):
    # Read and load the configuration and results data from JSON files
    with open(f"{path}/mme.json", "r") as f:
        data = json.load(f)
    with open(f"{path}/results.json", "r") as f:
        data_infor = json.load(f)

    # Extract and store model information
    domains = {}
    model = data_infor['model_configs']['model_args']
    stage = model.split("/")[7].split(",")[0]  # Extract the stage name
    name_model = model.split("/")[5]  # Extract the model name
    model_name = name_model  # Store the model name for file naming later
    infor = [name_model, stage]
    # Organize data by domain
    for subdata in data['logs']:
        if subdata['domain'][0] not in domains:
            domains[subdata['domain'][0]] = []
        domains[subdata['domain'][0]].append(subdata)

    # Create a dictionary to store entropy values for each subtask (domain)
    entropy_values = {subtask: [] for subtask in domains.keys()}

    # Determine the number of layers to process in the logs
    num_layers = ['26']

    # Plot radar charts for each layer
    for i, layer_idx in enumerate(num_layers):
        for j, (subtask, doc) in enumerate(domains.items()):
            sum_samples = 0
            # Extract expert selection data for the current layer
            for sample in doc:
                sum_samples += np.average([calculate_entropy(x) for x in sample['vision_id_experts'][0][0][str(layer_idx)]])
            sum_samples/= len(doc)
            
            entropy_values[subtask].append(sum_samples)

    # Save entropy values for the current path to the main score dictionary
    score_entropy_moes[stage] = entropy_values

# Save the final entropy values to a JSON file
output_json_path = "./results/confident_metric_entropy_values.json"
with open(output_json_path, 'w') as json_file:
    json.dump(score_entropy_moes, json_file, indent=4)

print(f"Entropy values saved to {output_json_path}")
