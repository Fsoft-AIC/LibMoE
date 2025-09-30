import matplotlib.pyplot as plt
import numpy as np
import ujson as json
from tqdm import tqdm

# Function to calculate entropy based on expert selection frequencies
def calculate_entropy(frequencies):
    total_frequency = sum(frequencies)
    if total_frequency == 0:
        return 0
    # Calculate probabilities for each expert based on frequency
    probabilities = [freq / total_frequency for freq in frequencies]
    # Compute entropy using the formula: -sum(p * log2(p))
    entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
    return entropy

# # List of data paths (currently using only one example path)
# path_datas = [
    
#     # "/cm/archive/anonymous/toolkitmoe/evaluate/logs/1008_2253_llava_v1.5_mme_llava_model_args_4f9d87",  # smoe
#     # "/cm/archive/anonymous/toolkitmoe/evaluate/logs/1008_2316_llava_v1.5_mme_llava_model_args_17cb19", # cosine
#     # "/cm/archive/anonymous/toolkitmoe/evaluate/logs/1008_2255_llava_v1.5_mme_llava_model_args_bd1332", # hyper
#     # "/cm/archive/anonymous/toolkitmoe/evaluate/logs/1008_2317_llava_v1.5_mme_llava_model_args_d97d84", # per
    
#     "/cm/archive/anonymous/toolkitmoe/evaluate/logs/1008_1629_llava_v1.5_mme_llava_model_args_93d999", # smoe
#     "/cm/archive/anonymous/toolkitmoe/evaluate/logs/1008_1712_llava_v1.5_mme_llava_model_args_5b4c4c",  # cosine
#     "/cm/archive/anonymous/toolkitmoe/evaluate/logs/1008_1607_llava_v1.5_mme_llava_model_args_982940", # hyper
#     "/cm/archive/anonymous/toolkitmoe/evaluate/logs/1008_1651_llava_v1.5_mme_llava_model_args_8312e2", # perturbed
    
# ]
# # Style configuration for the text in the plots
# font_style = {
#     'family': 'serif',
#     'color':  'black',
#     'weight': 'normal',
#     'size': 16,
# }

# # Dictionary to store the final entropy scores for each stage
# score_entropy_moes = {}

# # Loop through each path in the dataset
# for path in tqdm(path_datas):
#     # Read and load the configuration and results data from JSON files
#     with open(f"{path}/mme.json", "r") as f:
#         data = json.load(f)
#     with open(f"{path}/results.json", "r") as f:
#         data_infor = json.load(f)

#     # Extract and store model information
#     domains = {}
#     model = data_infor['model_configs']['model_args']
#     stage = model.split("/")[7].split(",")[0]  # Extract the stage name
#     name_model = model.split("/")[5]  # Extract the model name
#     model_name = name_model  # Store the model name for file naming later

#     # Organize data by domain
#     for subdata in data['logs']:
#         if subdata['domain'][0] not in domains:
#             domains[subdata['domain'][0]] = []
#         domains[subdata['domain'][0]].append(subdata)

#     # Create a dictionary to store entropy values for each subtask (domain)
#     entropy_values = {subtask: [] for subtask in domains.keys()}

#     # Determine the number of layers to process in the logs
#     num_layers = len(data['logs'][0]['vision_id_experts'][0][0].keys())

#     # Setup the radar charts with a grid layout
#     rows, cols = int(num_layers / 3 + 1), 3  # Define rows and columns for subplots
#     labels = [f'E_{i}' for i in range(4)]  # Labels for experts (E_0, E_1, ...)
#     angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
#     angles += angles[:1]  # Complete the circle for radar chart

#     # Create the figure with subplots for radar charts
#     fig, axes = plt.subplots(rows, cols, figsize=(22, 32), subplot_kw=dict(polar=True))
#     axes = axes.flatten()  # Flatten the axes for easy iteration

#     # Setup color palette for different domains
#     colors = plt.get_cmap('tab20', len(domains))

#     # Plot radar charts for each layer
#     for i, layer_idx in enumerate(range(num_layers)):
#         ax = axes[i]  # Select the appropriate subplot

#         for j, (subtask, doc) in enumerate(domains.items()):
#             experts = []
#             # Extract expert selection data for the current layer
#             for sample in doc:
#                 experts.extend([x_sub for x in sample['vision_id_experts'][0][0][str(layer_idx)] for x_sub in x])

#             total_choices = len(experts)  # Total selections for the current layer
#             if total_choices == 0:
#                 continue

#             # Calculate frequency and percentage of each expert being selected
#             expert_counts = [experts.count(expert_id) for expert_id in range(4)]
#             expert_percentages = [(count / total_choices) * 100 for count in expert_counts]

#             # Calculate entropy and store the value for the current subtask
#             entropy_value = calculate_entropy(expert_counts)
#             entropy_values[subtask].append(entropy_value)

#             # Plot the radar chart for the current subtask
#             stats = expert_percentages + expert_percentages[:1]  # Close the radar chart circle
#             ax.plot(angles, stats, linewidth=1.5, linestyle='solid', label=subtask, color=colors(j))
#             ax.fill(angles, stats, alpha=0.1, color=colors(j))

#         # Set title and labels for the radar chart
#         ax.text(0.5, -0.1, f'Layer {layer_idx}', ha='center', va='center', transform=ax.transAxes, fontdict=font_style)
#         ax.set_yticklabels([])
#         ax.set_xticks(angles[:-1])
#         ax.set_xticklabels(labels)

#     # Remove unused subplots
#     for i in range(num_layers, len(axes)):
#         fig.delaxes(axes[i])

#     # Adjust layout and add legend
#     plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5), title="Subtasks", fontsize=10)
#     plt.subplots_adjust(top=0.90, bottom=0.05, hspace=0.3, wspace=0.3)

#     # Save the radar chart figure
#     pdf_path = f"./results/{name_model}_{stage}.png"
#     plt.savefig(pdf_path, format='png')
#     plt.show()

#     # Calculate the average entropy score for each domain and store it
#     score = {domain: np.mean(entropy_values[domain]) for domain in entropy_values.keys()}
#     score_entropy_moes[stage] = score
# score_entropy_moes = 
# # Visualization of the calculated entropy for each stage in a bar chart
# fig, axes = plt.subplots(1, 4, figsize=(28, 6))  # Create a 1x4 subplot layout

# # Plot each stage's entropy values in separate subplots
# for i, (stage, scores) in enumerate(score_entropy_moes.items()):
#     ax = axes[i]  # Select the subplot for the current stage
#     domains = list(scores.keys())
#     entropy_values = list(scores.values())

#     # Sort the domains by entropy value in descending order
#     sorted_pairs = sorted(zip(entropy_values, domains), reverse=True)
#     sorted_entropy_values, sorted_domains = zip(*sorted_pairs)

#     # Remove the black border (spines) from the subplot
#     for spine in ax.spines.values():
#         spine.set_visible(False)  # Remove all the spines

#     # Create a horizontal bar chart
#     bars = ax.barh(sorted_domains, sorted_entropy_values, color='skyblue')
#     for index, value in enumerate(sorted_entropy_values):
#         ax.text(value, index, f'{value:.4f}', va='center')

#     # Configure axes labels and titles
#     ax.set_xlim(min(sorted_entropy_values) - 0.01, max(sorted_entropy_values) + 0.01)

#     # Set titles based on the router type
#     if "pertur" in stage:
#         stage = "Perturbed Cosine Router"
#     elif "cosin" in stage:
#         stage = "Cosine Router"
#     elif "hyper" in stage:
#         stage = "Hyper Router"
#     else:
#         stage = "SMoE Router"
#     ax.set_title(f'{stage}', fontsize=14)
#     ax.set_xlabel('Entropy')
#     ax.set_ylabel('Domains')

# # Adjust layout and save the final figure
# plt.tight_layout()
# plt.savefig(f"./results/entropy_{model_name}.png", format='png')
# plt.savefig(f"./results/entropy_{model_name}.pdf", format='pdf')
# plt.show()
