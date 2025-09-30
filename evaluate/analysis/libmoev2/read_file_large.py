import glob
import os 
import json
import ujson
import time

def calculate_expert_changes(epoch_1, epoch_2):
    
    score = []
    for token1, token2 in zip(epoch_1, epoch_2):
        changes = 0

        for e1, e2 in zip(token1, token2):
            if e1 != e2:
                changes += 1
        
        score.append(changes / len(token1))

    return np.average(score)

model_list = [
    "Full_smoe_sigmoidgating",
    "Full_smoe_share",
    "Full_smoe_plus_plus",
    "Full_smoe",
    "Full_smoe_tcmoe",
    "Full_xmoe",
    "Full_smoe_sharev3"
    
    
]
datas = [
    "mme",
    "mmmu_val",
    "mmstar",
    "mathvista_testmini"
]
for name_data in datas:
    for name_ml in model_list:
        data_selected_final = {} 
        path_root = f"/cm/archive/anonymous_new/revise_checkpoints/Xphi35-siglip224/SMOE/665K36/{name_ml}/analysts"
        # name_data = "mathvista_testmini"
        # Path to your input data
        def process(path_data):
            # Start timing
            start = time.time()
            try:
                # Load the input JSON file
                with open(path_data, 'r') as f:
                    data = ujson.load(f)
            except:
                return

            # Initialize list for processed data
            new_data = []
            data_selected = {}

            # Process the data
            for sample in data['logs']:
                sample = sample['logs_metrics_vision'][0][0]  # Access the nested structure
                # check logs moe layers in vision transformers
                for i in range(10):
                    if str(i) not in sample: return
                    
                new_sample = {}
                
                for id_layer, value in sample.items():
                    if "time_inference" in id_layer: continue
                    new_sample[id_layer] = {}
                    if id_layer not in data_selected:
                        data_selected[id_layer] = []
                    for name_metric, val_metric in value.items():
                        if "selected_experts" in name_metric: 
                            # new_sample[id_layer][name_metric] = val_metric[0]  # Store the first element
                            data_selected[id_layer].append(val_metric)
                            continue
                        new_sample[id_layer][name_metric] = val_metric
                
                # Append the processed sample to new_data
                new_data.append(new_sample)
                
            
            
            # End timing
            end = time.time()
            print(f"Processing time: {end - start:.2f} seconds")
            # Define output file path
            output_dir = os.path.dirname(path_data)  # Use the same directory as input
            
            with open(output_dir + "/results.json", 'r') as f:
                data_results = ujson.load(f)
            # /cm/archive/anonymous_new/revise_checkpoints/Xphi35-siglip224/SMOE/665K36/revise_Full_smoe_sharev3/analysts/0717_2237_llava...mstar_llava_model_args_82420a/mme.json
            name_model = data_results['model_configs']["model_args"].split('/')[-2]
            name_check = data_results['model_configs']["model_args"].split('/')[-1].split(',')[0].split('-')[-1]
            
            if name_model not in data_selected_final:
                data_selected_final[name_model] = {}
            if name_check not in data_selected_final[name_model]:
                data_selected_final[name_model][name_check] = []
            data_selected_final[name_model][name_check] = data_selected
            output_path = os.path.join(output_dir, name_data + "_processed_data.json")
            # Save new_data to a JSON file
            with open(output_path, 'w') as f:
                json.dump(new_data, f, indent=4)  # Using json for readability with indent
            with open(f"{path_root}/{name_data}_data_selected_final.json", 'w') as f:
                json.dump(data_selected_final, f, indent=4)  # Using json for readability with indent
            print(f"Saved processed data to: {output_path}")
        # with open(f"/cm/archive/anonymous/checkpoints/data_eval_experts_selected_{name_data}/data_selected_final.json", 'r') as f:
        #     data_selected_final = ujson.load(f)

        for path in glob.glob(f'{path_root}/*'):
            process(path + f"/{name_data}.json")
        percent_data = [
        #  'checkpoint-832',
        '4159',
        '8318',
        '12477',

        '16636',
        '20791'
        ]
        import numpy as np

                
        score_changes_final = {}

        for name_model in data_selected_final.keys():
            score_changes_final[name_model] = {}
            # process overtime process
            
            for i in range(1, len(percent_data)):
                score_changes_final[name_model][i-1] = {}
                for id_layer in range(27):
                    
                    id_layer = str(id_layer)
                    score_changes_final[name_model][i-1][id_layer] = []
                    changes_list = [
                        calculate_expert_changes(e1, e2)
                        for e1, e2 in zip(
                            data_selected_final[name_model][percent_data[i]][str(id_layer)],
                            data_selected_final[name_model][percent_data[i - 1]][str(id_layer)]
                        )
                    ]
                    score_changes = np.average(changes_list)
                    score_changes_final[name_model][i-1][id_layer].append(score_changes)

        with open(f"{path_root}/{name_data}_score_selected_final.json", 'w') as f:
            json.dump(score_changes_final, f, indent=4)