
import os
# variable constant
percent_data = [
#  'checkpoint-832',
 'checkpoint-3328',
#  'checkpoint-2496',
 'checkpoint-6656',
#  'checkpoint-4160',
 'checkpoint-9984',
 
#  'checkpoint-5824',
 
 'checkpoint-13312',
#  'checkpoint-7488',
 'checkpoint-16632'
]


# select benchmarks for select experts between layer on a amount of data
data_behaviour = [
    "mmbench_en_dev" ,
    'scienceqa_img', 
    'mmstar'
]


percentages = ['20%', '30%', '40%', '50%', '60%', '70%', '80%', '90%', '100%']

data2name = {
    'scienceqa_img': 'scienceqa_img', 
    'mmstar': 'mmstar', 
    'mmbench_en_dev':'mmbench_en_dev'
}
'smoe_cosinegating', 'smoe', 'smoe_sigmoidgating', 'hyperrouter', 'smoe_perturbed'
moe2name = {
    "smoe_cosinegating": "Cosine Router",
    "smoe_sigmoidgating" : "Sigmoid Router",
    "hyperrouter": "Hyper Router",
    "smoe_perturbed": "Perturbed Cosine Router",
    "smoe": "SMoE Router"

}

def check_miss_results(results):
    # pass
    # check weght had miss evaluate 
    for stage in results['phi3mini-clip'].keys():
        for data_name in data_behaviour:

            for percent in percent_data:
                try:
                    x =  results['phi3mini-clip'][stage][data_name][percent]
                except:
                    print(f"stage: {stage} - data name: {data_name} - percent {percent}")

def calculate_expert_changes(epoch_1, epoch_2):
    changes = 0
    for e1, e2 in zip(epoch_1, epoch_2):
        if e1 != e2:
            changes += 1
    return changes / len(epoch_1)

def check_exist_id_experts(path_experts, memory_capacity = 200):
    try:
        # Đo kích thước tệp (tính bằng byte)
        file_size_bytes = os.path.getsize(path_experts)



        # Chuyển đổi sang đơn vị Megabyte (MB)
        file_size_mb = file_size_bytes / (1024 * 1024)
        if file_size_mb < memory_capacity:
            return False
    except:
        return False
    
    return True
import json
# Save the cached entropy scores to a file for future runs
def save_entropy_scores(cache_file, entropy_data):
    try:
        with open(cache_file, "w", encoding='utf-8') as f:
            json.dump(entropy_data, f, ensure_ascii=False, indent=4)
        print(f"Entropy scores successfully saved to {cache_file}")
    except Exception as e:
        print(f"An error occurred while saving the file: {e}")