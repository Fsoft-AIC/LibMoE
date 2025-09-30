# Import necessary libraries and utility functions
from utils import check_exist_id_experts, percent_data
import glob
import json
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define functions to calculate scores for each dataset
def get_score_mme(data_logs):
    if isinstance(data_logs['results']['mme']["mme_cognition_score,none"], dict):
        score = (data_logs['results']['mme']["mme_cognition_score,none"]['total_score'] + data_logs['results']['mme']["mme_percetion_score,none"]['total_score'])/2800
    else: 
        score = (data_logs['results']['mme']["mme_cognition_score,none"] + data_logs['results']['mme']["mme_percetion_score,none"])/2800
    return score * 100

def get_score_mmbench(data_logs):
    score = data_logs['results']["mmbench_en_dev"]["gpt_eval_score,none"]
    return score 

def get_score_pope(data_logs):
    score = data_logs['results']["pope"]["pope_accuracy,none"] * 100
    return score 
def get_score_mmmu_pro_standard(data_logs):
    score = data_logs['results']["mmmu_pro_standard"]["mmmu_acc,none"] * 100
    return score 
def get_score_mmmu_pro_vision(data_logs):
    score = data_logs['results']["mmmu_pro_vision"]["mmmu_acc,none"] * 100
    return score 
def get_score_ai2d(data_logs):
    score = data_logs['results']["ai2d"]["exact_match,flexible-extract"] * 100
    return score 

def get_score_mmerealworld_lite(data_logs):
    score = data_logs['results']["mmerealworld_lite"]["mme_realworld_score,none"] * 100
    return score 

def get_score_scienceqa_img(data_logs):
    score = data_logs['results']["scienceqa_img"]["exact_match,none"] * 100
    return score 

def get_score_ocrbench(data_logs):
    score = data_logs['results']["ocrbench"]["ocrbench_accuracy,none"] * 100
    return score 

def get_score_mmstar(data_logs):
    score = (data_logs['results']["mmstar"]["coarse perception,none"] +
             data_logs['results']["mmstar"]["fine-grained perception,none"] +
             data_logs['results']["mmstar"]["instance reasoning,none"] +
             data_logs['results']["mmstar"]["logical reasoning,none"] +
             data_logs['results']["mmstar"]["math,none"] +
             data_logs['results']["mmstar"]["science & technology,none"]) / 6
    return score * 100

def get_score_mmmu_val(data_logs):
    score = data_logs['results']["mmmu_val"]["mmmu_acc,none"] * 100
    return score 

def get_score_textvqa_val(data_logs):
    score = data_logs['results']["textvqa_val"]["exact_match,none"] * 100
    return score 

def get_score_mathvista_testmini(data_logs):
    score = data_logs['results']["mathvista_testmini"]["gpt_eval_score,none"]
    return score 

def get_score_hallusion_bench_image(data_logs):
    score = data_logs['results']["hallusion_bench_image"]["aAcc,none"]
    return score 

def get_score_gqa(data_logs):
    score = data_logs['results']["gqa"]["exact_match,none"] * 100
    return score 
def get_seedbench_2_plus(data_logs):
    score = data_logs['results']["seedbench_2_plus"]["seedbench_2_plus_all,none"] * 100
    return score 
def get_infovqa_val(data_logs):
    score = data_logs['results']["infovqa_val"]["anls,none"] * 100
    return score 

# Dictionary linking dataset names to corresponding score functions
dataset_score_functions = {
    "mme": get_score_mme,
    "mmbench_en_dev": get_score_mmbench,
    "ai2d": get_score_ai2d,
    "scienceqa_img": get_score_scienceqa_img,
    "mmstar": get_score_mmstar,
    "mmmu_val": get_score_mmmu_val,
    "textvqa_val": get_score_textvqa_val,
    "mathvista_testmini": get_score_mathvista_testmini,
    "hallusion_bench_image": get_score_hallusion_bench_image,
    "mmerealworld_lite": get_score_mmerealworld_lite,
    "ocrbench": get_score_ocrbench, 
    "mmmu_pro_standard":get_score_mmmu_pro_standard,
    "seedbench_2_plus":get_seedbench_2_plus,
    "infovqa_val":get_infovqa_val,
    "gqa":get_score_gqa, 
    "pope": get_score_pope
}
def process_log_file(path_logs, results):
    """
    Process a single log file and extract relevant results.
    """
    path_insight = path_logs
    yes = False
    if "0714_2315_llava...l_mme_llava_model_args_579495" in path_insight:
        yes = True
        
    try:
        # Load data logs from JSON file
        with open(path_insight, "r") as f:
            data_logs = json.load(f)
        
        if "results" not in data_logs:
            
            return 
        # if yes:
        #     breakpoint()
        
    except:
        print("!")
        print(path_insight)
        return

    # model = data_logs['model_configs']['model_args']
    # "/cm/archive/anonymous/checkpoints/Xphi35-siglip224/SMOE/1M3/std_0.002_Full_smoe/checkpoint-27572/logs/0706_1521_llava...bench_llava_model_args_c7f12e/results.json"
    # /cm/archive/anonymous/checkpoints/Xphi35-siglip224/SMOE/1M3/std_0.002_Full_smoe/logs/0613_0336_llava...bench_llava_model_args_35cab6/results.json
    stage = path_insight.split("/")[-4] if "checkpoint-" in path_insight else "checkpoint-20791"
    name_model = path_insight.split("/")[-5] if "checkpoint-" in path_insight else path_insight.split("/")[-4]
    # if yes : breakpoint()
    for data_name in data_logs['results']:
        # if yes : 
        #     breakpoint()
        #     print(data_name)
        
        # if "textvqa_val" in data_name and yes == True:
        #     breakpoint()
        
        score_func = dataset_score_functions.get(data_name)
        if score_func:
            
            score = score_func(data_logs)
            if score is None:
                continue

            path_trainer_state = f"{path_insight.split('/logs/')[0]}/trainer_state.json"
            try:
                with open(path_trainer_state, "r") as s:
                    trainer_state = json.load(s)
            except Exception as e:
                print(e)
                continue

            # Ensure nested dictionary structure
            if name_model not in results:
                results[name_model] = {}
            check_exists = False
            if data_name not in results[name_model]:
                results[name_model][data_name] = {}
                check_exists = True
            # if stage not in results[name_model][data_name]:
            #     results[name_model][data_name][stage] = {}'
            time_consuming = 0
            for logs in trainer_state["log_history"]:
                time_consuming += logs["time_per_iteration"]
                break

            if check_exists not in results[name_model][data_name]:
                
                results[name_model][data_name][stage]= {
                    "score": score,
                    "num_input_tokens_seen": trainer_state.get('num_input_tokens_seen'),
                    # "total_flos": trainer_state.get('total_flos'),
                    'path': path_insight, 
                    "time_consuming": time_consuming
                }
                
            else:
                
                # if results[name_model][data_name][check]["score"] >  score:
                results[name_model][data_name][stage]= {
                    "score": score,
                    "num_input_tokens_seen": trainer_state.get('num_input_tokens_seen'),
                    # "total_flos": trainer_state.get('total_flos'),
                    'path': path_insight, 
                    
                }
            
def get_results(logs, path_save="./results/result_metric.json"):
    """
    Process log files and extract results based on model configurations using multi-threading.
    """
    # logs = []
    # for path in path_logs_results_list:
    #     for path_sub in glob.glob(path):
    #         for log in glob.glob(path_sub + "/*"):
    #             logs.append(log)
                
    logs = sorted(logs, key=os.path.getctime)
    if os.path.exists(path_save):


        with open(path_save, "r") as f:
            results = json.load(f)
    else:
        results ={}

    # Use ThreadPoolExecutor for multi-threaded log processing
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(process_log_file, path_logs, results) for path_logs in logs]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing logs"):
            pass

    os.makedirs(os.path.dirname(path_save), exist_ok=True)
    
    with open(path_save, 'w') as json_file:
        json.dump(results, json_file, indent=4)
    print(path_save)

# if __name__ == "__main__":
#     path_logs_results = [
#         # "/cm/archive/anonymous/toolkitmoe/evaluate/logs/*",
#         # "/cm/archive/anonymous/toolkitmoe/evaluate/logs/logs/*",
#         "/cm/archive/anonymous/checkpoints/logs_shared_moe/*",
#         # "/cm/archive/anonymous/toolkitmoe/evaluate/logs/*",
#     ]

#     get_results(path_logs_results, path_save="./results/result_metric_competesmoe.json")


if __name__ == "__main__":
    path_logs_all_results = "/cm/archive/anonymous_new/revise_checkpoints/Xphi35-siglip224/SMOE/665K36"
    from pathlib import Path

    root = Path(path_logs_all_results)
    json_files = list(root.rglob("*.json"))
    json_files_results = []
    for p in json_files:
        if "results" in str(p) and 'logs' in str(p) :
            # if "0714_2315_llava...l_mme_llava_model_args_579495" in str(p):
            #     print(p)
            #     breakpoint()
            json_files_results.append(str(p))
            
  
    get_results(logs=json_files_results, path_save="./results/result_metric_libmoev2_665k.json")
    