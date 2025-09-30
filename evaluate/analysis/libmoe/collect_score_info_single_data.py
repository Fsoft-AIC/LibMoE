from concurrent.futures import ThreadPoolExecutor, as_completed
from utils import check_exist_id_experts, percent_data
import glob
import json
from tqdm import tqdm
import os

# Define functions to calculate scores for each dataset
def get_score_mme(data_logs):
    score = data_logs['results']['mme']["mme_cognition_score,none"] + data_logs['results']['mme']["mme_percetion_score,none"]
    return score 

def get_score_mmbench(data_logs):
    score = data_logs['results']["mmbench_en_dev"]["gpt_eval_score,none"]
    return score 

def get_score_pope(data_logs):
    score = data_logs['results']["pope"]["pope_accuracy,none"] * 100
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

# Dictionary linking dataset names to corresponding score functions
dataset_score_functions = {
    # "mme": get_score_mme,
    "mmbench_en_dev": get_score_mmbench,
    "pope": get_score_pope,
    "ai2d": get_score_ai2d,
    # "scienceqa_img": get_score_scienceqa_img,
    "mmstar": get_score_mmstar,
    "mmmu_val": get_score_mmmu_val,
    "textvqa_val": get_score_textvqa_val,
    # "mathvista_testmini": get_score_mathvista_testmini,
    # "hallusion_bench_image": get_score_hallusion_bench_image,
    # "gqa": get_score_gqa,
    "mmerealworld_lite": get_score_mmerealworld_lite,
    "ocrbench": get_score_ocrbench
}

# Function to process each individual log file
def process_single_log(path_logs):
    if "logs/03" not in path_logs : return
    path_insight = f"{path_logs}/results.json"
    try:
        with open(path_insight, "r") as f:
            data_logs = json.load(f)
    except:
        return None  # Skip file if error occurs

    model = data_logs['model_configs']['model_args']
    check = None
    for name_percent in percent_data:
        if name_percent in model:
            check = name_percent

    if check is not None or "dense" in model:
        return None

    name_model, stage = "unknown", "unknown"
    if "CUMO" in model:
        name_model = model.split("/")[6]
        stage = name_model
    elif "namnv" in model:
        name_model = model.split("/")[5]
        try:
            stage = model.split("/")[7]
        except:
            return None
    else:
        name_model = model.split("/")[6]
        stage = model.split("/")[7]
    
    data_storage = "full" if "half" not in model else "half"
    results = {data_storage: {name_model: {stage: {}}}}
    for name_bench in dataset_score_functions.keys():
        
        if name_bench in list(data_logs['results'].keys()):

            score = dataset_score_functions.get(name_bench, lambda x: None)(data_logs)
        
            results[data_storage][name_model][stage]["scienceqa_img"] = {"score": score, "path": path_insight}
    # if "smoe,conv_template=phi35" == stage and  "scienceqa_img" in list(data_logs['results'].keys()):
    #     print(results)

    return results

# Main function to process log files in parallel
def get_results(path_logs_results_list, path_save="./results/result_metric.json"):
    logs = []
    for path_logs_results in path_logs_results_list:
        logs.extend(glob.glob(path_logs_results))
    
    final_results = {}

    # Using ThreadPoolExecutor to process logs in parallel
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_single_log, path_logs): path_logs for path_logs in logs}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing logs in parallel"):
            result = future.result()
            if result:
                for storage, models in result.items():
                    if storage not in final_results:
                        final_results[storage] = {}
                    for model, stages in models.items():
                        if model not in final_results[storage]:
                            final_results[storage][model] = {}
                        for stage, data in stages.items():
                            if stage not in final_results[storage][model]:
                                final_results[storage][model][stage] = {}
                            for data_name, score_data in data.items():
                                final_results[storage][model][stage][data_name] = score_data

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(path_save), exist_ok=True)

    # Save the processed results to a JSON file
    with open(path_save, 'w') as json_file:
        json.dump(final_results, json_file, indent=4)

if __name__ == "__main__":
    path_logs_results = [
        # "/cm/archive/anonymous/toolkitmoe/evaluate/logs/*",
        "/cm/shared/anonymous_H102/toolkitmoe/evaluate/logs/*",
        # "/cm/archive/anonymous/toolkitmoe/evaluate/logs/*",
        # "/cm/archive/anonymous/CUMO/lmms-eval/logs/*",
    ]
    get_results(path_logs_results, path_save="./results/result_metric_competesmoe_mmerealworld_lite.json")
