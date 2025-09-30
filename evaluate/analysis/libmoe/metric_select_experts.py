import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import ujson as json
from tqdm.notebook import tqdm
from utils import \
        percentages, \
        data2name, \
        percent_data, \
        calculate_expert_changes, \
        data_behaviour


with open("./results/result_metric.json", "r") as f:
    results = json.load(f)
# List of percentages in ascending order

# Tên file để lưu kết quả
output_file = "./results/expert_selection_change_perturbed_results_smoe.json"

# Dictionary để lưu trữ kết quả
results_data = {}
stages = [
    "smoe_perturbed", 
    "hyperrouter", 
    "smoe_cosinegating", 
    "smoe"
    ]
data_behaviour = [
    "mmbench_en_dev" ,
    'scienceqa_img', 
    'mmstar'
]


# Tính toán điểm số cho từng giai đoạn
for stage in tqdm( stages, desc="Processing Stages"):
    if "sigmoid" in stage: continue
    score_moe = {}

    for data_name in data_behaviour:
        progress = results['phi3mini-clip'][stage][data_name]
        score_moe[data_name] = {}
        sample_prev = []
        for percent in tqdm(percent_data, desc=f"Processing Percentages for {stage}"):
            path = progress[percent]["path"].replace("results.json", f"{data2name[data_name]}.json")
            # Đọc tệp JSON với ujson
            with open(path, "r") as f:
                data = json.load(f)
            # print(data)
            # In ra thông báo để kiểm tra việc tải thành công
            print(f"Loaded {percent} data successfully for {stage}")
            # print(data['logs'][0])
            # Xử lý từng log trong dữ liệu để lấy vision_expert
            sample_new = {}
            experts = ['0', '1', '17', '16', '22', '23']
            print(list(data['logs'][0]['vision_id_experts'][0][0].keys()))
            # '0', '1', '15', '16', '22', '23'
            for layer_id in list(experts):
                sample_tmp = []
                for sample in data['logs']:
                    sample_tmp.append(sample['vision_id_experts'][0][0][str(layer_id)])

                sample_new[layer_id] = sample_tmp
            # Kiểm tra và cập nhật score
            if len(sample_prev) == 0: 
                sample_prev = sample_new.copy()
            else:
                scores = []
                for layer_id in list(experts):
                    layer_id = str(layer_id)
                    try:
                        score = np.average([calculate_expert_changes(sample_new[layer_id][i], sample_prev[layer_id][i]) for i in range(len(sample_new[layer_id]))])
                    except:
                        breakpoint()
                    scores.append(score)
                score_moe[data_name][percent] = np.average(scores)
                results_data[stage] = score_moe

                # Ghi kết quả vào file JSON
                with open(output_file, "w") as out_file:
                    json.dump(results_data, out_file, indent=4)
                    print(f"Results saved to {output_file}")

                sample_prev = sample_new.copy()  # Cập nhật sample trước
            del data 
            del sample_new

    
