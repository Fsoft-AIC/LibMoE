
from safetensors import safe_open
import json
import torch

# Đường dẫn tới tệp .safetensors
safetensors_path = "/cm/archive/anonymous/checkpoints/Xphi35-siglip224/demo/competesmoe/checkpoint-3000/model-00002-of-00003.safetensors"
json_output_path = "model_weights.json"

# Mở tệp .safetensors và đọc dữ liệu
state_dict = {}
with safe_open(safetensors_path, framework="pt", device="cuda:0") as f:
    for key in f.keys():
        state_dict[key] = f.get_tensor(key)

# Chuyển tensor thành danh sách (vì JSON không hỗ trợ tensor)
state_dict_json = {key: value.tolist() if torch.is_tensor(value) else value 
                   for key, value in state_dict.items()}
x = torch.tensor(state_dict_json[f"model.vision_tower.vision_model.encoder.layers.{0}.moelayer.prob_flips"], dtype=int)
for i in range(1, 27):
    x+=torch.tensor(state_dict_json[f"model.vision_tower.vision_model.encoder.layers.{i}.moelayer.prob_flips"],  dtype=int)
    # print((torch.tensor(state_dict_json_real[f"model.vision_tower.vision_model.encoder.layers.{i}.moelayer.prob_flips"]) != torch.tensor(state_dict_json[f"model.vision_tower.vision_model.encoder.layers.{i}.moelayer.prob_flips"])).sum()/8316)
x+=torch.tensor(state_dict_json[f"model.mm_projector.moelayer.prob_flips"],  dtype=int)


breakpoint()