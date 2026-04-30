from moe_model.train.train import train
from huggingface_hub import login
import os
try:
    login(token = os.getenv('KEY_HF'))
except:
    print("Warning: Login hungingface fail!")


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
