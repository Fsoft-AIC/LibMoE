import argparse
import torch
import sys
import os

# Add the project's root directory to sys.path
sys.path.append('/cm/shared/anonymous_H102/toolkitmoe')

from moe_model.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from moe_model.conversation import conv_templates, SeparatorStyle
from moe_model.model.builder import load_pretrained_model
from moe_model.utils import disable_torch_init
from moe_model.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

from PIL import Image
import json
import requests
from PIL import Image
from io import BytesIO
from transformers import TextStreamer


def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def main(args):
    # Model
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, 
        args.model_base, 
        model_name, 
        args.load_8bit, 
        args.load_4bit, 
        device=args.device,
        use_flash_attn = True)
    model.config.training = False

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles

    image = load_image(args.image_file)
    image_size = image.size
    # Similar operation in model_worker.py
    image_tensor = process_images([image], image_processor, model.config)
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    convertation = []
    name_img = os.path.basename(args.image_file).split(".")[0]
    name_model = args.model_path.split("/")[-1]
    path = f"/cm/shared/anonymous_H102/toolkitmoe/moe_model/serve/examples/{name_model}_{name_img}.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            convertation = json.load(f)
    while True:
        try:
            inp = input(f"{roles[0]}: ")
        except EOFError:
            inp = ""
        # if not inp:
        #     print("exit...")
        #     break

        print(f"{roles[1]}: ", end="")
        subconv = {'user': inp, "image_path": args.image_file}
        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        attention_masks = input_ids.ne(tokenizer.eos_token_id).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_masks,
                images=image_tensor,
                image_sizes=[image_size],
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_id_experts = False,
                num_beams = 1
                )
        # Decode the output_ids assuming output_ids[0] is a tensor or list of token IDs
        if isinstance(output_ids[0], list):
           
            output_ids = output_ids[0]
    
        try:
            outputs = tokenizer.decode(output_ids[0][0], skip_special_tokens=True).strip()
            
        except:
            breakpoint()
        conv.messages[-1][-1] = outputs
        subconv['reponse'] = outputs
        convertation.append(subconv)
        
        
        with open(path, 'w') as file:
            json.dump(convertation, file, indent=4) 
        if args.debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-file", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--conv-mode", type=str, default="mistral_instruct_system")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=4000)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
