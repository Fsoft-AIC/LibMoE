import torch

torch.backends.cuda.matmul.allow_tf32 = True


import copy
from tqdm import tqdm
from datetime import timedelta

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
from packaging import version
import warnings
import sys
import os

warnings.filterwarnings("ignore")
#Setup env
sys.path.append(f"{os.environ['TOOLKIT_DIR']}")
from loguru import logger as eval_logger
from moe_model.model.builder import load_pretrained_model
from moe_model.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from moe_model.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from moe_model.conversation import conv_templates


# inference implementation for attention, can be "sdpa", "eager", "flash_attention_2". Seems FA2 is not effective during inference: https://discuss.huggingface.co/t/flash-attention-has-no-effect-on-inference/73453/5
# if is_flash_attn_2_available:
#     best_fit_attn_implementation = "flash_attention_2" # flash_attn has a bug that says: ERROR Error query and key must have the same dtype in generating

if version.parse(torch.__version__) >= version.parse("2.1.2"):
    best_fit_attn_implementation = "sdpa"
else:
    best_fit_attn_implementation = "eager"


import torch

torch.backends.cuda.matmul.allow_tf32 = True


import copy
from tqdm import tqdm
from datetime import timedelta

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.utils import stop_sequences_criteria

from accelerate import Accelerator, DistributedType, InitProcessGroupKwargs
from accelerate.state import AcceleratorState
from typing import List, Optional, Union, Tuple
from packaging import version
import warnings

warnings.filterwarnings("ignore")

from loguru import logger as eval_logger

@register_model("llava")
class Llava(lmms):
    """
    Llava Model
    """

    def __init__(
        self,
        pretrained: str = "liuhaotian/llava-v1.5-7b",
        truncation: Optional[bool] = True,
        device: Optional[str] = "cuda:0",
        batch_size: Optional[Union[int, str]] = 1,
        model_name=None,
        attn_implementation=best_fit_attn_implementation,
        device_map="cuda:0",
        conv_template="vicuna_v1",
        use_cache=True,
        tie_weights: bool = True,
        truncate_context=False,  # whether to truncate the context in generation, set it False for LLaVA-1.6
        customized_config=None,  # ends in json
        **kwargs,
    ) -> None:
        super().__init__()
        # Do not use kwargs for now
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        self.conv_template = conv_template
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        self.accelerator = accelerator
        print(f"accelerator.num_processes: {accelerator.num_processes}")
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        elif accelerator.num_processes == 1 and device_map == "auto":
            self._device = torch.device(device)
            self.device_map = device_map
        else:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"

        model_name = model_name if model_name is not None else get_model_name_from_path(pretrained)
        try:
            # Try to load the model with the multimodal argument
            self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(
                model_path = pretrained,
                model_base = None,
                model_name = pretrained,
                use_flash_attn = True,
                device_map = "cuda:0"
                
            )
            self._model.config.training = False
        except Exception as e:
            print(f"Model is not MultiModal LLM: {e}")
            # # for older versions of LLaVA that don't have multimodal argument
            # llava_model_args.pop("multimodal", None)
            # self._tokenizer, self._model, self._image_processor, self._max_length = load_pretrained_model(pretrained, None, model_name, device_map=self.device_map, **llava_model_args)
        # self.model = self._model
        
        self._config = self._model.config
        self.model.eval()
        self._model.eval()
        # if tie_weights:
        #     self.model.tie_weights()

        self.truncation = truncation
        self.batch_size_per_gpu = int(batch_size)
        self.conv_template = conv_template
        self.use_cache = use_cache
        self.truncate_context = truncate_context
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [DistributedType.FSDP, DistributedType.MULTI_GPU, DistributedType.DEEPSPEED], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            # If you want to use DistributedType.DEEPSPEED, you have to run accelerate config before using the model
            # Also, you have to select zero stage 0 (equivalent to DDP) in order to make the prepare model works
            # I tried to set different parameters in the kwargs to let default zero 2 stage works, but it didn't work.
            if accelerator.distributed_type == DistributedType.DEEPSPEED:
                kwargs = {
                    "train_micro_batch_size_per_gpu": self.batch_size_per_gpu,
                    "train_batch_size": self.batch_size_per_gpu * accelerator.num_processes,
                }
                AcceleratorState().deepspeed_plugin.deepspeed_config_process(must_match=True, **kwargs)
                eval_logger.info("Detected that you are using DistributedType.DEEPSPEED. Make sure you run `accelerate config` and set zero stage to 0")

            if accelerator.distributed_type == DistributedType.FSDP or accelerator.distributed_type == DistributedType.DEEPSPEED:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        elif accelerator.num_processes == 1 and device_map == "auto":
            eval_logger.info(f"Using {accelerator.num_processes} devices with tensor parallelism")
            self._rank = 0
            self._word_size = 1
        else:
            eval_logger.info(f"Using single device: {self._device}")
            self.model.to(self._device)
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        if hasattr(self, "accelerator"):
            return self.accelerator.unwrap_model(self._model)
        else:
            return self._model

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def tok_encode(self, string: str, left_truncate_len=None, add_special_tokens=None) -> List[int]:
        """ """
        add_special_tokens = False if add_special_tokens is None else add_special_tokens
        encoding = self.tokenizer.encode(string, add_special_tokens=add_special_tokens)
        # left-truncate the encoded context to be at most `left_truncate_len` tokens long
        if left_truncate_len:
            encoding = encoding[-left_truncate_len:]
        return encoding

    def tok_decode(self, tokens):
        try:
            return self.tokenizer.decode(tokens)
        except:
            return self.tokenizer.decode([tokens])

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        # TODO
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for contexts, doc_to_target, doc_to_visual, doc_id, task, split in [reg.args for reg in requests]:
            # encode, pad, and truncate contexts for this batch
            if type(doc_to_target) == str:
                continuation = doc_to_target
            else:
                continuation = doc_to_target(self.task_dict[task][split][doc_id])
            visuals = [doc_to_visual(self.task_dict[task][split][doc_id])]
            visuals = self.flatten(visuals)
            image_sizes = [[visual.size[0], visual.size[1]] for visual in visuals]
            if visuals:
                image = process_images(visuals, self._image_processor, self._config)
                if type(image) is list:
                    image = [_image.to(dtype=torch.float16, device=self.device) for _image in image]
                else:
                    image = image.to(dtype=torch.float16, device=self.device)
            else:
                image = None

            prompts_input = contexts[0] if isinstance(contexts, list) else contexts

            if image is not None and len(image) != 0 and DEFAULT_IMAGE_TOKEN not in prompts_input:
                """
                Three senarios:
                1. No image, and there for, no image token should be added.
                2. image token is already specified in the context, so we don't need to add it.
                3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                """
                image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visuals)
                image_tokens = " ".join(image_tokens)
                prompts_input = image_tokens + "\n" + (contexts[0] if isinstance(contexts, list) else contexts)

            # This is much safer for llama3, as we now have some object type in it
            if "llama_3" in self.conv_template:
                conv = copy.deepcopy(conv_templates[self.conv_template])
            else:
                conv = conv_templates[self.conv_template].copy()
            conv.append_message(conv.roles[0], prompts_input)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            contxt_id = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # Add the answer of the second role
            conv.messages[1][1] = continuation

            prompt = conv.get_prompt()
            input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            labels = input_ids.clone()
            # Context part no need to calculate for loss
            labels[0, : contxt_id.shape[1]] = -100
            with torch.inference_mode():
                outputs = self.model(input_ids=input_ids, labels=labels, images=image, use_cache=True, image_sizes=image_sizes)
            loss = outputs["loss"]
            # loss = torch.exp(loss)
            logits = outputs["logits"]
            greedy_tokens = logits.argmax(dim=-1)
            cont_toks = input_ids[:, contxt_id.shape[1] :]  # [1, seq]
            greedy_tokens = greedy_tokens[:, contxt_id.shape[1] : input_ids.shape[1]]  # [1, seq]
            max_equal = (greedy_tokens == cont_toks).all()
            res.append((float(loss.item()), bool(max_equal)))
            pbar.update(1)
        pbar.close()
        return res

    def flatten(self, input):
        new_list = []
        for i in input:
            for j in i:
                new_list.append(j)
        return new_list
    def compute_entropy_topk(self, weight_experts):
        '''
        Compute mean entropy for weight_experts with shape (B, N, K)
        B: Batch size
        N: Number of experts
        K: Number of top-k weights
        '''
        # Normalize the weights along the last axis (K) to get probabilities
        probs = weight_experts / weight_experts.sum(dim=-1, keepdim=True)
        
        # Compute entropy for each element in the batch
        entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # Add a small value to avoid log(0)
        # # we are normalization values to compare between vectors have diversity size (example deepseek spent 1 shared expert, so it just active n - 1 experts)
        # K = weight_experts.size(-1)
        # import math
        # max_entropy = math.log(K)          # cùng cơ số với torch.log (tức ln)
        # entropy_norm = entropy / max_entropy
        # Return mean entropy over the batch
        return torch.mean(entropy)
    def compute_expert_distribution(self, selected_experts: torch.Tensor, number_experts: int):
        """
        Converts a tensor of expert indices to one-hot encoding and computes the distribution 
        (i.e. the count of samples assigned to each expert).

        The function supports two scenarios:
        - When the input tensor has shape [batch, samples, 1] (top-1 selection)
        - When the input tensor has shape [batch, samples, k] (top-k selection, e.g. top-2)
        
        For top-k selections, the function will convert each index to one-hot and then sum 
        across the k-dimension to get the aggregated one-hot representation for each sample.

        Args:
            selected_experts (torch.Tensor): Tensor containing expert indices.
                                            Expected shape is [batch, samples, 1] for top-1 or 
                                            [batch, samples, k] for top-k.
            number_experts (int): Total number of experts.

        Returns:
            onehot (torch.Tensor): One-hot encoded tensor.
                                For top-1, the shape is [batch, samples, number_experts].
                                For top-k, the shape is [batch, samples, number_experts] after summing over k.
            expert_distribution (torch.Tensor): Tensor of shape [number_experts] containing the count 
                                                of samples assigned to each expert.
        """
        # Check the number of top selections based on the last dimension
        top_k = selected_experts.shape[-1]
        
        import torch.nn.functional as F

        if top_k == 1:
            # For top-1 selection, remove the last singleton dimension: [batch, samples, 1] -> [batch, samples]
            indices = selected_experts.squeeze(-1)
            # Convert indices to one-hot encoding: shape becomes [batch, samples, number_experts]
            onehot = F.one_hot(indices, num_classes=number_experts)
        else:
            # For top-k selections (e.g., top-2), keep the last dimension.
            # Convert each index to one-hot: shape becomes [batch, samples, k, number_experts]
            onehot_k = F.one_hot(selected_experts, num_classes=number_experts)
            # Sum over the top-k dimension to aggregate multiple selections per sample:
            # resulting shape becomes [batch, samples, number_experts]
            onehot = onehot_k.sum(dim=-2)

        # Compute the expert distribution by summing over batch and sample dimensions.
        expert_distribution = onehot.sum(dim=(0, 1))
        
        return expert_distribution
    def generate_until(self, requests: List[Instance], **kwargs) -> List[str]:
        res = []
        # breakpoint()
        logs_metrics_vision,  logs_metrics_mlp=  [], []
        def _collate(x):
            # the negative sign on len(toks) sorts descending - this has a few advantages:
            # - time estimates will always be over not underestimates, which is more useful for planning
            # - to know the size of a batch when going through the list, you know the first one is always the batch
            #   padded context length. this is useful to simplify the batching logic and more importantly to make
            #   automatic adaptive batches much much easier to implement
            # - any OOMs will happen right away rather than near the end
            toks = self.tok_encode(x[0])
            return -len(toks), x[0]

        # we group requests by their generation_kwargs,
        # so that we don't try to execute e.g. greedy sampling and temp=0.8 sampling
        # in the same batch.
        re_ords = utils.Collator([reg.args for reg in requests], _collate, grouping=True)
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        
        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_id, task, split = zip(*chunk)
            task = task[0]
            split = split[0]
            batched_visuals = [doc_to_visual[0](self.task_dict[task][split][ids]) for ids in doc_id]  # [B, N]
            flattened_visuals = self.flatten(batched_visuals)  # [B*N]
            # we assume all gen kwargs in the batch are the same
            # this is safe to assume because the `grouper` object ensures it.
            gen_kwargs = all_gen_kwargs[0]

            # Set default values for until and max_new_tokens
            until = [self.tok_decode(self.eot_token_id)]

            # Update values from gen_kwargs if present
            if "until" in gen_kwargs:
                until = gen_kwargs.pop("until")
                if isinstance(until, str):
                    until = [until]
                elif not isinstance(until, list):
                    raise ValueError(f"Expected `gen_kwargs['until']` to be of type Union[str,list] but got {type(until)}")
            if "image_aspect_ratio" in gen_kwargs.keys() and "image_aspect_ratio" not in self._config.__dict__:
                # here we should pop it out of gen_kwargs so that it doesn't get passed to the model for next step of generation
                self._config.image_aspect_ratio = gen_kwargs.pop("image_aspect_ratio")
                eval_logger.info(f"Setting image aspect ratio: {self._config.image_aspect_ratio}")
            # encode, pad, and truncate contexts for this batch
            if flattened_visuals:
                image_tensor = process_images(flattened_visuals, self._image_processor, self._config)
                if type(image_tensor) is list:
                    image_tensor = [_image.to(dtype=torch.float16, device=self.device) for _image in image_tensor]
                else:
                    image_tensor = image_tensor.to(dtype=torch.float16, device=self.device)
            else:
                image_tensor = None

            # prompts_input = contexts[0]

            question_input = []
            
            for visual, context in zip(batched_visuals, contexts):
                if image_tensor is not None and len(image_tensor) != 0 and DEFAULT_IMAGE_TOKEN not in context:
                    """
                    Three senarios:
                    1. No image, and there for, no image token should be added.
                    2. image token is already specified in the context, so we don't need to add it.
                    3. image token is not specified in the context and there is image inputs, so we need to add it. In this case, we add the image token at the beginning of the context and add a new line.
                    """
                    image_tokens = [DEFAULT_IMAGE_TOKEN] * len(visual) if isinstance(visual, list) else [DEFAULT_IMAGE_TOKEN]
                    image_tokens = " ".join(image_tokens)
                    question = image_tokens + "\n" + context
                else:
                    question = context
                # This is much safer for llama3, as we now have some object type in it
                if "llama_3" in self.conv_template:
                    conv = copy.deepcopy(conv_templates[self.conv_template])
                else:
                    conv = conv_templates[self.conv_template].copy()
                conv.append_message(conv.roles[0], question)
                conv.append_message(conv.roles[1], None)
                prompt_question = conv.get_prompt()
                question_input.append(prompt_question)

            # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(self.device)
            # preconfigure gen_kwargs with defaults
            gen_kwargs["image_sizes"] = [flattened_visuals[idx].size for idx in range(len(flattened_visuals))]
            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            if "top_p" not in gen_kwargs:
                gen_kwargs["top_p"] = None
            if "num_beams" not in gen_kwargs:
                gen_kwargs["num_beams"] = 1
            input_ids_list = [tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in question_input]
            # pad_token_ids = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            pad_token_ids = self.tokenizer.eos_token_id
            input_ids = self.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_token_ids).to(self.device)
            attention_masks = input_ids.ne(pad_token_ids).to(self.device)
            # These steps are not in LLaVA's original code, but are necessary for generation to work
            # TODO: attention to this major generation step...
            
            try:
                
                self.model.eval()
                import time 
                start_time = time.time()
                with torch.inference_mode():
                    cont, vision_id_expert_tmp, mlp_id_expert = self.model.generate(
                        input_ids,
                        attention_mask=attention_masks,
                        pad_token_id=pad_token_ids,
                        images=image_tensor,
                        image_sizes=gen_kwargs["image_sizes"],
                        do_sample=True if gen_kwargs["temperature"] > 0 else False,
                        temperature=gen_kwargs["temperature"],
                        top_p=gen_kwargs["top_p"],
                        num_beams=gen_kwargs["num_beams"],
                        max_new_tokens=gen_kwargs["max_new_tokens"],
                        use_cache=self.use_cache,
                        return_id_experts = kwargs['return_id_experts']
                    )
                
                log_layers = {"time_inference": time.time() - start_time}
                
                pattern = r'\.layers\.(\d+)'
                import re
                
                if kwargs['return_id_experts'] == True and image_tensor is not None:
                    # Iterate through all layers in the model
                    for name, layer in self.model.named_modules():
                        # Check if the layer has the 'storage_metrixs' method
                        if hasattr(layer, 'log_metrics'):
                            id_layer = re.search(pattern, name)
                            match = re.search(pattern, name)
                            if match:
                                id_layer = str(match.group(1))  # Extract the matched group
                            else:
                                # print("errrorr!")
                                id_layer = 'mm_projector'  # Handle case where no match is found

                            log_layers[id_layer] = {}

                            if 'weights' in layer.log_metrics:
                                entropy_weight_topk = self.compute_entropy_topk(layer.log_metrics['weights'])
                                log_layers[id_layer]["entropy_weight_topk"] = entropy_weight_topk.item()

                            if 'gate_softmax' in layer.log_metrics:
                                entropy_weight_all = self.compute_entropy_topk(layer.log_metrics['gate_softmax'])
                                _, selected_experts_top1 = torch.topk(layer.log_metrics['gate_softmax'], k=1, dim=2)
                                num_experts = layer.log_metrics['gate_softmax'].shape[-1]
                                log_layers[id_layer]["entropy_weight_all"] = entropy_weight_all.item()
                                

                                # Chỉ compute dist_experts_top1 nếu có gate_softmax
                                dist_experts_top1 = self.compute_expert_distribution(selected_experts_top1, num_experts)
                                log_layers[id_layer]["dist_experts_top1"] = dist_experts_top1.squeeze().tolist()
                                
                                # compute confidence to experts selection
                            if 'selected_experts' in layer.log_metrics:
                                dist_experts_top2 = self.compute_expert_distribution(layer.log_metrics['selected_experts'], num_experts)
                                log_layers[id_layer]["dist_experts_top2"] = dist_experts_top2.squeeze().tolist()
                                # log_layers[id_layer]["selected_experts"] = layer.log_metrics['selected_experts'][0].tolist()
                            
                            if 'router_magine' in layer.log_metrics:
                                log_layers[id_layer]["router_magine"] = layer.log_metrics['router_magine'][0].tolist()
                             
                            # Các metric đơn giản khác
                            for key in ['balance_loss', 'router_z_loss', 'diver_loss']:
                                if key in layer.log_metrics:
                                    log_layers[id_layer][key] = layer.log_metrics[key]
                            for metric, value in layer.log_metrics.items():
                                if metric not in log_layers[id_layer] and isinstance(value, (int, float)):
                                    log_layers[id_layer][metric] = value
                text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)
                if isinstance(text_outputs, list):
                    for i in range(len(text_outputs)):
                        if "<|end|>" in text_outputs[i]:
                        
                            text_outputs[i] = text_outputs[i].replace("<|end|>", "").strip()
                        if "assistant\n" in text_outputs[i]:
                            text_outputs[i] = text_outputs[i].replace("assistant\n", "").strip()
                    if text_outputs[i] == '':
                        print("\n")
                        print(question_input[i])
                elif isinstance(text_outputs, str):
                    text_outputs = text_outputs.replace("<|end|>", "").strip()
                    text_outputs = text_outputs.replace("assistant\n", "").strip()
                else:
                    raise TypeError(f"Unexpected type for text_outputs: {type(text_outputs)}")
                # print(text_outputs)
            except Exception as e:
                raise e
                eval_logger.error(f"Error {e} in generating")
                cont = ""
                text_outputs = [""]

            res.extend(text_outputs)
            logs_metrics_vision.append(log_layers)
            logs_metrics_mlp.append(mlp_id_expert)
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), text_outputs)
            
            pbar.update(1)
            # reorder this group of results back to original unsorted form
        res = re_ords.get_original(res)
        logs_metrics_vision = re_ords.get_original(logs_metrics_vision)
        logs_metrics_mlp = re_ords.get_original(logs_metrics_mlp)
        pbar.close()
        return res, logs_metrics_vision, logs_metrics_mlp