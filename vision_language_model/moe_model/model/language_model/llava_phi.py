#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# Copyright 2024 Jiachen Li
# ------------------------------------------------------------------------

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         Phi3Config, Phi3Model, Phi3ForCausalLM, Phi3Config
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
import copy
class LlavaPhiConfig(Phi3Config):
    model_type = "llava_phi"


class LlavaPhiModel(LlavaMetaModel, Phi3Model):
    config_class = LlavaPhiConfig

    def __init__(self, config: Phi3Config):
        super(LlavaPhiModel, self).__init__(config)


class LlavaPhiForCausalLM(Phi3ForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaPhiConfig

    def __init__(self, config):
        super(Phi3ForCausalLM, self).__init__(config)
        self.model = LlavaPhiModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                auxiliary_loss_mlp,
                auxiliary_loss_clip,
                vision_id_expert_tmp, 
                mlp_id_expert, 
                aux_mlp, 
                aux_clip
                
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                return_id_experts = False
            )

        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        out["trainer_logs"] = {}
        if self.config.training:
            if self.config.mlp_smoe or self.config.clip_smoe:
                out["trainer_logs"]['language_loss'] = out['loss'].detach().clone()
                loss = out['loss']
                if self.config.local_rank == 0:
                    print(f'Language loss: {loss.item()}')
                if self.config.mlp_smoe:
                    auxiliary_loss_mlp = auxiliary_loss_mlp.sum(dim=-1).mean()
                    loss += auxiliary_loss_mlp
                if self.config.clip_smoe:
                    auxiliary_loss_clip = auxiliary_loss_clip.sum(dim=-1).mean()
                    loss += auxiliary_loss_clip
                    
                
                if self.config.local_rank == 0:
                    if self.config.mlp_smoe:
                        print(f'mlp auxiliary loss: {auxiliary_loss_mlp.item()} ')
                        print(f'clip auxiliary loss: {auxiliary_loss_clip.item()} ')
                for i in aux_mlp.keys():
                    if i in aux_mlp.keys():
                        aux_mlp[i] = aux_mlp[i].float().sum(dim=-1).mean().detach().clone()
                        out["trainer_logs"][i + '_mlp'] = aux_mlp[i]
                for i in aux_clip.keys():
                    if i in aux_clip.keys():
                        aux_clip[i] = aux_clip[i].float().sum(dim=-1).mean().detach().clone()
                    
                        out["trainer_logs"][i+ '_clip'] = aux_clip[i]

                out["trainer_logs"]['auxiliary_loss_clip'] = auxiliary_loss_clip.detach().clone()
                out["trainer_logs"]['auxiliary_loss_mlp'] = auxiliary_loss_mlp.detach().clone()
                
                out['loss'] = loss

        return out
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
     
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")
        vision_id_expert_tmp, mlp_id_expert = {}, {}
        aux_mlp, aux_clip = {}, {}
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                _,
                _,
                vision_id_expert_tmp, 
                mlp_id_expert ,
                aux_mlp, 
                aux_clip
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                return_id_experts=kwargs['return_id_experts']
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        # print("Check: =========================== ")
        # print(kwargs)
        try:
            kwargs.pop("use_cache", None)
            kwargs.pop("cache_position", None)
            kwargs.pop("return_id_experts", None)
            return super().generate(
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                
                **kwargs
            ), aux_clip, aux_mlp 
        except TypeError as e:
            print(f"Warning: {e}")
            
            

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

AutoConfig.register("llava_phi", LlavaPhiConfig)
AutoModelForCausalLM.register(LlavaPhiConfig, LlavaPhiForCausalLM)
