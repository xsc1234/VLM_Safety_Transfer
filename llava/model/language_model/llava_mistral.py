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


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from my_transformers import AutoConfig, AutoModelForCausalLM

from my_transformers.models.mistral.modeling_mistral_rag_mlp_fusion import MistralConfig, MistralModel, MistralForCausalLM

from my_transformers.modeling_outputs import CausalLMOutputWithPast
from my_transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaMistralConfig(MistralConfig):
    model_type = "llava_mistral"


class LlavaMistralModel(LlavaMetaModel, MistralModel):
    config_class = LlavaMistralConfig

    def __init__(self, config: MistralConfig):
        super(LlavaMistralModel, self).__init__(config)


class LlavaMistralForCausalLM(MistralForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMistralConfig

    def __init__(self, config):
        super(MistralForCausalLM, self).__init__(config)
        self.model = LlavaMistralModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_ids_original=None,
        add_text_ids=None,
        caption_ids=None,
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
        return_analysis: Optional[bool] = False,
        with_image: Optional[bool] = None,
        language_hidden_dict=None,
        rag_hidden_dict=None,
        pretrain=None,
        analysis_inference=False,
        new_ids=None,
        mean_text_features=None,
        first_generate=False,
        decode_layer=None,
        cut_flow_layers=None,
        repeat_layers=None,
        tox=None,
        steps_radio=None,
        return_analysis_rag=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if analysis_inference:
            input_ids_original = input_ids

        if (inputs_embeds is None and not return_analysis) or analysis_inference:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                new_ids,
                mean_text_features,
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                image_sizes,
                add_text_ids
            )
            mean_text_features = None
        return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                return_analysis=return_analysis,
                input_ids_original=input_ids_original,
                with_image=with_image,
                language_hidden_dict=language_hidden_dict,
                rag_hidden_dict=rag_hidden_dict,
                pretrain=pretrain,
                analysis_inference=analysis_inference,
                new_ids=new_ids,
                mean_text_features=mean_text_features,
                add_text_ids=add_text_ids,
                caption_ids=caption_ids,
                first_generate=first_generate,
                decode_layer=decode_layer,
                cut_flow_layers=cut_flow_layers,
                repeat_layers=repeat_layers,
                tox=tox,
                steps_radio=steps_radio,
                return_analysis_rag=return_analysis_rag,
            )

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        add_text_ids=None,
        decode_layer=None,
        cut_flow_layers=None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        mean_text_features=None
        new_ids=None
        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
                new_ids,
                mean_text_features,
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                add_text_ids=add_text_ids,
            )
            #print('mean text features is {}'.format(mean_text_features.shape))
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
            #print('no image')

        mean_text_features = None
        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            new_ids=new_ids,
            mean_text_features=mean_text_features,
            add_text_ids=add_text_ids,
            decode_layer=decode_layer,
            cut_flow_layers=cut_flow_layers,
            **kwargs
        )

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

AutoConfig.register("llava_mistral", LlavaMistralConfig)
AutoModelForCausalLM.register(LlavaMistralConfig, LlavaMistralForCausalLM)
