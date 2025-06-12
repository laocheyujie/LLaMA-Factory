# Copyright 2025 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/examples/pytorch/language-modeling/run_clm.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from itertools import chain
from typing import Any

from .processor_utils import DatasetProcessor


@dataclass
class PretrainDatasetProcessor(DatasetProcessor):
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build grouped texts with format `X1 X2 X3 ...` if packing is enabled
        # NOTE: examples:
        # examples = {
        #     "_prompt": [
        #         [{'content': 'Don’t think you need all the bells and whistles? No problem. McKinley Heating Service...rns related to your HVAC system addressed.', 'role': 'user'}],
        #         [{'content': 'To the apparent surprise of everyone, the Walt Disney Company has announced a deal to...e near future, and hope for years to come.', 'role': 'user'}]
        #     ],
        #     "_response": [
        #         [],
        #         []
        #     ],
        #     "_system": ["", ""],
        #     "_tools": ["", ""],
        #     "_images": [None, None],
        #     "_videos": [None, None],
        #     "_audios": [None, None]
        # }
        eos_token = "<|end_of_text|>" if self.data_args.template == "llama3" else self.tokenizer.eos_token
        text_examples = [messages[0]["content"] + eos_token for messages in examples["_prompt"]]
        # NOTE: text_examples
        # [
        #     'Don’t think you need all the bells and whistles? No problem. McKinley Heating Service...d to your HVAC system addressed.<|im_end|>', 
        #     'To the apparent surprise of everyone, the Walt Disney Company has announced a deal to...ure, and hope for years to come.<|im_end|>'
        # ]

        if not self.data_args.packing:
            if getattr(self.tokenizer, "add_bos_token", False):
                text_examples = [self.tokenizer.bos_token + example for example in text_examples]

            result = self.tokenizer(
                text_examples, add_special_tokens=False, truncation=True, max_length=self.data_args.cutoff_len
            )
        else:
            tokenized_examples = self.tokenizer(text_examples, add_special_tokens=False)
            # NOTE: tokenized_examples
            # {
            #     "input_ids": [
            #         [8002, 1405, 1744, 498, 1184, 678, 279, 60694, 323, 420],
            #         [65943, 30, 2308, 3491, 13, 66393, 8002, 1405, 1744, 498]
            #     ],
            #     "attention_mask": [
            #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            #         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            #     ]
            # }
            # NOTE: packing 就是把所有训练样本的token拼接到一起，然后按block_size切分
            concatenated_examples = {k: list(chain(*tokenized_examples[k])) for k in tokenized_examples.keys()}
            # NOTE: concatenated_examples
            # {
            #     "input_ids": [8002, 1405, 1744, 498, 1184, 678, 279, 60694, 323, 420, 65943, 30, 2308, 3491, 13, 66393, 8002, 1405, 1744, 498],
            #     "attention_mask": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            # }
            total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
            block_size = self.data_args.cutoff_len
            total_length = (total_length // block_size) * block_size
            # BUG:
            total_length = 1 if total_length == 0 else total_length
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            if getattr(self.tokenizer, "add_bos_token", False):
                for i in range(len(result["input_ids"])):
                    # BUG: 这里我感觉应该是在前面插入bos_token_id，而不是替换
                    # result["input_ids"][i] = [self.tokenizer.bos_token_id] + result["input_ids"][i]
                    result["input_ids"][i][0] = self.tokenizer.bos_token_id

        # NOTE: result
        # {
        #     'input_ids': [[8002, 1405, 1744, 498, 1184, 678, 279, 60694, 323, 420, 65943, 30, 2308, 3491, 13, 66393, 8002, 1405, 1744, 498]],
        #     'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
        # }
        return result

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        print("input_ids:\n{}".format(example["input_ids"]))
        print("inputs:\n{}".format(self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
