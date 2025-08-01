# Copyright 2025 the LlamaFactory team.
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

import os
from collections import OrderedDict, defaultdict
from enum import Enum, unique
from typing import Optional

from peft.utils import SAFETENSORS_WEIGHTS_NAME as SAFE_ADAPTER_WEIGHTS_NAME
from peft.utils import WEIGHTS_NAME as ADAPTER_WEIGHTS_NAME
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME


AUDIO_PLACEHOLDER = os.getenv("AUDIO_PLACEHOLDER", "<audio>")

CHECKPOINT_NAMES = {
    SAFE_ADAPTER_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
}

CHOICES = ["A", "B", "C", "D"]

DATA_CONFIG = "dataset_info.json"

DEFAULT_TEMPLATE = defaultdict(str)

FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}

IGNORE_INDEX = -100

IMAGE_PLACEHOLDER = os.getenv("IMAGE_PLACEHOLDER", "<image>")

LAYERNORM_NAMES = {"norm", "ln"}

LLAMABOARD_CONFIG = "llamaboard_config.yaml"

METHODS = ["full", "freeze", "lora"]

MOD_SUPPORTED_MODELS = {"bloom", "falcon", "gemma", "llama", "mistral", "mixtral", "phi", "starcoder2"}

MULTIMODAL_SUPPORTED_MODELS = set()

PEFT_METHODS = {"lora"}

RUNNING_LOG = "running_log.txt"

SUBJECTS = ["Average", "STEM", "Social Sciences", "Humanities", "Other"]

SUPPORTED_MODELS = OrderedDict()

TRAINER_LOG = "trainer_log.jsonl"

TRAINING_ARGS = "training_args.yaml"

TRAINING_STAGES = {
    "Supervised Fine-Tuning": "sft",
    "Reward Modeling": "rm",
    "PPO": "ppo",
    "DPO": "dpo",
    "KTO": "kto",
    "Pre-Training": "pt",
}

STAGES_USE_PAIR_DATA = {"rm", "dpo"}

SUPPORTED_CLASS_FOR_S2ATTN = {"llama"}

SWANLAB_CONFIG = "swanlab_public_config.json"

VIDEO_PLACEHOLDER = os.getenv("VIDEO_PLACEHOLDER", "<video>")

V_HEAD_WEIGHTS_NAME = "value_head.bin"

V_HEAD_SAFE_WEIGHTS_NAME = "value_head.safetensors"


class AttentionFunction(str, Enum):
    AUTO = "auto"
    DISABLED = "disabled"
    SDPA = "sdpa"
    FA2 = "fa2"


class EngineName(str, Enum):
    HF = "huggingface"
    VLLM = "vllm"
    SGLANG = "sglang"


class DownloadSource(str, Enum):
    DEFAULT = "hf"
    MODELSCOPE = "ms"
    OPENMIND = "om"


@unique
class QuantizationMethod(str, Enum):
    r"""Borrowed from `transformers.utils.quantization_config.QuantizationMethod`."""

    BNB = "bnb"
    GPTQ = "gptq"
    AWQ = "awq"
    AQLM = "aqlm"
    QUANTO = "quanto"
    EETQ = "eetq"
    HQQ = "hqq"


class RopeScaling(str, Enum):
    LINEAR = "linear"
    DYNAMIC = "dynamic"
    YARN = "yarn"
    LLAMA3 = "llama3"


def register_model_group(
    models: dict[str, dict[DownloadSource, str]],
    template: Optional[str] = None,
    multimodal: bool = False,
) -> None:
    for name, path in models.items():
        SUPPORTED_MODELS[name] = path
        if template is not None and (
            any(suffix in name for suffix in ("-Chat", "-Distill", "-Instruct", "-Thinking")) or multimodal
        ):
            DEFAULT_TEMPLATE[name] = template

        if multimodal:
            MULTIMODAL_SUPPORTED_MODELS.add(name)


register_model_group(
    models={
        "Aya-23-8B-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/aya-23-8B",
        },
        "Aya-23-35B-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/aya-23-35B",
        },
    },
    template="cohere",
)


register_model_group(
    models={
        "Baichuan-7B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan-7B",
            DownloadSource.MODELSCOPE: "baichuan-inc/baichuan-7B",
        },
        "Baichuan-13B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan-13B-Base",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan-13B-Base",
        },
        "Baichuan-13B-Chat": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan-13B-Chat",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan-13B-Chat",
        },
    },
    template="baichuan",
)


register_model_group(
    models={
        "Baichuan2-7B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-7B-Base",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-7B-Base",
        },
        "Baichuan2-13B-Base": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-13B-Base",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-13B-Base",
            DownloadSource.OPENMIND: "Baichuan/Baichuan2_13b_base_pt",
        },
        "Baichuan2-7B-Chat": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-7B-Chat",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-7B-Chat",
            DownloadSource.OPENMIND: "Baichuan/Baichuan2_7b_chat_pt",
        },
        "Baichuan2-13B-Chat": {
            DownloadSource.DEFAULT: "baichuan-inc/Baichuan2-13B-Chat",
            DownloadSource.MODELSCOPE: "baichuan-inc/Baichuan2-13B-Chat",
            DownloadSource.OPENMIND: "Baichuan/Baichuan2_13b_chat_pt",
        },
    },
    template="baichuan2",
)


register_model_group(
    models={
        "BLOOM-560M": {
            DownloadSource.DEFAULT: "bigscience/bloom-560m",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloom-560m",
        },
        "BLOOM-3B": {
            DownloadSource.DEFAULT: "bigscience/bloom-3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloom-3b",
        },
        "BLOOM-7B1": {
            DownloadSource.DEFAULT: "bigscience/bloom-7b1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloom-7b1",
        },
    },
)


register_model_group(
    models={
        "BLOOMZ-560M": {
            DownloadSource.DEFAULT: "bigscience/bloomz-560m",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloomz-560m",
        },
        "BLOOMZ-3B": {
            DownloadSource.DEFAULT: "bigscience/bloomz-3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloomz-3b",
        },
        "BLOOMZ-7B1-mt": {
            DownloadSource.DEFAULT: "bigscience/bloomz-7b1-mt",
            DownloadSource.MODELSCOPE: "AI-ModelScope/bloomz-7b1-mt",
        },
    },
)


register_model_group(
    models={
        "BlueLM-7B-Base": {
            DownloadSource.DEFAULT: "vivo-ai/BlueLM-7B-Base",
            DownloadSource.MODELSCOPE: "vivo-ai/BlueLM-7B-Base",
        },
        "BlueLM-7B-Chat": {
            DownloadSource.DEFAULT: "vivo-ai/BlueLM-7B-Chat",
            DownloadSource.MODELSCOPE: "vivo-ai/BlueLM-7B-Chat",
        },
    },
    template="bluelm",
)


register_model_group(
    models={
        "Breeze-7B": {
            DownloadSource.DEFAULT: "MediaTek-Research/Breeze-7B-Base-v1_0",
        },
        "Breeze-7B-Instruct": {
            DownloadSource.DEFAULT: "MediaTek-Research/Breeze-7B-Instruct-v1_0",
        },
    },
    template="breeze",
)


register_model_group(
    models={
        "ChatGLM2-6B-Chat": {
            DownloadSource.DEFAULT: "zai-org/chatglm2-6b",
            DownloadSource.MODELSCOPE: "ZhipuAI/chatglm2-6b",
        }
    },
    template="chatglm2",
)


register_model_group(
    models={
        "ChatGLM3-6B-Base": {
            DownloadSource.DEFAULT: "zai-org/chatglm3-6b-base",
            DownloadSource.MODELSCOPE: "ZhipuAI/chatglm3-6b-base",
        },
        "ChatGLM3-6B-Chat": {
            DownloadSource.DEFAULT: "zai-org/chatglm3-6b",
            DownloadSource.MODELSCOPE: "ZhipuAI/chatglm3-6b",
        },
    },
    template="chatglm3",
)


register_model_group(
    models={
        "Chinese-Llama-2-1.3B": {
            DownloadSource.DEFAULT: "hfl/chinese-llama-2-1.3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-llama-2-1.3b",
        },
        "Chinese-Llama-2-7B": {
            DownloadSource.DEFAULT: "hfl/chinese-llama-2-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-llama-2-7b",
        },
        "Chinese-Llama-2-13B": {
            DownloadSource.DEFAULT: "hfl/chinese-llama-2-13b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-llama-2-13b",
        },
        "Chinese-Alpaca-2-1.3B-Chat": {
            DownloadSource.DEFAULT: "hfl/chinese-alpaca-2-1.3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-alpaca-2-1.3b",
        },
        "Chinese-Alpaca-2-7B-Chat": {
            DownloadSource.DEFAULT: "hfl/chinese-alpaca-2-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-alpaca-2-7b",
        },
        "Chinese-Alpaca-2-13B-Chat": {
            DownloadSource.DEFAULT: "hfl/chinese-alpaca-2-13b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/chinese-alpaca-2-13b",
        },
    },
    template="llama2_zh",
)


register_model_group(
    models={
        "CodeGeeX4-9B-Chat": {
            DownloadSource.DEFAULT: "zai-org/codegeex4-all-9b",
            DownloadSource.MODELSCOPE: "ZhipuAI/codegeex4-all-9b",
        },
    },
    template="codegeex4",
)


register_model_group(
    models={
        "CodeGemma-7B": {
            DownloadSource.DEFAULT: "google/codegemma-7b",
        },
        "CodeGemma-7B-Instruct": {
            DownloadSource.DEFAULT: "google/codegemma-7b-it",
            DownloadSource.MODELSCOPE: "AI-ModelScope/codegemma-7b-it",
        },
        "CodeGemma-1.1-2B": {
            DownloadSource.DEFAULT: "google/codegemma-1.1-2b",
        },
        "CodeGemma-1.1-7B-Instruct": {
            DownloadSource.DEFAULT: "google/codegemma-1.1-7b-it",
        },
    },
    template="gemma",
)


register_model_group(
    models={
        "Codestral-22B-v0.1-Chat": {
            DownloadSource.DEFAULT: "mistralai/Codestral-22B-v0.1",
            DownloadSource.MODELSCOPE: "swift/Codestral-22B-v0.1",
        },
    },
    template="mistral",
)


register_model_group(
    models={
        "CommandR-35B-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/c4ai-command-r-v01",
            DownloadSource.MODELSCOPE: "AI-ModelScope/c4ai-command-r-v01",
        },
        "CommandR-Plus-104B-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/c4ai-command-r-plus",
            DownloadSource.MODELSCOPE: "AI-ModelScope/c4ai-command-r-plus",
        },
        "CommandR-35B-4bit-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/c4ai-command-r-v01-4bit",
            DownloadSource.MODELSCOPE: "mirror013/c4ai-command-r-v01-4bit",
        },
        "CommandR-Plus-104B-4bit-Chat": {
            DownloadSource.DEFAULT: "CohereForAI/c4ai-command-r-plus-4bit",
        },
    },
    template="cohere",
)


register_model_group(
    models={
        "DBRX-132B-Base": {
            DownloadSource.DEFAULT: "databricks/dbrx-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/dbrx-base",
        },
        "DBRX-132B-Instruct": {
            DownloadSource.DEFAULT: "databricks/dbrx-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/dbrx-instruct",
        },
    },
    template="dbrx",
)


register_model_group(
    models={
        "DeepSeek-LLM-7B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-7b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-7b-base",
        },
        "DeepSeek-LLM-67B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-67b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-67b-base",
        },
        "DeepSeek-LLM-7B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-7b-chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-7b-chat",
        },
        "DeepSeek-LLM-67B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-llm-67b-chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-llm-67b-chat",
        },
        "DeepSeek-Math-7B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-math-7b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-math-7b-base",
        },
        "DeepSeek-Math-7B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-math-7b-instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-math-7b-instruct",
        },
        "DeepSeek-MoE-16B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-moe-16b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-moe-16b-base",
        },
        "DeepSeek-MoE-16B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-moe-16b-chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-moe-16b-chat",
        },
        "DeepSeek-V2-16B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2-Lite",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Lite",
        },
        "DeepSeek-V2-236B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2",
        },
        "DeepSeek-V2-16B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2-Lite-Chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Lite-Chat",
        },
        "DeepSeek-V2-236B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2-Chat",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Chat",
        },
        "DeepSeek-Coder-V2-16B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        },
        "DeepSeek-Coder-V2-236B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-Coder-V2-Base",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-Coder-V2-Base",
        },
        "DeepSeek-Coder-V2-16B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
        },
        "DeepSeek-Coder-V2-236B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-Coder-V2-Instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        },
    },
    template="deepseek",
)


register_model_group(
    models={
        "DeepSeek-Coder-6.7B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-6.7b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-6.7b-base",
        },
        "DeepSeek-Coder-7B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-7b-base-v1.5",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-7b-base-v1.5",
        },
        "DeepSeek-Coder-33B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-33b-base",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-33b-base",
        },
        "DeepSeek-Coder-6.7B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-6.7b-instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-6.7b-instruct",
        },
        "DeepSeek-Coder-7B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-7b-instruct-v1.5",
        },
        "DeepSeek-Coder-33B-Instruct": {
            DownloadSource.DEFAULT: "deepseek-ai/deepseek-coder-33b-instruct",
            DownloadSource.MODELSCOPE: "deepseek-ai/deepseek-coder-33b-instruct",
        },
    },
    template="deepseekcoder",
)


register_model_group(
    models={
        "DeepSeek-V2-0628-236B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2-Chat-0628",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2-Chat-0628",
        },
        "DeepSeek-V2.5-236B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2.5",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2.5",
        },
        "DeepSeek-V2.5-1210-236B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V2.5-1210",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V2.5-1210",
        },
        "DeepSeek-V3-671B-Base": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V3-Base",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V3-Base",
        },
        "DeepSeek-V3-671B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V3",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V3",
        },
        "DeepSeek-V3-0324-671B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-V3-0324",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-V3-0324",
        },
    },
    template="deepseek3",
)


register_model_group(
    models={
        "DeepSeek-R1-1.5B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        },
        "DeepSeek-R1-7B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        },
        "DeepSeek-R1-8B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        },
        "DeepSeek-R1-14B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        },
        "DeepSeek-R1-32B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        },
        "DeepSeek-R1-70B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        },
        "DeepSeek-R1-671B-Chat-Zero": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-Zero",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-Zero",
        },
        "DeepSeek-R1-671B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1",
        },
        "DeepSeek-R1-0528-8B-Distill": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
        },
        "DeepSeek-R1-0528-671B-Chat": {
            DownloadSource.DEFAULT: "deepseek-ai/DeepSeek-R1-0528",
            DownloadSource.MODELSCOPE: "deepseek-ai/DeepSeek-R1-0528",
        },
    },
    template="deepseekr1",
)


register_model_group(
    models={
        "Devstral-Small-2507-Instruct": {
            DownloadSource.DEFAULT: "mistralai/Devstral-Small-2507",
            DownloadSource.MODELSCOPE: "mistralai/Devstral-Small-2507",
        },
    },
    template="mistral_small",
)


register_model_group(
    models={
        "EXAONE-3.0-7.8B-Instruct": {
            DownloadSource.DEFAULT: "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
        },
    },
    template="exaone",
)


register_model_group(
    models={
        "Falcon-7B": {
            DownloadSource.DEFAULT: "tiiuae/falcon-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/falcon-7b",
        },
        "Falcon-11B": {
            DownloadSource.DEFAULT: "tiiuae/falcon-11B",
            DownloadSource.MODELSCOPE: "tiiuae/falcon-11B",
        },
        "Falcon-40B": {
            DownloadSource.DEFAULT: "tiiuae/falcon-40b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/falcon-40b",
        },
        "Falcon-180B": {
            DownloadSource.DEFAULT: "tiiuae/falcon-180b",
            DownloadSource.MODELSCOPE: "modelscope/falcon-180B",
        },
        "Falcon-7B-Instruct": {
            DownloadSource.DEFAULT: "tiiuae/falcon-7b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/falcon-7b-instruct",
        },
        "Falcon-40B-Instruct": {
            DownloadSource.DEFAULT: "tiiuae/falcon-40b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/falcon-40b-instruct",
        },
        "Falcon-180B-Chat": {
            DownloadSource.DEFAULT: "tiiuae/falcon-180b-chat",
            DownloadSource.MODELSCOPE: "modelscope/falcon-180B-chat",
        },
    },
    template="falcon",
)

register_model_group(
    models={
        "Falcon-H1-0.5B-Base": {
            DownloadSource.DEFAULT: "tiiuae/Falcon-H1-0.5B-Base",
            DownloadSource.MODELSCOPE: "tiiuae/Falcon-H1-0.5B-Base",
        },
        "Falcon-H1-1.5B-Base": {
            DownloadSource.DEFAULT: "tiiuae/Falcon-H1-1.5B-Base",
            DownloadSource.MODELSCOPE: "tiiuae/Falcon-H1-1.5B-Base",
        },
        "Falcon-H1-1.5B-Deep-Base": {
            DownloadSource.DEFAULT: "tiuae/Falcon-H1-1.5B-Deep-Base",
            DownloadSource.MODELSCOPE: "tiiuae/Falcon-H1-1.5B-Deep-Base",
        },
        "Falcon-H1-3B-Base": {
            DownloadSource.DEFAULT: "tiiuae/Falcon-H1-3B-Base",
            DownloadSource.MODELSCOPE: "tiiuae/Falcon-H1-3B-Base",
        },
        "Falcon-H1-7B-Base": {
            DownloadSource.DEFAULT: "tiiuae/Falcon-H1-7B-Base",
            DownloadSource.MODELSCOPE: "tiiuae/Falcon-H1-7B-Base",
        },
        "Falcon-H1-34B-Base": {
            DownloadSource.DEFAULT: "tiiuae/Falcon-H1-34B-Base",
            DownloadSource.MODELSCOPE: "tiiuae/Falcon-H1-34B-Base",
        },
        "Falcon-H1-0.5B-Instruct": {
            DownloadSource.DEFAULT: "tiiuae/Falcon-H1-0.5B-Instruct",
            DownloadSource.MODELSCOPE: "tiiuae/Falcon-H1-0.5B-Instruct",
        },
        "Falcon-H1-1.5B-Instruct": {
            DownloadSource.DEFAULT: "tiiuae/Falcon-H1-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "tiiuae/Falcon-H1-1.5B-Instruct",
        },
        "Falcon-H1-1.5B-Deep-Instruct": {
            DownloadSource.DEFAULT: "tiiuae/Falcon-H1-1.5B-Deep-Instruct",
            DownloadSource.MODELSCOPE: "tiiuae/Falcon-H1-1.5B-Deep-Instruct",
        },
        "Falcon-H1-3B-Instruct": {
            DownloadSource.DEFAULT: "tiiuae/Falcon-H1-3B-Instruct",
            DownloadSource.MODELSCOPE: "tiiuae/Falcon-H1-3B-Instruct",
        },
        "Falcon-H1-7B-Instruct": {
            DownloadSource.DEFAULT: "tiiuae/Falcon-H1-7B-Instruct",
            DownloadSource.MODELSCOPE: "tiiuae/Falcon-H1-7B-Instruct",
        },
        "Falcon-H1-34B-Instruct": {
            DownloadSource.DEFAULT: "tiiuae/Falcon-H1-34B-Instruct",
            DownloadSource.MODELSCOPE: "tiiuae/Falcon-H1-34B-Instruct",
        },
    },
    template="falcon_h1",
)


register_model_group(
    models={
        "Gemma-2B": {
            DownloadSource.DEFAULT: "google/gemma-2b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/gemma-2b",
        },
        "Gemma-7B": {
            DownloadSource.DEFAULT: "google/gemma-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/gemma-2b-it",
        },
        "Gemma-2B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-2b-it",
            DownloadSource.MODELSCOPE: "AI-ModelScope/gemma-7b",
        },
        "Gemma-7B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-7b-it",
            DownloadSource.MODELSCOPE: "AI-ModelScope/gemma-7b-it",
        },
        "Gemma-1.1-2B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-1.1-2b-it",
        },
        "Gemma-1.1-7B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-1.1-7b-it",
        },
    },
    template="gemma",
)


register_model_group(
    models={
        "Gemma-2-2B": {
            DownloadSource.DEFAULT: "google/gemma-2-2b",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-2-2b",
        },
        "Gemma-2-9B": {
            DownloadSource.DEFAULT: "google/gemma-2-9b",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-2-9b",
        },
        "Gemma-2-27B": {
            DownloadSource.DEFAULT: "google/gemma-2-27b",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-2-27b",
        },
        "Gemma-2-2B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-2-2b-it",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-2-2b-it",
            DownloadSource.OPENMIND: "LlamaFactory/gemma-2-2b-it",
        },
        "Gemma-2-9B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-2-9b-it",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-2-9b-it",
            DownloadSource.OPENMIND: "LlamaFactory/gemma-2-9b-it",
        },
        "Gemma-2-27B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-2-27b-it",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-2-27b-it",
        },
        "Gemma-3-1B": {
            DownloadSource.DEFAULT: "google/gemma-3-1b-pt",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-3-1b-pt",
        },
        "Gemma-3-1B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-3-1b-it",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-3-1b-it",
        },
        "MedGemma-27B-Instruct": {
            DownloadSource.DEFAULT: "google/medgemma-27b-text-it",
            DownloadSource.MODELSCOPE: "google/medgemma-27b-text-it",
        },
    },
    template="gemma2",
)


register_model_group(
    models={
        "Gemma-3-4B": {
            DownloadSource.DEFAULT: "google/gemma-3-4b-pt",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-3-4b-pt",
        },
        "Gemma-3-12B": {
            DownloadSource.DEFAULT: "google/gemma-3-12b-pt",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-3-12b-pt",
        },
        "Gemma-3-27B": {
            DownloadSource.DEFAULT: "google/gemma-3-27b-pt",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-3-27b-pt",
        },
        "Gemma-3-4B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-3-4b-it",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-3-4b-it",
        },
        "Gemma-3-12B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-3-12b-it",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-3-12b-it",
        },
        "Gemma-3-27B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-3-27b-it",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-3-27b-it",
        },
        "MedGemma-4B": {
            DownloadSource.DEFAULT: "google/medgemma-4b-pt",
            DownloadSource.MODELSCOPE: "google/medgemma-4b-pt",
        },
        "MedGemma-4B-Instruct": {
            DownloadSource.DEFAULT: "google/medgemma-4b-it",
            DownloadSource.MODELSCOPE: "google/medgemma-4b-it",
        },
    },
    template="gemma3",
    multimodal=True,
)


register_model_group(
    models={
        "Gemma-3n-E2B": {
            DownloadSource.DEFAULT: "google/gemma-3n-E2B",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-3n-E2B",
        },
        "Gemma-3n-E4B": {
            DownloadSource.DEFAULT: "google/gemma-3n-E4B",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-3n-E4B",
        },
        "Gemma-3n-E2B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-3n-E2B-it",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-3n-E2B-it",
        },
        "Gemma-3n-E4B-Instruct": {
            DownloadSource.DEFAULT: "google/gemma-3n-E4B-it",
            DownloadSource.MODELSCOPE: "LLM-Research/gemma-3n-E4B-it",
        },
    },
    template="gemma3n",
    multimodal=True,
)


register_model_group(
    models={
        "GLM-4-9B": {
            DownloadSource.DEFAULT: "zai-org/glm-4-9b",
            DownloadSource.MODELSCOPE: "ZhipuAI/glm-4-9b",
        },
        "GLM-4-9B-Chat": {
            DownloadSource.DEFAULT: "zai-org/glm-4-9b-chat",
            DownloadSource.MODELSCOPE: "ZhipuAI/glm-4-9b-chat",
            DownloadSource.OPENMIND: "LlamaFactory/glm-4-9b-chat",
        },
        "GLM-4-9B-1M-Chat": {
            DownloadSource.DEFAULT: "zai-org/glm-4-9b-chat-1m",
            DownloadSource.MODELSCOPE: "ZhipuAI/glm-4-9b-chat-1m",
        },
        "GLM-4-0414-9B-Chat": {
            DownloadSource.DEFAULT: "zai-org/GLM-4-9B-0414",
            DownloadSource.MODELSCOPE: "ZhipuAI/GLM-4-9B-0414",
        },
        "GLM-4-0414-32B-Base": {
            DownloadSource.DEFAULT: "zai-org/GLM-4-32B-Base-0414",
            DownloadSource.MODELSCOPE: "ZhipuAI/GLM-4-32B-Base-0414",
        },
        "GLM-4-0414-32B-Chat": {
            DownloadSource.DEFAULT: "zai-org/GLM-4-32B-0414",
            DownloadSource.MODELSCOPE: "ZhipuAI/GLM-4-32B-0414",
        },
    },
    template="glm4",
)


register_model_group(
    models={
        "GLM-4.1V-9B-Base": {
            DownloadSource.DEFAULT: "zai-org/GLM-4.1V-9B-Base",
            DownloadSource.MODELSCOPE: "ZhipuAI/GLM-4.1V-9B-Base",
        },
        "GLM-4.1V-9B-Thinking": {
            DownloadSource.DEFAULT: "zai-org/GLM-4.1V-9B-Thinking",
            DownloadSource.MODELSCOPE: "ZhipuAI/GLM-4.1V-9B-Thinking",
        },
    },
    template="glm4v",
    multimodal=True,
)


register_model_group(
    models={
        "GLM-4.5-Air-Base": {
            DownloadSource.DEFAULT: "zai-org/GLM-4.5-Air-Base",
            DownloadSource.MODELSCOPE: "ZhipuAI/GLM-4.5-Air-Base",
        },
        "GLM-4.5-Base": {
            DownloadSource.DEFAULT: "zai-org/GLM-4.5-Base",
            DownloadSource.MODELSCOPE: "ZhipuAI/GLM-4.5-Base",
        },
        "GLM-4.5-Air-Chat": {
            DownloadSource.DEFAULT: "zai-org/GLM-4.5-Air",
            DownloadSource.MODELSCOPE: "ZhipuAI/GLM-4.5-Air",
        },
        "GLM-4.5-Chat": {
            DownloadSource.DEFAULT: "zai-org/GLM-4.5",
            DownloadSource.MODELSCOPE: "ZhipuAI/GLM-4.5",
        },
    },
    template="glm4_moe",
)


register_model_group(
    models={
        "GLM-Z1-0414-9B-Chat": {
            DownloadSource.DEFAULT: "zai-org/GLM-Z1-9B-0414",
            DownloadSource.MODELSCOPE: "ZhipuAI/GLM-Z1-9B-0414",
        },
        "GLM-Z1-0414-32B-Chat": {
            DownloadSource.DEFAULT: "zai-org/GLM-Z1-32B-0414",
            DownloadSource.MODELSCOPE: "ZhipuAI/GLM-Z1-32B-0414",
        },
    },
    template="glmz1",
)


register_model_group(
    models={
        "GPT-2-Small": {
            DownloadSource.DEFAULT: "openai-community/gpt2",
            DownloadSource.MODELSCOPE: "AI-ModelScope/gpt2",
        },
        "GPT-2-Medium": {
            DownloadSource.DEFAULT: "openai-community/gpt2-medium",
            DownloadSource.MODELSCOPE: "AI-ModelScope/gpt2-medium",
        },
        "GPT-2-Large": {
            DownloadSource.DEFAULT: "openai-community/gpt2-large",
            DownloadSource.MODELSCOPE: "AI-ModelScope/gpt2-large",
        },
        "GPT-2-XL": {
            DownloadSource.DEFAULT: "openai-community/gpt2-xl",
            DownloadSource.MODELSCOPE: "goodbai95/GPT2-xl",
        },
    },
)


register_model_group(
    models={
        "Granite-3.0-1B-A400M-Base": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.0-1b-a400m-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.0-1b-a400m-base",
        },
        "Granite-3.0-3B-A800M-Base": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.0-3b-a800m-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.0-3b-a800m-base",
        },
        "Granite-3.0-2B-Base": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.0-2b-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.0-2b-base",
        },
        "Granite-3.0-8B-Base": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.0-8b-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.0-8b-base",
        },
        "Granite-3.0-1B-A400M-Instruct": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.0-1b-a400m-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.0-1b-a400m-instruct",
        },
        "Granite-3.0-3B-A800M-Instruct": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.0-3b-a800m-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.0-3b-a800m-instruct",
        },
        "Granite-3.0-2B-Instruct": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.0-2b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.0-2b-instruct",
        },
        "Granite-3.0-8B-Instruct": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.0-8b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.0-8b-instruct",
        },
        "Granite-3.1-1B-A400M-Base": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.1-1b-a400m-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.1-1b-a400m-base",
        },
        "Granite-3.1-3B-A800M-Base": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.1-3b-a800m-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.1-3b-a800m-base",
        },
        "Granite-3.1-2B-Base": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.1-2b-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.1-2b-base",
        },
        "Granite-3.1-8B-Base": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.1-8b-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.1-8b-base",
        },
        "Granite-3.1-1B-A400M-Instruct": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.1-1b-a400m-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.1-1b-a400m-instruct",
        },
        "Granite-3.1-3B-A800M-Instruct": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.1-3b-a800m-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.1-3b-a800m-instruct",
        },
        "Granite-3.1-2B-Instruct": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.1-2b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.1-2b-instruct",
        },
        "Granite-3.1-8B-Instruct": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.1-8b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.1-8b-instruct",
        },
        "Granite-3.2-2B-Instruct": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.2-2b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.2-2b-instruct",
        },
        "Granite-3.2-8B-Instruct": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.2-8b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.2-8b-instruct",
        },
        "Granite-3.3-2B-Base": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.3-2b-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.3-2b-base",
        },
        "Granite-3.3-8B-Base": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.3-8b-base",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.3-8b-base",
        },
        "Granite-3.3-2B-Instruct": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.3-2b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.3-2b-instruct",
        },
        "Granite-3.3-8B-Instruct": {
            DownloadSource.DEFAULT: "ibm-granite/granite-3.3-8b-instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-3.3-8b-instruct",
        },
    },
    template="granite3",
)


register_model_group(
    models={
        "Granite-Vision-3.2-2B": {
            DownloadSource.DEFAULT: "ibm-granite/granite-vision-3.2-2b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/granite-vision-3.2-2b",
        },
    },
    template="granite3_vision",
    multimodal=True,
)


register_model_group(
    models={
        "Granite-4.0-tiny-preview": {
            DownloadSource.DEFAULT: "ibm-granite/granite-4.0-tiny-preview",
            DownloadSource.MODELSCOPE: "ibm-granite/granite-4.0-tiny-preview",
        },
    },
    template="granite4",
)


register_model_group(
    models={
        "Hunyuan-7B-Instruct": {
            DownloadSource.DEFAULT: "tencent/Hunyuan-7B-Instruct",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Hunyuan-7B-Instruct",
        },
    },
    template="hunyuan",
)


register_model_group(
    models={
        "Index-1.9B-Base": {
            DownloadSource.DEFAULT: "IndexTeam/Index-1.9B",
            DownloadSource.MODELSCOPE: "IndexTeam/Index-1.9B",
        },
        "Index-1.9B-Base-Pure": {
            DownloadSource.DEFAULT: "IndexTeam/Index-1.9B-Pure",
            DownloadSource.MODELSCOPE: "IndexTeam/Index-1.9B-Pure",
        },
        "Index-1.9B-Chat": {
            DownloadSource.DEFAULT: "IndexTeam/Index-1.9B-Chat",
            DownloadSource.MODELSCOPE: "IndexTeam/Index-1.9B-Chat",
        },
        "Index-1.9B-Character-Chat": {
            DownloadSource.DEFAULT: "IndexTeam/Index-1.9B-Character",
            DownloadSource.MODELSCOPE: "IndexTeam/Index-1.9B-Character",
        },
        "Index-1.9B-Chat-32K": {
            DownloadSource.DEFAULT: "IndexTeam/Index-1.9B-32K",
            DownloadSource.MODELSCOPE: "IndexTeam/Index-1.9B-32K",
        },
    },
    template="index",
)


register_model_group(
    models={
        "InternLM-7B": {
            DownloadSource.DEFAULT: "internlm/internlm-7b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm-7b",
        },
        "InternLM-20B": {
            DownloadSource.DEFAULT: "internlm/internlm-20b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm-20b",
        },
        "InternLM-7B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm-chat-7b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm-chat-7b",
        },
        "InternLM-20B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm-chat-20b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm-chat-20b",
        },
    },
    template="intern",
)


register_model_group(
    models={
        "InternLM2-7B": {
            DownloadSource.DEFAULT: "internlm/internlm2-7b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2-7b",
        },
        "InternLM2-20B": {
            DownloadSource.DEFAULT: "internlm/internlm2-20b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2-20b",
        },
        "InternLM2-7B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm2-chat-7b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2-chat-7b",
        },
        "InternLM2-20B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm2-chat-20b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2-chat-20b",
        },
        "InternLM2.5-1.8B": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-1_8b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-1_8b",
            DownloadSource.OPENMIND: "Intern/internlm2_5-1_8b",
        },
        "InternLM2.5-7B": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-7b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-7b",
        },
        "InternLM2.5-20B": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-20b",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-20b",
            DownloadSource.OPENMIND: "Intern/internlm2_5-20b",
        },
        "InternLM2.5-1.8B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-1_8b-chat",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-1_8b-chat",
            DownloadSource.OPENMIND: "Intern/internlm2_5-1_8b-chat",
        },
        "InternLM2.5-7B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-7b-chat",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-7b-chat",
            DownloadSource.OPENMIND: "Intern/internlm2_5-7b-chat",
        },
        "InternLM2.5-7B-1M-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-7b-chat-1m",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-7b-chat-1m",
            DownloadSource.OPENMIND: "Intern/internlm2_5-7b-chat-1m",
        },
        "InternLM2.5-20B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm2_5-20b-chat",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm2_5-20b-chat",
            DownloadSource.OPENMIND: "Intern/internlm2_5-20b-chat",
        },
        "InternLM3-8B-Chat": {
            DownloadSource.DEFAULT: "internlm/internlm3-8b-instruct",
            DownloadSource.MODELSCOPE: "Shanghai_AI_Laboratory/internlm3-8b-instruct",
        },
    },
    template="intern2",
)


register_model_group(
    models={
        "InternVL2.5-2B-MPO": {
            DownloadSource.DEFAULT: "OpenGVLab/InternVL2_5-2B-MPO-hf",
            DownloadSource.MODELSCOPE: "OpenGVLab/InternVL2_5-2B-MPO-hf",
        },
        "InternVL2.5-8B-MPO": {
            DownloadSource.DEFAULT: "OpenGVLab/InternVL2_5-8B-MPO-hf",
            DownloadSource.MODELSCOPE: "OpenGVLab/InternVL2_5-8B-MPO-hf",
        },
        "InternVL3-1B-hf": {
            DownloadSource.DEFAULT: "OpenGVLab/InternVL3-1B-hf",
            DownloadSource.MODELSCOPE: "OpenGVLab/InternVL3-1B-hf",
        },
        "InternVL3-2B-hf": {
            DownloadSource.DEFAULT: "OpenGVLab/InternVL3-2B-hf",
            DownloadSource.MODELSCOPE: "OpenGVLab/InternVL3-2B-hf",
        },
        "InternVL3-8B-hf": {
            DownloadSource.DEFAULT: "OpenGVLab/InternVL3-8B-hf",
            DownloadSource.MODELSCOPE: "OpenGVLab/InternVL3-8B-hf",
        },
        "InternVL3-14B-hf": {
            DownloadSource.DEFAULT: "OpenGVLab/InternVL3-14B-hf",
            DownloadSource.MODELSCOPE: "OpenGVLab/InternVL3-14B-hf",
        },
        "InternVL3-38B-hf": {
            DownloadSource.DEFAULT: "OpenGVLab/InternVL3-38B-hf",
            DownloadSource.MODELSCOPE: "OpenGVLab/InternVL3-38B-hf",
        },
        "InternVL3-78B-hf": {
            DownloadSource.DEFAULT: "OpenGVLab/InternVL3-78B-hf",
            DownloadSource.MODELSCOPE: "OpenGVLab/InternVL3-78B-hf",
        },
    },
    template="intern_vl",
    multimodal=True,
)


register_model_group(
    models={
        "Jamba-v0.1": {
            DownloadSource.DEFAULT: "ai21labs/Jamba-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Jamba-v0.1",
        }
    },
)


register_model_group(
    models={
        "Keye-VL-8B-Chat": {
            DownloadSource.DEFAULT: "Kwai-Keye/Keye-VL-8B-Preview",
            DownloadSource.MODELSCOPE: "Kwai-Keye/Keye-VL-8B-Preview",
        },
    },
    template="keye_vl",
    multimodal=True,
)


register_model_group(
    models={
        "Kimi-Dev-72B-Instruct": {
            DownloadSource.DEFAULT: "moonshotai/Kimi-Dev-72B",
            DownloadSource.MODELSCOPE: "moonshotai/Kimi-Dev-72B",
        },
    },
    template="qwen",
)


register_model_group(
    models={
        "Kimi-VL-A3B-Instruct": {
            DownloadSource.DEFAULT: "moonshotai/Kimi-VL-A3B-Instruct",
            DownloadSource.MODELSCOPE: "moonshotai/Kimi-VL-A3B-Instruct",
        },
        "Kimi-VL-A3B-Thinking": {
            DownloadSource.DEFAULT: "moonshotai/Kimi-VL-A3B-Thinking",
            DownloadSource.MODELSCOPE: "moonshotai/Kimi-VL-A3B-Thinking",
        },
        "Kimi-VL-A3B-Thinking-2506": {
            DownloadSource.DEFAULT: "moonshotai/Kimi-VL-A3B-Thinking-2506",
            DownloadSource.MODELSCOPE: "moonshotai/Kimi-VL-A3B-Thinking-2506",
        },
    },
    template="kimi_vl",
    multimodal=True,
)


register_model_group(
    models={
        "LingoWhale-8B": {
            DownloadSource.DEFAULT: "deeplang-ai/LingoWhale-8B",
            DownloadSource.MODELSCOPE: "DeepLang/LingoWhale-8B",
        }
    },
)


register_model_group(
    models={
        "Llama-7B": {
            DownloadSource.DEFAULT: "huggyllama/llama-7b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-7b",
        },
        "Llama-13B": {
            DownloadSource.DEFAULT: "huggyllama/llama-13b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-13b",
        },
        "Llama-30B": {
            DownloadSource.DEFAULT: "huggyllama/llama-30b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-30b",
        },
        "Llama-65B": {
            DownloadSource.DEFAULT: "huggyllama/llama-65b",
            DownloadSource.MODELSCOPE: "skyline2006/llama-65b",
        },
    }
)


register_model_group(
    models={
        "Llama-2-7B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-7b-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-7b-ms",
        },
        "Llama-2-13B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-13b-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-13b-ms",
        },
        "Llama-2-70B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-70b-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-70b-ms",
        },
        "Llama-2-7B-Chat": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-7b-chat-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-7b-chat-ms",
        },
        "Llama-2-13B-Chat": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-13b-chat-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-13b-chat-ms",
        },
        "Llama-2-70B-Chat": {
            DownloadSource.DEFAULT: "meta-llama/Llama-2-70b-chat-hf",
            DownloadSource.MODELSCOPE: "modelscope/Llama-2-70b-chat-ms",
        },
    },
    template="llama2",
)


register_model_group(
    models={
        "Llama-3-8B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-8B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-8B",
        },
        "Llama-3-70B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-70B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-70B",
        },
        "Llama-3-8B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-8B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-8B-Instruct",
        },
        "Llama-3-70B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3-70B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3-70B-Instruct",
        },
        "Llama-3-8B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3-8B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama3-8B-Chinese-Chat",
            DownloadSource.OPENMIND: "LlamaFactory/Llama3-Chinese-8B-Instruct",
        },
        "Llama-3-70B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3-70B-Chinese-Chat",
        },
        "Llama-3.1-8B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-8B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-8B",
        },
        "Llama-3.1-70B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-70B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-70B",
        },
        "Llama-3.1-405B": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-405B",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-405B",
        },
        "Llama-3.1-8B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-8B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-8B-Instruct",
        },
        "Llama-3.1-70B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-70B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-70B-Instruct",
        },
        "Llama-3.1-405B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Meta-Llama-3.1-405B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Meta-Llama-3.1-405B-Instruct",
        },
        "Llama-3.1-8B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3.1-8B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "XD_AI/Llama3.1-8B-Chinese-Chat",
        },
        "Llama-3.1-70B-Chinese-Chat": {
            DownloadSource.DEFAULT: "shenzhi-wang/Llama3.1-70B-Chinese-Chat",
            DownloadSource.MODELSCOPE: "XD_AI/Llama3.1-70B-Chinese-Chat",
        },
        "Llama-3.2-1B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-1B",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-1B",
        },
        "Llama-3.2-3B": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-3B",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-3B",
        },
        "Llama-3.2-1B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-1B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-1B-Instruct",
        },
        "Llama-3.2-3B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-3B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-3B-Instruct",
        },
        "Llama-3.3-70B-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.3-70B-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.3-70B-Instruct",
        },
    },
    template="llama3",
)


register_model_group(
    models={
        "Llama-3.2-11B-Vision": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-11B-Vision",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-11B-Vision",
        },
        "Llama-3.2-11B-Vision-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-11B-Vision-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-11B-Vision-Instruct",
        },
        "Llama-3.2-90B-Vision": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-90B-Vision",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-90B-Vision",
        },
        "Llama-3.2-90B-Vision-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-3.2-90B-Vision-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-3.2-90B-Vision-Instruct",
        },
    },
    template="mllama",
    multimodal=True,
)


register_model_group(
    models={
        "Llama-4-Scout-17B-16E": {
            DownloadSource.DEFAULT: "meta-llama/Llama-4-Scout-17B-16E",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-4-Scout-17B-16E",
        },
        "Llama-4-Scout-17B-16E-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-4-Scout-17B-16E-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-4-Scout-17B-16E-Instruct",
        },
        "Llama-4-Maverick-17B-128E": {
            DownloadSource.DEFAULT: "meta-llama/Llama-4-Maverick-17B-128E",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-4-Maverick-17B-128E",
        },
        "Llama-4-Maverick-17B-128E-Instruct": {
            DownloadSource.DEFAULT: "meta-llama/Llama-4-Maverick-17B-128E-Instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Llama-4-Maverick-17B-128E-Instruct",
        },
    },
    template="llama4",
    multimodal=True,
)


register_model_group(
    models={
        "LLaVA-1.5-7B-Chat": {
            DownloadSource.DEFAULT: "llava-hf/llava-1.5-7b-hf",
            DownloadSource.MODELSCOPE: "swift/llava-1.5-7b-hf",
        },
        "LLaVA-1.5-13B-Chat": {
            DownloadSource.DEFAULT: "llava-hf/llava-1.5-13b-hf",
            DownloadSource.MODELSCOPE: "swift/llava-1.5-13b-hf",
        },
    },
    template="llava",
    multimodal=True,
)


register_model_group(
    models={
        "LLaVA-NeXT-7B-Chat": {
            DownloadSource.DEFAULT: "llava-hf/llava-v1.6-vicuna-7b-hf",
            DownloadSource.MODELSCOPE: "swift/llava-v1.6-vicuna-7b-hf",
        },
        "LLaVA-NeXT-13B-Chat": {
            DownloadSource.DEFAULT: "llava-hf/llava-v1.6-vicuna-13b-hf",
            DownloadSource.MODELSCOPE: "swift/llava-v1.6-vicuna-13b-hf",
        },
    },
    template="llava_next",
    multimodal=True,
)


register_model_group(
    models={
        "LLaVA-NeXT-Mistral-7B-Chat": {
            DownloadSource.DEFAULT: "llava-hf/llava-v1.6-mistral-7b-hf",
            DownloadSource.MODELSCOPE: "swift/llava-v1.6-mistral-7b-hf",
        },
    },
    template="llava_next_mistral",
    multimodal=True,
)


register_model_group(
    models={
        "LLaVA-NeXT-Llama3-8B-Chat": {
            DownloadSource.DEFAULT: "llava-hf/llama3-llava-next-8b-hf",
            DownloadSource.MODELSCOPE: "swift/llama3-llava-next-8b-hf",
        },
    },
    template="llava_next_llama3",
    multimodal=True,
)


register_model_group(
    models={
        "LLaVA-NeXT-34B-Chat": {
            DownloadSource.DEFAULT: "llava-hf/llava-v1.6-34b-hf",
            DownloadSource.MODELSCOPE: "LLM-Research/llava-v1.6-34b-hf",
        },
    },
    template="llava_next_yi",
    multimodal=True,
)


register_model_group(
    models={
        "LLaVA-NeXT-72B-Chat": {
            DownloadSource.DEFAULT: "llava-hf/llava-next-72b-hf",
            DownloadSource.MODELSCOPE: "AI-ModelScope/llava-next-72b-hf",
        },
        "LLaVA-NeXT-110B-Chat": {
            DownloadSource.DEFAULT: "llava-hf/llava-next-110b-hf",
            DownloadSource.MODELSCOPE: "AI-ModelScope/llava-next-110b-hf",
        },
    },
    template="llava_next_qwen",
    multimodal=True,
)


register_model_group(
    models={
        "LLaVA-NeXT-Video-7B-Chat": {
            DownloadSource.DEFAULT: "llava-hf/LLaVA-NeXT-Video-7B-hf",
            DownloadSource.MODELSCOPE: "swift/LLaVA-NeXT-Video-7B-hf",
        },
        "LLaVA-NeXT-Video-7B-DPO-Chat": {
            DownloadSource.DEFAULT: "llava-hf/LLaVA-NeXT-Video-7B-DPO-hf",
            DownloadSource.MODELSCOPE: "swift/LLaVA-NeXT-Video-7B-DPO-hf",
        },
    },
    template="llava_next_video",
    multimodal=True,
)


register_model_group(
    models={
        "LLaVA-NeXT-Video-7B-32k-Chat": {
            DownloadSource.DEFAULT: "llava-hf/LLaVA-NeXT-Video-7B-32K-hf",
            DownloadSource.MODELSCOPE: "swift/LLaVA-NeXT-Video-7B-32K-hf",
        },
    },
    template="llava_next_video_mistral",
    multimodal=True,
)


register_model_group(
    models={
        "LLaVA-NeXT-Video-34B-Chat": {
            DownloadSource.DEFAULT: "llava-hf/LLaVA-NeXT-Video-34B-hf",
            DownloadSource.MODELSCOPE: "swift/LLaVA-NeXT-Video-34B-hf",
        },
        "LLaVA-NeXT-Video-34B-DPO-Chat": {
            DownloadSource.DEFAULT: "llava-hf/LLaVA-NeXT-Video-34B-DPO-hf",
        },
    },
    template="llava_next_video_yi",
    multimodal=True,
)


register_model_group(
    models={
        "Marco-o1-Chat": {
            DownloadSource.DEFAULT: "AIDC-AI/Marco-o1",
            DownloadSource.MODELSCOPE: "AIDC-AI/Marco-o1",
        },
    },
    template="marco",
)


register_model_group(
    models={
        "MiMo-7B-Base": {
            DownloadSource.DEFAULT: "XiaomiMiMo/MiMo-7B-Base",
            DownloadSource.MODELSCOPE: "XiaomiMiMo/MiMo-7B-Base",
        },
        "MiMo-7B-Instruct": {
            DownloadSource.DEFAULT: "XiaomiMiMo/MiMo-7B-SFT",
            DownloadSource.MODELSCOPE: "XiaomiMiMo/MiMo-7B-SFT",
        },
        "MiMo-7B-Instruct-RL": {
            DownloadSource.DEFAULT: "XiaomiMiMo/MiMo-7B-RL",
            DownloadSource.MODELSCOPE: "XiaomiMiMo/MiMo-7B-RL",
        },
        "MiMo-7B-RL-ZERO": {
            DownloadSource.DEFAULT: "XiaomiMiMo/MiMo-7B-RL-ZERO",
            DownloadSource.MODELSCOPE: "XiaomiMiMo/MiMo-7B-RL-ZERO",
        },
    },
    template="mimo",
)


register_model_group(
    models={
        "MiMo-7B-VL-Instruct": {
            DownloadSource.DEFAULT: "XiaomiMiMo/MiMo-VL-7B-SFT",
            DownloadSource.MODELSCOPE: "XiaomiMiMo/MiMo-VL-7B-SFT",
        },
        "MiMo-7B-VL-RL": {
            DownloadSource.DEFAULT: "XiaomiMiMo/MiMo-VL-7B-RL",
            DownloadSource.MODELSCOPE: "XiaomiMiMo/MiMo-VL-7B-RL",
        },
    },
    template="mimo_vl",
    multimodal=True,
)


register_model_group(
    models={
        "MiniCPM-2B-SFT-Chat": {
            DownloadSource.DEFAULT: "openbmb/MiniCPM-2B-sft-bf16",
            DownloadSource.MODELSCOPE: "OpenBMB/miniCPM-bf16",
        },
        "MiniCPM-2B-DPO-Chat": {
            DownloadSource.DEFAULT: "openbmb/MiniCPM-2B-dpo-bf16",
            DownloadSource.MODELSCOPE: "OpenBMB/MiniCPM-2B-dpo-bf16",
        },
    },
    template="cpm",
)


register_model_group(
    models={
        "MiniCPM3-4B-Chat": {
            DownloadSource.DEFAULT: "openbmb/MiniCPM3-4B",
            DownloadSource.MODELSCOPE: "OpenBMB/MiniCPM3-4B",
            DownloadSource.OPENMIND: "LlamaFactory/MiniCPM3-4B",
        },
    },
    template="cpm3",
)


register_model_group(
    models={
        "MiniCPM4-0.5B-Chat": {
            DownloadSource.DEFAULT: "openbmb/MiniCPM4-0.5B",
            DownloadSource.MODELSCOPE: "OpenBMB/MiniCPM4-0.5B",
        },
        "MiniCPM4-8B-Chat": {
            DownloadSource.DEFAULT: "openbmb/MiniCPM4-8B",
            DownloadSource.MODELSCOPE: "OpenBMB/MiniCPM4-8B",
        },
    },
    template="cpm4",
)


register_model_group(
    models={
        "MiniCPM-o-2_6": {
            DownloadSource.DEFAULT: "openbmb/MiniCPM-o-2_6",
            DownloadSource.MODELSCOPE: "OpenBMB/MiniCPM-o-2_6",
        },
    },
    template="minicpm_o",
    multimodal=True,
)


register_model_group(
    models={
        "MiniCPM-V-2_6": {
            DownloadSource.DEFAULT: "openbmb/MiniCPM-V-2_6",
            DownloadSource.MODELSCOPE: "OpenBMB/MiniCPM-V-2_6",
        },
    },
    template="minicpm_v",
    multimodal=True,
)


register_model_group(
    models={
        "Ministral-8B-Instruct-2410": {
            DownloadSource.DEFAULT: "mistralai/Ministral-8B-Instruct-2410",
            DownloadSource.MODELSCOPE: "mistralai/Ministral-8B-Instruct-2410",
        },
        "Mistral-Nemo-Base-2407": {
            DownloadSource.DEFAULT: "mistralai/Mistral-Nemo-Base-2407",
            DownloadSource.MODELSCOPE: "LLM-Research/Mistral-Nemo-Base-2407",
        },
        "Mistral-Nemo-Instruct-2407": {
            DownloadSource.DEFAULT: "mistralai/Mistral-Nemo-Instruct-2407",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-Nemo-Instruct-2407",
        },
    },
    template="ministral",
)


register_model_group(
    models={
        "Mistral-7B-v0.1": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-v0.1",
        },
        "Mistral-7B-v0.2": {
            DownloadSource.DEFAULT: "alpindale/Mistral-7B-v0.2-hf",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-v0.2-hf",
        },
        "Mistral-7B-v0.3": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-v0.3",
            DownloadSource.MODELSCOPE: "LLM-Research/mistral-7b-v0.3",
        },
        "Mistral-7B-Instruct-v0.1": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-Instruct-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-Instruct-v0.1",
        },
        "Mistral-7B-Instruct-v0.2": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-Instruct-v0.2",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mistral-7B-Instruct-v0.2",
        },
        "Mistral-7B-Instruct-v0.3": {
            DownloadSource.DEFAULT: "mistralai/Mistral-7B-Instruct-v0.3",
            DownloadSource.MODELSCOPE: "LLM-Research/Mistral-7B-Instruct-v0.3",
        },
    },
    template="mistral",
)


register_model_group(
    models={
        "Mistral-Small-24B-Base-2501": {
            DownloadSource.DEFAULT: "mistralai/Mistral-Small-24B-Base-2501",
            DownloadSource.MODELSCOPE: "mistralai/Mistral-Small-24B-Base-2501",
        },
        "Mistral-Small-24B-Instruct-2501": {
            DownloadSource.DEFAULT: "mistralai/Mistral-Small-24B-Instruct-2501",
            DownloadSource.MODELSCOPE: "mistralai/Mistral-Small-24B-Instruct-2501",
        },
    },
    template="mistral_small",
)


register_model_group(
    models={
        "Mistral-Small-3.1-24B-Base": {
            DownloadSource.DEFAULT: "mistralai/Mistral-Small-3.1-24B-Base-2503",
            DownloadSource.MODELSCOPE: "mistralai/Mistral-Small-3.1-24B-Base-2503",
        },
        "Mistral-Small-3.1-24B-Instruct": {
            DownloadSource.DEFAULT: "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
            DownloadSource.MODELSCOPE: "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        },
        "Mistral-Small-3.2-24B-Instruct": {
            DownloadSource.DEFAULT: "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            DownloadSource.MODELSCOPE: "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        },
    },
    template="mistral_small",
    multimodal=True,
)


register_model_group(
    models={
        "Mixtral-8x7B-v0.1": {
            DownloadSource.DEFAULT: "mistralai/Mixtral-8x7B-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mixtral-8x7B-v0.1",
        },
        "Mixtral-8x22B-v0.1": {
            DownloadSource.DEFAULT: "mistralai/Mixtral-8x22B-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mixtral-8x22B-v0.1",
        },
        "Mixtral-8x7B-v0.1-Instruct": {
            DownloadSource.DEFAULT: "mistralai/Mixtral-8x7B-Instruct-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mixtral-8x7B-Instruct-v0.1",
        },
        "Mixtral-8x22B-v0.1-Instruct": {
            DownloadSource.DEFAULT: "mistralai/Mixtral-8x22B-Instruct-v0.1",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Mixtral-8x22B-Instruct-v0.1",
        },
    },
    template="mistral",
)


register_model_group(
    models={
        "Moonlight-16B-A3B": {
            DownloadSource.DEFAULT: "moonshotai/Moonlight-16B-A3B",
            DownloadSource.MODELSCOPE: "moonshotai/Moonlight-16B-A3B",
        },
        "Moonlight-16B-A3B-Instruct": {
            DownloadSource.DEFAULT: "moonshotai/Moonlight-16B-A3B-Instruct",
            DownloadSource.MODELSCOPE: "moonshotai/Moonlight-16B-A3B-Instruct",
        },
    },
    template="moonlight",
)


register_model_group(
    models={
        "OLMo-1B": {
            DownloadSource.DEFAULT: "allenai/OLMo-1B-hf",
        },
        "OLMo-7B": {
            DownloadSource.DEFAULT: "allenai/OLMo-7B-hf",
        },
        "OLMo-7B-Chat": {
            DownloadSource.DEFAULT: "ssec-uw/OLMo-7B-Instruct-hf",
        },
        "OLMo-1.7-7B": {
            DownloadSource.DEFAULT: "allenai/OLMo-1.7-7B-hf",
        },
    },
)


register_model_group(
    models={
        "OpenChat3.5-7B-Chat": {
            DownloadSource.DEFAULT: "openchat/openchat-3.5-0106",
            DownloadSource.MODELSCOPE: "xcwzxcwz/openchat-3.5-0106",
        }
    },
    template="openchat",
)


register_model_group(
    models={
        "OpenChat3.6-8B-Chat": {
            DownloadSource.DEFAULT: "openchat/openchat-3.6-8b-20240522",
        }
    },
    template="openchat-3.6",
)


register_model_group(
    models={
        "OpenCoder-1.5B-Base": {
            DownloadSource.DEFAULT: "infly/OpenCoder-1.5B-Base",
            DownloadSource.MODELSCOPE: "infly/OpenCoder-1.5B-Base",
        },
        "OpenCoder-8B-Base": {
            DownloadSource.DEFAULT: "infly/OpenCoder-8B-Base",
            DownloadSource.MODELSCOPE: "infly/OpenCoder-8B-Base",
        },
        "OpenCoder-1.5B-Instruct": {
            DownloadSource.DEFAULT: "infly/OpenCoder-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "infly/OpenCoder-1.5B-Instruct",
        },
        "OpenCoder-8B-Instruct": {
            DownloadSource.DEFAULT: "infly/OpenCoder-8B-Instruct",
            DownloadSource.MODELSCOPE: "infly/OpenCoder-8B-Instruct",
        },
    },
    template="opencoder",
)


register_model_group(
    models={
        "Orion-14B-Base": {
            DownloadSource.DEFAULT: "OrionStarAI/Orion-14B-Base",
            DownloadSource.MODELSCOPE: "OrionStarAI/Orion-14B-Base",
        },
        "Orion-14B-Chat": {
            DownloadSource.DEFAULT: "OrionStarAI/Orion-14B-Chat",
            DownloadSource.MODELSCOPE: "OrionStarAI/Orion-14B-Chat",
        },
        "Orion-14B-Long-Chat": {
            DownloadSource.DEFAULT: "OrionStarAI/Orion-14B-LongChat",
            DownloadSource.MODELSCOPE: "OrionStarAI/Orion-14B-LongChat",
        },
        "Orion-14B-RAG-Chat": {
            DownloadSource.DEFAULT: "OrionStarAI/Orion-14B-Chat-RAG",
            DownloadSource.MODELSCOPE: "OrionStarAI/Orion-14B-Chat-RAG",
        },
        "Orion-14B-Plugin-Chat": {
            DownloadSource.DEFAULT: "OrionStarAI/Orion-14B-Chat-Plugin",
            DownloadSource.MODELSCOPE: "OrionStarAI/Orion-14B-Chat-Plugin",
        },
    },
    template="orion",
)


register_model_group(
    models={
        "PaliGemma-3B-pt-224": {
            DownloadSource.DEFAULT: "google/paligemma-3b-pt-224",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma-3b-pt-224",
        },
        "PaliGemma-3B-pt-448": {
            DownloadSource.DEFAULT: "google/paligemma-3b-pt-448",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma-3b-pt-448",
        },
        "PaliGemma-3B-pt-896": {
            DownloadSource.DEFAULT: "google/paligemma-3b-pt-896",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma-3b-pt-896",
        },
        "PaliGemma-3B-mix-224": {
            DownloadSource.DEFAULT: "google/paligemma-3b-mix-224",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma-3b-mix-224",
        },
        "PaliGemma-3B-mix-448": {
            DownloadSource.DEFAULT: "google/paligemma-3b-mix-448",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma-3b-mix-448",
        },
    },
    template="paligemma",
    multimodal=True,
)


register_model_group(
    models={
        "PaliGemma2-3B-pt-224": {
            DownloadSource.DEFAULT: "google/paligemma2-3b-pt-224",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma2-3b-pt-224",
        },
        "PaliGemma2-3B-pt-448": {
            DownloadSource.DEFAULT: "google/paligemma2-3b-pt-448",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma2-3b-pt-448",
        },
        "PaliGemma2-3B-pt-896": {
            DownloadSource.DEFAULT: "google/paligemma2-3b-pt-896",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma2-3b-pt-896",
        },
        "PaliGemma2-10B-pt-224": {
            DownloadSource.DEFAULT: "google/paligemma2-10b-pt-224",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma2-10b-pt-224",
        },
        "PaliGemma2-10B-pt-448": {
            DownloadSource.DEFAULT: "google/paligemma2-10b-pt-448",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma2-10b-pt-448",
        },
        "PaliGemma2-10B-pt-896": {
            DownloadSource.DEFAULT: "google/paligemma2-10b-pt-896",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma2-10b-pt-896",
        },
        "PaliGemma2-28B-pt-224": {
            DownloadSource.DEFAULT: "google/paligemma2-28b-pt-224",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma2-28b-pt-224",
        },
        "PaliGemma2-28B-pt-448": {
            DownloadSource.DEFAULT: "google/paligemma2-28b-pt-448",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma2-28b-pt-448",
        },
        "PaliGemma2-28B-pt-896": {
            DownloadSource.DEFAULT: "google/paligemma2-28b-pt-896",
            DownloadSource.MODELSCOPE: "AI-ModelScope/paligemma2-28b-pt-896",
        },
        "PaliGemma2-3B-mix-224": {
            DownloadSource.DEFAULT: "google/paligemma2-3b-mix-224",
            DownloadSource.MODELSCOPE: "mlx-community/paligemma2-3b-mix-224-bf16",
        },
        "PaliGemma2-3B-mix-448": {
            DownloadSource.DEFAULT: "google/paligemma2-3b-mix-448",
            DownloadSource.MODELSCOPE: "mlx-community/paligemma2-3b-mix-448-bf16",
        },
        "PaliGemma2-10B-mix-224": {
            DownloadSource.DEFAULT: "google/paligemma2-10b-mix-224",
            DownloadSource.MODELSCOPE: "mlx-community/paligemma2-10b-mix-224-bf16",
        },
        "PaliGemma2-10B-mix-448": {
            DownloadSource.DEFAULT: "google/paligemma2-10b-mix-448",
            DownloadSource.MODELSCOPE: "mlx-community/paligemma2-10b-mix-448-bf16",
        },
        "PaliGemma2-28B-mix-224": {
            DownloadSource.DEFAULT: "google/paligemma2-28b-mix-224",
            DownloadSource.MODELSCOPE: "mlx-community/paligemma2-28b-mix-224-bf16",
        },
        "PaliGemma2-28B-mix-448": {
            DownloadSource.DEFAULT: "google/paligemma2-28b-mix-448",
            DownloadSource.MODELSCOPE: "mlx-community/paligemma2-28b-mix-448-bf16",
        },
    },
    template="paligemma",
    multimodal=True,
)


register_model_group(
    models={
        "Phi-1.5-1.3B": {
            DownloadSource.DEFAULT: "microsoft/phi-1_5",
            DownloadSource.MODELSCOPE: "allspace/PHI_1-5",
        },
        "Phi-2-2.7B": {
            DownloadSource.DEFAULT: "microsoft/phi-2",
            DownloadSource.MODELSCOPE: "AI-ModelScope/phi-2",
        },
    }
)


register_model_group(
    models={
        "Phi-3-4B-4k-Instruct": {
            DownloadSource.DEFAULT: "microsoft/Phi-3-mini-4k-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3-mini-4k-instruct",
        },
        "Phi-3-4B-128k-Instruct": {
            DownloadSource.DEFAULT: "microsoft/Phi-3-mini-128k-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3-mini-128k-instruct",
        },
        "Phi-3-14B-8k-Instruct": {
            DownloadSource.DEFAULT: "microsoft/Phi-3-medium-4k-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3-medium-4k-instruct",
        },
        "Phi-3-14B-128k-Instruct": {
            DownloadSource.DEFAULT: "microsoft/Phi-3-medium-128k-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3-medium-128k-instruct",
        },
        "Phi-3.5-4B-instruct": {
            DownloadSource.DEFAULT: "microsoft/Phi-3.5-mini-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3.5-mini-instruct",
        },
        "Phi-3.5-MoE-42B-A6.6B-instruct": {
            DownloadSource.DEFAULT: "microsoft/Phi-3.5-MoE-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3.5-MoE-instruct",
        },
    },
    template="phi",
)


register_model_group(
    models={
        "Phi-3-7B-8k-Instruct": {
            DownloadSource.DEFAULT: "microsoft/Phi-3-small-8k-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3-small-8k-instruct",
        },
        "Phi-3-7B-128k-Instruct": {
            DownloadSource.DEFAULT: "microsoft/Phi-3-small-128k-instruct",
            DownloadSource.MODELSCOPE: "LLM-Research/Phi-3-small-128k-instruct",
        },
    },
    template="phi_small",
)


register_model_group(
    models={
        "Phi-4-14B-Instruct": {
            DownloadSource.DEFAULT: "microsoft/phi-4",
            DownloadSource.MODELSCOPE: "LLM-Research/phi-4",
        },
    },
    template="phi4",
)


register_model_group(
    models={
        "Pixtral-12B": {
            DownloadSource.DEFAULT: "mistral-community/pixtral-12b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/pixtral-12b",
        }
    },
    template="pixtral",
    multimodal=True,
)


register_model_group(
    models={
        "Qwen-1.8B": {
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-1_8B",
        },
        "Qwen-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-7B",
        },
        "Qwen-14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen-14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-14B",
        },
        "Qwen-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-72B",
        },
        "Qwen-1.8B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-1_8B-Chat",
        },
        "Qwen-7B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-7B-Chat",
        },
        "Qwen-14B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-14B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-14B-Chat",
        },
        "Qwen-72B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen-72B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-72B-Chat",
        },
        "Qwen-1.8B-Chat-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B-Chat-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-1_8B-Chat-Int8",
        },
        "Qwen-1.8B-Chat-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen-1_8B-Chat-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-1_8B-Chat-Int4",
        },
        "Qwen-7B-Chat-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-7B-Chat-Int8",
        },
        "Qwen-7B-Chat-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen-7B-Chat-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-7B-Chat-Int4",
        },
        "Qwen-14B-Chat-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen-14B-Chat-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-14B-Chat-Int8",
        },
        "Qwen-14B-Chat-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen-14B-Chat-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-14B-Chat-Int4",
        },
        "Qwen-72B-Chat-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen-72B-Chat-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-72B-Chat-Int8",
        },
        "Qwen-72B-Chat-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen-72B-Chat-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen-72B-Chat-Int4",
        },
    },
    template="qwen",
)


register_model_group(
    models={
        "Qwen1.5-0.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-0.5B",
        },
        "Qwen1.5-1.8B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-1.8B",
        },
        "Qwen1.5-4B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-4B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-4B",
        },
        "Qwen1.5-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-7B",
        },
        "Qwen1.5-14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-14B",
        },
        "Qwen1.5-32B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-32B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-32B",
        },
        "Qwen1.5-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-72B",
        },
        "Qwen1.5-110B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-110B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-110B",
        },
        "Qwen1.5-MoE-A2.7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-MoE-A2.7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-MoE-A2.7B",
        },
        "Qwen1.5-0.5B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-0.5B-Chat",
        },
        "Qwen1.5-1.8B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-1.8B-Chat",
        },
        "Qwen1.5-4B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-4B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-4B-Chat",
        },
        "Qwen1.5-7B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-7B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-7B-Chat",
        },
        "Qwen1.5-14B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-14B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-14B-Chat",
        },
        "Qwen1.5-32B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-32B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-32B-Chat",
        },
        "Qwen1.5-72B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-72B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-72B-Chat",
        },
        "Qwen1.5-110B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-110B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-110B-Chat",
        },
        "Qwen1.5-MoE-A2.7B-Chat": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-MoE-A2.7B-Chat",
        },
        "Qwen1.5-0.5B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-0.5B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-0.5B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-0.5B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-0.5B-Chat-AWQ",
        },
        "Qwen1.5-1.8B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-1.8B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-1.8B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-1.8B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-1.8B-Chat-AWQ",
        },
        "Qwen1.5-4B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-4B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-4B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-4B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-4B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-4B-Chat-AWQ",
        },
        "Qwen1.5-7B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-7B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-7B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-7B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-7B-Chat-AWQ",
        },
        "Qwen1.5-14B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-14B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-14B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-14B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-14B-Chat-AWQ",
        },
        "Qwen1.5-32B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-32B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-32B-Chat-AWQ",
        },
        "Qwen1.5-72B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-72B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-72B-Chat-GPTQ-Int8",
        },
        "Qwen1.5-72B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-72B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-72B-Chat-AWQ",
        },
        "Qwen1.5-110B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-110B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-110B-Chat-AWQ",
        },
        "Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen1.5-MoE-A2.7B-Chat-GPTQ-Int4",
        },
        "CodeQwen1.5-7B": {
            DownloadSource.DEFAULT: "Qwen/CodeQwen1.5-7B",
            DownloadSource.MODELSCOPE: "Qwen/CodeQwen1.5-7B",
        },
        "CodeQwen1.5-7B-Chat": {
            DownloadSource.DEFAULT: "Qwen/CodeQwen1.5-7B-Chat",
            DownloadSource.MODELSCOPE: "Qwen/CodeQwen1.5-7B-Chat",
        },
        "CodeQwen1.5-7B-Chat-AWQ": {
            DownloadSource.DEFAULT: "Qwen/CodeQwen1.5-7B-Chat-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/CodeQwen1.5-7B-Chat-AWQ",
        },
    },
    template="qwen",
)


register_model_group(
    models={
        "Qwen2-0.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-0.5B",
        },
        "Qwen2-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-1.5B",
        },
        "Qwen2-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-7B",
        },
        "Qwen2-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-72B",
        },
        "Qwen2-MoE-57B-A14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-57B-A14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-57B-A14B",
        },
        "Qwen2-0.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-0.5B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-0.5B-Instruct",
        },
        "Qwen2-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-1.5B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-1.5B-Instruct",
        },
        "Qwen2-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-7B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-7B-Instruct",
        },
        "Qwen2-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-72B-Instruct",
        },
        "Qwen2-MoE-57B-A14B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-57B-A14B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-57B-A14B-Instruct",
        },
        "Qwen2-0.5B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2-0.5B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-0.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2-0.5B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-0.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-0.5B-Instruct-AWQ",
        },
        "Qwen2-1.5B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2-1.5B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-1.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2-1.5B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-1.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-1.5B-Instruct-AWQ",
        },
        "Qwen2-7B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-7B-Instruct-GPTQ-Int8",
        },
        "Qwen2-7B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-7B-Instruct-GPTQ-Int4",
        },
        "Qwen2-7B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-7B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-7B-Instruct-AWQ",
        },
        "Qwen2-72B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-72B-Instruct-GPTQ-Int8",
        },
        "Qwen2-72B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-72B-Instruct-GPTQ-Int4",
        },
        "Qwen2-72B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-72B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-72B-Instruct-AWQ",
        },
        "Qwen2-57B-A14B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-57B-A14B-Instruct-GPTQ-Int4",
        },
        "Qwen2-Math-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-1.5B",
        },
        "Qwen2-Math-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-7B",
        },
        "Qwen2-Math-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-72B",
        },
        "Qwen2-Math-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-1.5B-Instruct",
        },
        "Qwen2-Math-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-7B-Instruct",
        },
        "Qwen2-Math-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Math-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Math-72B-Instruct",
        },
    },
    template="qwen",
)


register_model_group(
    models={
        "Qwen2.5-0.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B",
        },
        "Qwen2.5-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B",
        },
        "Qwen2.5-3B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B",
        },
        "Qwen2.5-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B",
        },
        "Qwen2.5-14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B",
        },
        "Qwen2.5-32B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B",
        },
        "Qwen2.5-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B",
        },
        "Qwen2.5-0.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct",
        },
        "Qwen2.5-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct",
        },
        "Qwen2.5-3B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct",
        },
        "Qwen2.5-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct",
        },
        "Qwen2.5-14B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct",
        },
        "Qwen2.5-32B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct",
        },
        "Qwen2.5-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct",
        },
        "Qwen2.5-7B-Instruct-1M": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-1M",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct-1M",
        },
        "Qwen2.5-14B-Instruct-1M": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-1M",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct-1M",
        },
        "Qwen2.5-0.5B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-0.5B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-0.5B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-0.5B-Instruct-AWQ",
        },
        "Qwen2.5-1.5B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-1.5B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-1.5B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        },
        "Qwen2.5-3B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-3B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-3B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-3B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-3B-Instruct-AWQ",
        },
        "Qwen2.5-7B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-7B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-7B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-7B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-7B-Instruct-AWQ",
        },
        "Qwen2.5-14B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-14B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-14B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-14B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-14B-Instruct-AWQ",
        },
        "Qwen2.5-32B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-32B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-32B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-32B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-32B-Instruct-AWQ",
        },
        "Qwen2.5-72B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int8",
        },
        "Qwen2.5-72B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4",
        },
        "Qwen2.5-72B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-72B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-72B-Instruct-AWQ",
        },
        "Qwen2.5-Coder-0.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-0.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-0.5B",
        },
        "Qwen2.5-Coder-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-1.5B",
        },
        "Qwen2.5-Coder-3B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-3B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-3B",
        },
        "Qwen2.5-Coder-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-7B",
        },
        "Qwen2.5-Coder-14B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-14B",
        },
        "Qwen2.5-Coder-32B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-32B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-32B",
        },
        "Qwen2.5-Coder-0.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-0.5B-Instruct",
        },
        "Qwen2.5-Coder-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        },
        "Qwen2.5-Coder-3B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-3B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-3B-Instruct",
        },
        "Qwen2.5-Coder-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        "Qwen2.5-Coder-14B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-14B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-14B-Instruct",
        },
        "Qwen2.5-Coder-32B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Coder-32B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-32B-Instruct",
        },
        "Qwen2.5-Math-1.5B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-1.5B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-1.5B",
        },
        "Qwen2.5-Math-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-7B",
        },
        "Qwen2.5-Math-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Math-72B",
        },
        "Qwen2.5-Math-1.5B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-1.5B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        },
        "Qwen2.5-Math-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-7B-Instruct",
        },
        "Qwen2.5-Math-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Math-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Coder-72B-Instruct",
        },
        "QwQ-32B-Preview-Instruct": {
            DownloadSource.DEFAULT: "Qwen/QwQ-32B-Preview",
            DownloadSource.MODELSCOPE: "Qwen/QwQ-32B-Preview",
        },
        "QwQ-32B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/QwQ-32B",
            DownloadSource.MODELSCOPE: "Qwen/QwQ-32B",
        },
    },
    template="qwen",
)


register_model_group(
    models={
        "Qwen3-0.6B-Base": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-0.6B-Base",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-0.6B-Base",
        },
        "Qwen3-1.7B-Base": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-1.7B-Base",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-1.7B-Base",
        },
        "Qwen3-4B-Base": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-4B-Base",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B-Base",
        },
        "Qwen3-8B-Base": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-8B-Base",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-8B-Base",
        },
        "Qwen3-14B-Base": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-14B-Base",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-14B-Base",
        },
        "Qwen3-30B-A3B-Base": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-30B-A3B-Base",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-Base",
        },
        "Qwen3-0.6B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-0.6B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-0.6B",
        },
        "Qwen3-1.7B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-1.7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-1.7B",
        },
        "Qwen3-4B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-4B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B",
        },
        "Qwen3-8B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-8B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-8B",
        },
        "Qwen3-14B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-14B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-14B",
        },
        "Qwen3-32B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-32B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-32B",
        },
        "Qwen3-30B-A3B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-30B-A3B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B",
        },
        "Qwen3-30B-A3B-Instruct-2507": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-30B-A3B-Instruct-2507",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-Instruct-2507",
        },
        "Qwen3-30B-A3B-Thinking-2507": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-30B-A3B-Thinking-2507",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-Thinking-2507",
        },
        "Qwen3-235B-A22B-Thinking": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-235B-A22B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B",
        },
        "Qwen3-235B-A22B-Instruct-2507": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-235B-A22B-Instruct-2507",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B-Instruct-2507",
        },
        "Qwen3-235B-A22B-Thinking-2507": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-235B-A22B-Thinking-2507",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B-Thinking-2507",
        },
        "Qwen3-0.6B-Thinking-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-0.6B-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-0.6B-GPTQ-Int8",
        },
        "Qwen3-1.7B-Thinking-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-1.7B-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-1.7B-GPTQ-Int8",
        },
        "Qwen3-4B-Thinking-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-4B-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-4B-AWQ",
        },
        "Qwen3-8B-Thinking-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-8B-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-8B-AWQ",
        },
        "Qwen3-14B-Thinking-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-14B-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-14B-AWQ",
        },
        "Qwen3-32B-Thinking-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-32B-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-32B-AWQ",
        },
        "Qwen3-30B-A3B-Thinking-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-30B-A3B-GPTQ-Int4",
        },
        "Qwen3-235B-A22B-Thinking-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen3-235B-A22B-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen3-235B-A22B-GPTQ-Int4",
        },
    },
    template="qwen3",
)


register_model_group(
    models={
        "Qwen2-Audio-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Audio-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Audio-7B",
        },
        "Qwen2-Audio-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-Audio-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-Audio-7B-Instruct",
        },
    },
    template="qwen2_audio",
    multimodal=True,
)


register_model_group(
    models={
        "Qwen2.5-Omni-3B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Omni-3B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Omni-3B",
        },
        "Qwen2.5-Omni-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Omni-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Omni-7B",
        },
        "Qwen2.5-Omni-7B-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Omni-7B-GPTQ-Int4",
        },
        "Qwen2.5-Omni-7B-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-Omni-7B-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-Omni-7B-AWQ",
        },
    },
    template="qwen2_omni",
    multimodal=True,
)


register_model_group(
    models={
        "Qwen2-VL-2B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-2B",
        },
        "Qwen2-VL-7B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-7B",
        },
        "Qwen2-VL-72B": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-72B",
        },
        "Qwen2-VL-2B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-2B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-VL-2B-Instruct",
        },
        "Qwen2-VL-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-7B-Instruct",
            DownloadSource.OPENMIND: "LlamaFactory/Qwen2-VL-7B-Instruct",
        },
        "Qwen2-VL-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-72B-Instruct",
        },
        "Qwen2-VL-2B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int8",
        },
        "Qwen2-VL-2B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-2B-Instruct-GPTQ-Int4",
        },
        "Qwen2-VL-2B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-2B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-2B-Instruct-AWQ",
        },
        "Qwen2-VL-7B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int8",
        },
        "Qwen2-VL-7B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-7B-Instruct-GPTQ-Int4",
        },
        "Qwen2-VL-7B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-7B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-7B-Instruct-AWQ",
        },
        "Qwen2-VL-72B-Instruct-GPTQ-Int8": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int8",
        },
        "Qwen2-VL-72B-Instruct-GPTQ-Int4": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-72B-Instruct-GPTQ-Int4",
        },
        "Qwen2-VL-72B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2-VL-72B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2-VL-72B-Instruct-AWQ",
        },
        "QVQ-72B-Preview": {
            DownloadSource.DEFAULT: "Qwen/QVQ-72B-Preview",
            DownloadSource.MODELSCOPE: "Qwen/QVQ-72B-Preview",
        },
        "Qwen2.5-VL-3B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-3B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-3B-Instruct",
        },
        "Qwen2.5-VL-7B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-7B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-7B-Instruct",
        },
        "Qwen2.5-VL-32B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-32B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-32B-Instruct",
        },
        "Qwen2.5-VL-72B-Instruct": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-72B-Instruct",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-72B-Instruct",
        },
        "Qwen2.5-VL-3B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        },
        "Qwen2.5-VL-7B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
        },
        "Qwen2.5-VL-72B-Instruct-AWQ": {
            DownloadSource.DEFAULT: "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
            DownloadSource.MODELSCOPE: "Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        },
    },
    template="qwen2_vl",
    multimodal=True,
)


register_model_group(
    models={
        "Seed-Coder-8B-Base": {
            DownloadSource.DEFAULT: "ByteDance-Seed/Seed-Coder-8B-Base",
        },
        "Seed-Coder-8B-Instruct": {
            DownloadSource.DEFAULT: "ByteDance-Seed/Seed-Coder-8B-Instruct",
        },
        "Seed-Coder-8B-Instruct-Reasoning": {
            DownloadSource.DEFAULT: "ByteDance-Seed/Seed-Coder-8B-Reasoning-bf16",
        },
    },
    template="seed_coder",
)


register_model_group(
    models={
        "Skywork-13B-Base": {
            DownloadSource.DEFAULT: "Skywork/Skywork-13B-base",
            DownloadSource.MODELSCOPE: "skywork/Skywork-13B-base",
        }
    }
)


register_model_group(
    models={
        "Skywork-o1-Open-Llama-3.1-8B": {
            DownloadSource.DEFAULT: "Skywork/Skywork-o1-Open-Llama-3.1-8B",
            DownloadSource.MODELSCOPE: "AI-ModelScope/Skywork-o1-Open-Llama-3.1-8B",
        }
    },
    template="skywork_o1",
)


register_model_group(
    models={
        "SmolLM-135M": {
            DownloadSource.DEFAULT: "HuggingFaceTB/SmolLM-135M",
            DownloadSource.MODELSCOPE: "HuggingFaceTB/SmolLM-135M",
        },
        "SmolLM-360M": {
            DownloadSource.DEFAULT: "HuggingFaceTB/SmolLM-360M",
            DownloadSource.MODELSCOPE: "HuggingFaceTB/SmolLM-360M",
        },
        "SmolLM-1.7B": {
            DownloadSource.DEFAULT: "HuggingFaceTB/SmolLM-1.7B",
            DownloadSource.MODELSCOPE: "HuggingFaceTB/SmolLM-1.7B",
        },
        "SmolLM-135M-Instruct": {
            DownloadSource.DEFAULT: "HuggingFaceTB/SmolLM-135M-Instruct",
            DownloadSource.MODELSCOPE: "HuggingFaceTB/SmolLM-135M-Instruct",
        },
        "SmolLM-360M-Instruct": {
            DownloadSource.DEFAULT: "HuggingFaceTB/SmolLM-360M-Instruct",
            DownloadSource.MODELSCOPE: "HuggingFaceTB/SmolLM-360M-Instruct",
        },
        "SmolLM-1.7B-Instruct": {
            DownloadSource.DEFAULT: "HuggingFaceTB/SmolLM-1.7B-Instruct",
            DownloadSource.MODELSCOPE: "HuggingFaceTB/SmolLM-1.7B-Instruct",
        },
    },
    template="smollm",
)


register_model_group(
    models={
        "SmolLM2-135M": {
            DownloadSource.DEFAULT: "HuggingFaceTB/SmolLM2-135M",
            DownloadSource.MODELSCOPE: "HuggingFaceTB/SmolLM2-135M",
        },
        "SmolLM2-360M": {
            DownloadSource.DEFAULT: "HuggingFaceTB/SmolLM2-360M",
            DownloadSource.MODELSCOPE: "HuggingFaceTB/SmolLM2-360M",
        },
        "SmolLM2-1.7B": {
            DownloadSource.DEFAULT: "HuggingFaceTB/SmolLM2-1.7B",
            DownloadSource.MODELSCOPE: "HuggingFaceTB/SmolLM2-1.7B",
        },
        "SmolLM2-135M-Instruct": {
            DownloadSource.DEFAULT: "HuggingFaceTB/SmolLM2-135M-Instruct",
            DownloadSource.MODELSCOPE: "HuggingFaceTB/SmolLM2-135M-Instruct",
        },
        "SmolLM2-360M-Instruct": {
            DownloadSource.DEFAULT: "HuggingFaceTB/SmolLM2-360M-Instruct",
            DownloadSource.MODELSCOPE: "HuggingFaceTB/SmolLM2-360M-Instruct",
        },
        "SmolLM2-1.7B-Instruct": {
            DownloadSource.DEFAULT: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            DownloadSource.MODELSCOPE: "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        },
    },
    template="smollm2",
)


register_model_group(
    models={
        "SOLAR-10.7B-v1.0": {
            DownloadSource.DEFAULT: "upstage/SOLAR-10.7B-v1.0",
        },
        "SOLAR-10.7B-Instruct-v1.0": {
            DownloadSource.DEFAULT: "upstage/SOLAR-10.7B-Instruct-v1.0",
            DownloadSource.MODELSCOPE: "AI-ModelScope/SOLAR-10.7B-Instruct-v1.0",
        },
    },
    template="solar",
)


register_model_group(
    models={
        "StarCoder2-3B": {
            DownloadSource.DEFAULT: "bigcode/starcoder2-3b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/starcoder2-3b",
        },
        "StarCoder2-7B": {
            DownloadSource.DEFAULT: "bigcode/starcoder2-7b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/starcoder2-7b",
        },
        "StarCoder2-15B": {
            DownloadSource.DEFAULT: "bigcode/starcoder2-15b",
            DownloadSource.MODELSCOPE: "AI-ModelScope/starcoder2-15b",
        },
    }
)


register_model_group(
    models={
        "TeleChat-1B-Chat": {
            DownloadSource.DEFAULT: "Tele-AI/TeleChat-1B",
            DownloadSource.MODELSCOPE: "TeleAI/TeleChat-1B",
        },
        "TeleChat-7B-Chat": {
            DownloadSource.DEFAULT: "Tele-AI/telechat-7B",
            DownloadSource.MODELSCOPE: "TeleAI/telechat-7B",
            DownloadSource.OPENMIND: "TeleAI/TeleChat-7B-pt",
        },
        "TeleChat-12B-Chat": {
            DownloadSource.DEFAULT: "Tele-AI/TeleChat-12B-v2",
            DownloadSource.MODELSCOPE: "TeleAI/TeleChat-12B-v2",
            DownloadSource.OPENMIND: "TeleAI/TeleChat-12B-pt",
        },
        "TeleChat-52B-Chat": {
            DownloadSource.DEFAULT: "Tele-AI/TeleChat-52B",
        },
    },
    template="telechat",
)


register_model_group(
    models={
        "TeleChat2-3B-Chat": {
            DownloadSource.DEFAULT: "Tele-AI/TeleChat2-3B",
            DownloadSource.MODELSCOPE: "TeleAI/TeleChat2-3B",
        },
        "TeleChat2-7B-Chat": {
            DownloadSource.DEFAULT: "Tele-AI/TeleChat2-7B",
            DownloadSource.MODELSCOPE: "TeleAI/TeleChat2-7B",
        },
        "TeleChat2-35B-Chat": {
            DownloadSource.MODELSCOPE: "TeleAI/TeleChat2-35B-Nov",
        },
        "TeleChat2-115B-Chat": {
            DownloadSource.DEFAULT: "Tele-AI/TeleChat2-115B",
            DownloadSource.MODELSCOPE: "TeleAI/TeleChat2-115B",
        },
    },
    template="telechat2",
)


register_model_group(
    models={
        "Vicuna-v1.5-7B-Chat": {
            DownloadSource.DEFAULT: "lmsys/vicuna-7b-v1.5",
            DownloadSource.MODELSCOPE: "Xorbits/vicuna-7b-v1.5",
        },
        "Vicuna-v1.5-13B-Chat": {
            DownloadSource.DEFAULT: "lmsys/vicuna-13b-v1.5",
            DownloadSource.MODELSCOPE: "Xorbits/vicuna-13b-v1.5",
        },
    },
    template="vicuna",
)


register_model_group(
    models={
        "Video-LLaVA-7B-Chat": {
            DownloadSource.DEFAULT: "LanguageBind/Video-LLaVA-7B-hf",
        },
    },
    template="video_llava",
    multimodal=True,
)


register_model_group(
    models={
        "XuanYuan-6B": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan-6B",
            DownloadSource.MODELSCOPE: "Duxiaoman-DI/XuanYuan-6B",
        },
        "XuanYuan-70B": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan-70B",
            DownloadSource.MODELSCOPE: "Duxiaoman-DI/XuanYuan-70B",
        },
        "XuanYuan2-70B": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan2-70B",
            DownloadSource.MODELSCOPE: "Duxiaoman-DI/XuanYuan2-70B",
        },
        "XuanYuan-6B-Chat": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan-6B-Chat",
            DownloadSource.MODELSCOPE: "Duxiaoman-DI/XuanYuan-6B-Chat",
        },
        "XuanYuan-70B-Chat": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan-70B-Chat",
            DownloadSource.MODELSCOPE: "Duxiaoman-DI/XuanYuan-70B-Chat",
        },
        "XuanYuan2-70B-Chat": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan2-70B-Chat",
            DownloadSource.MODELSCOPE: "Duxiaoman-DI/XuanYuan2-70B-Chat",
        },
        "XuanYuan-6B-Chat-8bit": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan-6B-Chat-8bit",
            DownloadSource.MODELSCOPE: "Duxiaoman-DI/XuanYuan-6B-Chat-8bit",
        },
        "XuanYuan-6B-Chat-4bit": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan-6B-Chat-4bit",
            DownloadSource.MODELSCOPE: "Duxiaoman-DI/XuanYuan-6B-Chat-4bit",
        },
        "XuanYuan-70B-Chat-8bit": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan-70B-Chat-8bit",
            DownloadSource.MODELSCOPE: "Duxiaoman-DI/XuanYuan-70B-Chat-8bit",
        },
        "XuanYuan-70B-Chat-4bit": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan-70B-Chat-4bit",
            DownloadSource.MODELSCOPE: "Duxiaoman-DI/XuanYuan-70B-Chat-4bit",
        },
        "XuanYuan2-70B-Chat-8bit": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan2-70B-Chat-8bit",
            DownloadSource.MODELSCOPE: "Duxiaoman-DI/XuanYuan2-70B-Chat-8bit",
        },
        "XuanYuan2-70B-Chat-4bit": {
            DownloadSource.DEFAULT: "Duxiaoman-DI/XuanYuan2-70B-Chat-4bit",
            DownloadSource.MODELSCOPE: "Duxiaoman-DI/XuanYuan2-70B-Chat-4bit",
        },
    },
    template="xuanyuan",
)


register_model_group(
    models={
        "XVERSE-7B": {
            DownloadSource.DEFAULT: "xverse/XVERSE-7B",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-7B",
        },
        "XVERSE-13B": {
            DownloadSource.DEFAULT: "xverse/XVERSE-13B",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-13B",
        },
        "XVERSE-65B": {
            DownloadSource.DEFAULT: "xverse/XVERSE-65B",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-65B",
        },
        "XVERSE-65B-2": {
            DownloadSource.DEFAULT: "xverse/XVERSE-65B-2",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-65B-2",
        },
        "XVERSE-7B-Chat": {
            DownloadSource.DEFAULT: "xverse/XVERSE-7B-Chat",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-7B-Chat",
        },
        "XVERSE-13B-Chat": {
            DownloadSource.DEFAULT: "xverse/XVERSE-13B-Chat",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-13B-Chat",
        },
        "XVERSE-65B-Chat": {
            DownloadSource.DEFAULT: "xverse/XVERSE-65B-Chat",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-65B-Chat",
        },
        "XVERSE-MoE-A4.2B": {
            DownloadSource.DEFAULT: "xverse/XVERSE-MoE-A4.2B",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-MoE-A4.2B",
        },
        "XVERSE-7B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "xverse/XVERSE-7B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-7B-Chat-GPTQ-Int8",
        },
        "XVERSE-7B-Chat-GPTQ-Int4": {
            DownloadSource.DEFAULT: "xverse/XVERSE-7B-Chat-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-7B-Chat-GPTQ-Int4",
        },
        "XVERSE-13B-Chat-GPTQ-Int8": {
            DownloadSource.DEFAULT: "xverse/XVERSE-13B-Chat-GPTQ-Int8",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-13B-Chat-GPTQ-Int8",
        },
        "XVERSE-13B-Chat-GPTQ-Int4": {
            DownloadSource.DEFAULT: "xverse/XVERSE-13B-Chat-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-13B-Chat-GPTQ-Int4",
        },
        "XVERSE-65B-Chat-GPTQ-Int4": {
            DownloadSource.DEFAULT: "xverse/XVERSE-65B-Chat-GPTQ-Int4",
            DownloadSource.MODELSCOPE: "xverse/XVERSE-65B-Chat-GPTQ-Int4",
        },
    },
    template="xverse",
)


register_model_group(
    models={
        "Yayi-7B": {
            DownloadSource.DEFAULT: "wenge-research/yayi-7b-llama2",
            DownloadSource.MODELSCOPE: "AI-ModelScope/yayi-7b-llama2",
        },
        "Yayi-13B": {
            DownloadSource.DEFAULT: "wenge-research/yayi-13b-llama2",
            DownloadSource.MODELSCOPE: "AI-ModelScope/yayi-13b-llama2",
        },
    },
    template="yayi",
)


register_model_group(
    models={
        "Yi-6B": {
            DownloadSource.DEFAULT: "01-ai/Yi-6B",
            DownloadSource.MODELSCOPE: "01ai/Yi-6B",
        },
        "Yi-9B": {
            DownloadSource.DEFAULT: "01-ai/Yi-9B",
            DownloadSource.MODELSCOPE: "01ai/Yi-9B",
        },
        "Yi-34B": {
            DownloadSource.DEFAULT: "01-ai/Yi-34B",
            DownloadSource.MODELSCOPE: "01ai/Yi-34B",
        },
        "Yi-6B-Chat": {
            DownloadSource.DEFAULT: "01-ai/Yi-6B-Chat",
            DownloadSource.MODELSCOPE: "01ai/Yi-6B-Chat",
        },
        "Yi-34B-Chat": {
            DownloadSource.DEFAULT: "01-ai/Yi-34B-Chat",
            DownloadSource.MODELSCOPE: "01ai/Yi-34B-Chat",
        },
        "Yi-6B-Chat-8bits": {
            DownloadSource.DEFAULT: "01-ai/Yi-6B-Chat-8bits",
            DownloadSource.MODELSCOPE: "01ai/Yi-6B-Chat-8bits",
        },
        "Yi-6B-Chat-4bits": {
            DownloadSource.DEFAULT: "01-ai/Yi-6B-Chat-4bits",
            DownloadSource.MODELSCOPE: "01ai/Yi-6B-Chat-4bits",
        },
        "Yi-34B-Chat-8bits": {
            DownloadSource.DEFAULT: "01-ai/Yi-34B-Chat-8bits",
            DownloadSource.MODELSCOPE: "01ai/Yi-34B-Chat-8bits",
        },
        "Yi-34B-Chat-4bits": {
            DownloadSource.DEFAULT: "01-ai/Yi-34B-Chat-4bits",
            DownloadSource.MODELSCOPE: "01ai/Yi-34B-Chat-4bits",
        },
        "Yi-1.5-6B": {
            DownloadSource.DEFAULT: "01-ai/Yi-1.5-6B",
            DownloadSource.MODELSCOPE: "01ai/Yi-1.5-6B",
        },
        "Yi-1.5-9B": {
            DownloadSource.DEFAULT: "01-ai/Yi-1.5-9B",
            DownloadSource.MODELSCOPE: "01ai/Yi-1.5-9B",
        },
        "Yi-1.5-34B": {
            DownloadSource.DEFAULT: "01-ai/Yi-1.5-34B",
            DownloadSource.MODELSCOPE: "01ai/Yi-1.5-34B",
        },
        "Yi-1.5-6B-Chat": {
            DownloadSource.DEFAULT: "01-ai/Yi-1.5-6B-Chat",
            DownloadSource.MODELSCOPE: "01ai/Yi-1.5-6B-Chat",
            DownloadSource.OPENMIND: "LlamaFactory/Yi-1.5-6B-Chat",
        },
        "Yi-1.5-9B-Chat": {
            DownloadSource.DEFAULT: "01-ai/Yi-1.5-9B-Chat",
            DownloadSource.MODELSCOPE: "01ai/Yi-1.5-9B-Chat",
        },
        "Yi-1.5-34B-Chat": {
            DownloadSource.DEFAULT: "01-ai/Yi-1.5-34B-Chat",
            DownloadSource.MODELSCOPE: "01ai/Yi-1.5-34B-Chat",
        },
        "Yi-Coder-1.5B": {
            DownloadSource.DEFAULT: "01-ai/Yi-Coder-1.5B",
            DownloadSource.MODELSCOPE: "01ai/Yi-Coder-1.5B",
        },
        "Yi-Coder-9B": {
            DownloadSource.DEFAULT: "01-ai/Yi-Coder-9B",
            DownloadSource.MODELSCOPE: "01ai/Yi-Coder-9B",
        },
        "Yi-Coder-1.5B-Chat": {
            DownloadSource.DEFAULT: "01-ai/Yi-Coder-1.5B-Chat",
            DownloadSource.MODELSCOPE: "01ai/Yi-Coder-1.5B-Chat",
        },
        "Yi-Coder-9B-Chat": {
            DownloadSource.DEFAULT: "01-ai/Yi-Coder-9B-Chat",
            DownloadSource.MODELSCOPE: "01ai/Yi-Coder-9B-Chat",
        },
    },
    template="yi",
)


register_model_group(
    models={
        "Yi-VL-6B-Chat": {
            DownloadSource.DEFAULT: "BUAADreamer/Yi-VL-6B-hf",
        },
        "Yi-VL-34B-Chat": {
            DownloadSource.DEFAULT: "BUAADreamer/Yi-VL-34B-hf",
        },
    },
    template="yi_vl",
    multimodal=True,
)


register_model_group(
    models={
        "Yuan2-2B-Chat": {
            DownloadSource.DEFAULT: "IEITYuan/Yuan2-2B-hf",
            DownloadSource.MODELSCOPE: "YuanLLM/Yuan2.0-2B-hf",
        },
        "Yuan2-51B-Chat": {
            DownloadSource.DEFAULT: "IEITYuan/Yuan2-51B-hf",
            DownloadSource.MODELSCOPE: "YuanLLM/Yuan2.0-51B-hf",
        },
        "Yuan2-102B-Chat": {
            DownloadSource.DEFAULT: "IEITYuan/Yuan2-102B-hf",
            DownloadSource.MODELSCOPE: "YuanLLM/Yuan2.0-102B-hf",
        },
    },
    template="yuan",
)


register_model_group(
    models={
        "Zephyr-7B-Alpha-Chat": {
            DownloadSource.DEFAULT: "HuggingFaceH4/zephyr-7b-alpha",
            DownloadSource.MODELSCOPE: "AI-ModelScope/zephyr-7b-alpha",
        },
        "Zephyr-7B-Beta-Chat": {
            DownloadSource.DEFAULT: "HuggingFaceH4/zephyr-7b-beta",
            DownloadSource.MODELSCOPE: "modelscope/zephyr-7b-beta",
        },
        "Zephyr-141B-ORPO-Chat": {
            DownloadSource.DEFAULT: "HuggingFaceH4/zephyr-orpo-141b-A35b-v0.1",
        },
    },
    template="zephyr",
)
