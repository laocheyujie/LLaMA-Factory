# Copyright 2025 HuggingFace Inc., Daniel Han-Chen & the Unsloth team and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's Transformers and PEFT library,
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/modeling_utils.py
# https://github.com/huggingface/peft/blob/v0.10.0/src/peft/utils/other.py
# and the Unsloth library.
# https://github.com/unslothai/unsloth/blob/July-2024/unsloth/models/_utils.py
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

import inspect
from functools import WRAPPER_ASSIGNMENTS, partial, wraps
from types import MethodType
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch

from ...extras import logging
from ...extras.constants import LAYERNORM_NAMES


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    from ...hparams import ModelArguments


logger = logging.get_logger(__name__)


def get_unsloth_gradient_checkpointing_func() -> Callable:
    class UnslothGradientCheckpointing(torch.autograd.Function):
        r"""Saves VRAM by smartly offloading to RAM."""

        @staticmethod
        @torch.cuda.amp.custom_fwd
        def forward(
            ctx: "torch.autograd.Function",
            forward_function: "torch.Module",
            hidden_states: "torch.Tensor",
            *args: Union["torch.Tensor", Any],
        ) -> "torch.Tensor":
            saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
            # NOTE: 禁用了 autograd 保存中间值（等于“激活不保存”）
            # NOTE: with torch.no_grad(): 不记录操作、不保存中间值、不计算梯度
            with torch.no_grad():
                outputs = forward_function(hidden_states, *args)

            ctx.save_for_backward(saved_hidden_states)
            ctx.forward_function = forward_function
            ctx.args = args
            return outputs

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx: "torch.autograd.Function", grad_output: "torch.Tensor") -> "torch.Tensor":
            (hidden_states,) = ctx.saved_tensors
            hidden_states = hidden_states.to("cuda", non_blocking=True).detach()
            hidden_states.requires_grad_(True)
            # NOTE: 打开梯度计算，以便计算梯度
            with torch.enable_grad():
                outputs = ctx.forward_function(hidden_states, *ctx.args)
                output = outputs[0] if isinstance(outputs, tuple) else outputs

            torch.autograd.backward(output, grad_output)
            return (None, hidden_states.grad) + (None,) * len(ctx.args)

    return UnslothGradientCheckpointing.apply


def get_custom_gradient_checkpointing_func(gradient_checkpointing_func: Callable) -> Callable:
    r"""Only applies gradient checkpointing to trainable layers."""

    @wraps(gradient_checkpointing_func, assigned=WRAPPER_ASSIGNMENTS + ("__self__",))
    def custom_gradient_checkpointing_func(func: Callable, *args: Union["torch.Tensor", Any], **kwargs):
        # NOTE: 获取模块实例
        if isinstance(func, partial):
            module: torch.nn.Module = func.func.__self__
        else:
            module: torch.nn.Module = func.__self__

        # NOTE: 检查模块是否有需要梯度的参数
        has_grad = False
        if any(param.requires_grad for param in module.parameters()):
            has_grad = True
            # NOTE: 确保第一个张量参数（通常是隐藏状态）需要梯度
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):
                    arg.requires_grad_(True)
                    # 只处理第一个浮点张量，因为通常这是隐藏状态
                    break  # assume the first tensor is always the hidden states

        if has_grad:
            # NOTE: gradient_checkpointing_func 其实就是预填了一些参数的 checkpoint 函数
            return gradient_checkpointing_func(func, *args, **kwargs)
        else:
            return func(*args, **kwargs)

    return custom_gradient_checkpointing_func


def _gradient_checkpointing_enable(
    self: "PreTrainedModel",
    gradient_checkpointing_kwargs: Optional[dict[str, Any]] = None,
    use_unsloth_gc: bool = False,
) -> None:
    r"""Activates gradient checkpointing for the current model.

    Modification of the original method to enable gradient checkpointing for block-wise optimizer.
    """
    from torch.utils.checkpoint import checkpoint

    if not self.supports_gradient_checkpointing:
        raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}

    if use_unsloth_gc:
        gradient_checkpointing_func = get_unsloth_gradient_checkpointing_func()
    else:
        gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

    gradient_checkpointing_func = get_custom_gradient_checkpointing_func(gradient_checkpointing_func)
    # NOTE: 检查模型使用的是新格式还是旧格式的梯度检查点
    if "value" in inspect.signature(self._set_gradient_checkpointing).parameters:  # old GC format
        # NOTE: 对于旧格式，直接设置 value=True 并启用输入梯度
        self.apply(partial(self._set_gradient_checkpointing, value=True))
        self.enable_input_require_grads()
        logger.warning_rank0_once("You are using the old GC format, some features (e.g. BAdam) will be invalid.")
    else:  # have already enabled input require gradients
        # NOTE:对于新格式，使用新的设置方式，传入自定义的梯度检查点函数
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)


def _fp32_forward_post_hook(
    module: "torch.nn.Module", args: tuple["torch.Tensor"], output: "torch.Tensor"
) -> "torch.Tensor":
    return output.to(torch.float32)


def prepare_model_for_training(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    r"""Prepare the model before training.

    Include:
    (1) cast the layernorm in fp32
    (2) make output embedding layer require grads
    (3) add the upcasting of the lm_head in fp32.
    """
    if model_args.upcast_layernorm:
        logger.info_rank0("Upcasting layernorm weights in float32.")
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in LAYERNORM_NAMES):
                param.data = param.data.to(torch.float32)

    if not model_args.disable_gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning_rank0("Current model does not support gradient checkpointing.")
        else:
            # use_reentrant=False might increase VRAM usage (have not been empirically verified yet)
            # According to: https://github.com/huggingface/transformers/issues/28339
            # NOTE: 使用partial创建一个新的函数，预设了use_unsloth_gc参数
            gradient_checkpointing_enable = partial(
                _gradient_checkpointing_enable, use_unsloth_gc=model_args.use_unsloth_gc
            )
            # NOTE: 将梯度检查点启用函数绑定到模型实例上，这样模型就可以直接调用gradient_checkpointing_enable方法
            model.gradient_checkpointing_enable = MethodType(gradient_checkpointing_enable, model)
            # NOTE: 实际上就是用 _gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": model_args.use_reentrant_gc}, use_unsloth_gc=model_args.use_unsloth_gc)
            # NOTE: use_reentrant 参数控制是否使用可重入的梯度检查点实现，use_reentrant=False可能会增加显存使用，但尚未得到经验验证
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": model_args.use_reentrant_gc}
            )
            setattr(model.config, "use_cache", False)  # turn off when gradient checkpointing is enabled
            logger.info_rank0("Gradient checkpointing enabled.")

    if model_args.upcast_lmhead_output:
        output_layer = model.get_output_embeddings()
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            logger.info_rank0("Upcasting lm_head outputs in float32.")
            output_layer.register_forward_hook(_fp32_forward_post_hook)
