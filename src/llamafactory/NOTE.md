## 整体流程
### cli.py
根据不同命调用不同的函数，如果是 train 且 gpu > 1，则通过子进程用 trochrun 开始分布式训练
torchrun 启动的是 launcher 脚本，launcher 又调用乐 train.tuner.run_exp()


## 源码中的重点
- `dataset_attr.num_samples`: 单个数据集样本的数量（少了就随机采样扩充，多了就随机丢弃）
- `data_args.max_samples`: 单个数据集中的最大样本数（min(data_args.max_samples, len(datasets))
- `data_args.cutoff_len`: tokenize后的token最大长度
- `config.json`里的`max_position_embeddings`是模型原本支持的最大长度
- 梯度检查点和模型缓存不能同时使用 TODO: WHY
- `finetuning_args.freeze_trainable_layers`大于0，微调指定module的后n layers；小于0，微调指定module的前n layers
- 多模态模型的DataCollator对于batch里没有任何多模态的数据时，会伪造第一个样本的多模态信息，避免分布式训练被挂起
- attention_mask = 0 表示不被关注的 padding
- labels = -100 表示不参与计算损失
- pt 就是把`text`直接tokenizer后得到`input_ids`用于自回归训练；sft 就是把`messages`按照`Template`转换后，`query`+`res`作为`input_ids`，`query`位置填充`-100`的`query`+`res`作为`label`，即把`query`不计算loss而已
- Dataset负责加载和预处理原始数据，个样本会被转换成包含`input_ids`、`attention_mask`、`labels`等字段的字典；DataCollator负责将多个样本组合成一个batch
- pt 的`PretrainDatasetProcessor`在最后加了eos，所以不用做shift


## train
### train.tuner
#### run_exp
1. `read_args`获取参数`args`
2. `get_ray_args`获取`ray_args`参数，返回的是展平后的字典
3. `_training_function`训练

#### _training_function
1. `get_train_args`获取分类好的参数
2. 添加`LogCallback()`, `ReporterCallback`等 Callback
3. 根据训练类型的不同分别调用相应的`run_xx`
4. 如果 Ray 已经初始化，就直接返回，销毁 ray 进程组
5. 如果已初始化 PyTorch 分布式环境，就手动销毁进程组


### pt.workflow
#### run_pt
1. `model.loader.load_tokenizer`获取`tokenizer`和`processor`
2. `data.template.get_template_and_fix_tokenizer`获取`template`
3. `get_dataset`获取`dataset_module`
4. `model.loader.load_model`获取模型
5. 创建`data_collator`数据收集器
6. 创建`CustomTrainer`训练器
7. 训练时：
    1. 训练
    2. 评估
    3. 保存评估
    4. 保存状态
    5. 绘制损失函数
8. 评估时：
    1. 增加困惑度指标
    2. 评估
    3. 保存评估
9. 创建 Model Card


### pt.trainer
#### CustomTrainer
1. 如果有`processor`，添加保存`processor`的Callback`SaveProcessorCallback`
2. 如果使用`badam`，设置相应的`clip_grad_norm_`和`BAdamCallback`
3. 重载自定义的`create_optimizer`, `create_scheduler`, `_get_train_sampler`, `compute_loss`


### sft.workflow
#### run_sft
1. `model.loader.load_tokenizer`获取`tokenizer`和`processor`
2. `data.template.get_template_and_fix_tokenizer`获取`template`
3. `get_dataset`获取`dataset_module`
4. `model.loader.load_model`获取模型
5. 创建`SFTDataCollatorWith4DAttentionMask`数据收集器
6. 创建`CustomSeq2SeqTrainer`训练器
7. 训练时：
    1. 训练
    2. 评估
    3. 保存评估
    4. 保存状态
    5. 绘制损失函数
8. 评估时：
    2. 评估
    3. 保存评估
9. 创建 Model Card

### sft.trainer
#### CustomSeq2SeqTrainer
`prediction_step`中根据`args.predict_with_generate`参数，控制模型在预测阶段是否使用生成模式，它在训练时不用生成模式；评估时启用


## model
### model.loader
#### load_tokenizer
获取`tokenizer`和`processor`
    1. `tokenizer = AutoTokenizer.from_pretrained` 加载 tokenizer (tokenizer_config.json)，注意是 right padding
    2. `patch_tokenizer`对`tokenizer`进行一些加工处理
    3. `processor = AutoProcessor.from_pretrained` 加载 processor (preprocessor_config.json)
    4. `patch_processor`对`processor`进行一些加工处理

#### load_model
1. `load_config`加载模型的`config.json`配置`config`
2. `patch_config`对配置`config`打一些配置补丁
3. `apply_liger_kernel`应用Liger kernel
4. 如果设置了`model_args.use_unsloth = True`，就用 unsloth 加载 4bit 量化的模型；否则根据`config`的类型，取对应的`AutoModelForXXX`
5. 如果设置了`model_args.train_from_scratch = True`，就根据`config`加载一个初始化的空模型
6. 如果`model_args.mixture_of_depths == "load"`则加载`MOD`模型；否则根据`init_kwargs`加载预训练的模型
7. 如果`model_args.mixture_of_depths == "convert"`则把预训练模型转换为`MOD`模型
8. `patch_model`对模型进行词表扩充、精度转换等补丁
9. `register_autoclass`注册`config`, `model`, `tokenizer`为自动类型，可以通过Hugging Face的自动类系统被自动识别和加载
10. `init_adapter`初始化模型适配器（adapters）
11. add_valuehead：TODO: 没看懂
12. 如果不训练：进行精度转换并`model.eval()`；如果训练：`model.train()`



#### load_config
`AutoConfig.from_pretrained`加载模型的`config.json`配置


### model.pathcer
#### patch_config
1. `model_args.compute_dtype`的优先级: config.json里的torch_dtype的设定 > bf16 > fp16 > fp32
2. `configure_attn_implementation`设置`flash_attention`配置
3. `configure_rope`设置`rope`配置
4. `configure_longlora`设置`shift_attention`配置，只有`llama`模型支持
5. `configure_quantization`设置`quantization`配置
6. `configure_moe`设置`moe`配置
7. `configure_visual_model`设置`vlm`配置
8. `configure_packing`设置 Flash Attention 的序列打包，能够更高效地处理变长序列
9. `configure_kv_cache`设置`kv_cache`配置
10. 对`init_kwargs`进行一些后处理，得到
```py
init_kwargs = {'trust_remote_code': True, 'cache_dir': None, 'revision': 'main', 'token': None, 'low_cpu_mem_usage': True, 'torch_dtype': torch.bfloat16, 'device_map': {'': device(type='cuda', index=0)}}
```

#### patch_model
1. check and fix generation config
2. `prepare_valuehead_model` TODO: 这个是干什么的待查
3. `resize_embedding_layer` 扩充词表/embedding大小
4. `prepare_model_for_training` 模型训练前的准备工作
5. `autocast_projector_dtype` 转为半精度供量化的 VLMs 微调
6. `add_z3_leaf_module` 把 moe 模块设置为DeepSpeed Zero-3中的"叶子模块"（leaf modules），防止被分区（partitioning）


### model.model_utils.attention
#### configure_attn_implementation
在`config`属性里增加`_attn_implementation`，可选值: `['eager', 'sdpa', 'flash_attention_2']`

### model.model_utils.rope
#### configure_rope
1. `rope_scaling`可选值: `['linear', 'dynamic', 'yarn', 'llama3']`
2. `max_position_embeddings`如果大于训练时设定的`model_args.model_max_length`（`cutoff_len`），无需做rope
3. `rope_factor = float(math.ceil(model_args.model_max_length / old_max_length))`
4. 设定`config`的`max_position_embeddings`为`max_position_embeddings * rope_factor`
5. 设定`config`的`rope_scaling`为：
```json
{
    "rope_type": "yarn/dynamic/linear/llama3",
    "factor": "计算得到的factor",
    "original_max_position_embeddings": "原始的max_position_embeddings值"
}
```
如果是`llama3`，还要增加两个参数：
```json
{
    "low_freq_factor": 1,
    "high_freq_factor": 4
}
```

### model.model_utils.quantization
#### configure_quantization
支持：`['bnb', 'gptq', 'awq', 'aqlm', 'quanto', 'eetq', 'hqq']`
1. `config`里如果有`quantization_config`参数就直接用，但注意：
    1. `model_args.quantization_bit`不适用于`PTQ-quantized models`
    2. `DeepSpeed ZeRO-3 or FSDP`不适用于`PTQ-quantized models`
2. 如果使用`GPTQ`，即设置了`model_args.export_quantization_bit`
    1. `model_args.export_quantization_bit`只能是`[8, 4, 3, 2]`
    2. 设置`init_kwargs["quantization_config"] = GPTQConfig`
3. 如果设置了`model_args.quantization_bit`
    1. 如果使用`BNB`，即`model_args.quantization_method == 'bnb'`，设置`init_kwargs["quantization_config"] = BitsAndBytesConfig`
    2. 如果使用`HQQ`，即`model_args.quantization_method == 'hqq'`，设置`init_kwargs["quantization_config"] = HqqConfig`
    3. 如果使用`EETQ`，即`model_args.quantization_method == 'hqq'`，设置`init_kwargs["quantization_config"] = EetqConfig()`

### model.model_utils.moe
#### configure_moe
在`config`里添加路由配置，比如`output_router_logits`, `router_aux_loss_coef`等


### model.model_utils.packing
#### configure_packing
为 Flash Attention 的变长序列处理准备数据。在 Transformer 模型中，我们经常需要处理不同长度的序列，而 Flash Attention 需要特定的数据格式来高效处理这些变长序列。`get_unpad_data`函数会返回三个值：
1. `indices`：所有非零（非掩码）位置的索引
2. `cu_seqlens`：累积序列长度
3. `max_seqlen_in_batch`：当前批次中最长序列的长度
这些数据的作用是：
1. 帮助 Flash Attention 知道每个序列的实际长度
2. 允许模型只处理有效的token，跳过填充的token
3. 优化内存使用，因为不需要为填充token分配注意力计算资源


### model.model_utils.kv_cache
#### configure_kv_cache
- 训练时: `config.use_cache = False`, `config.text_config.use_cache = False`
- 推理时: `config.use_cache = model_args.use_cache`, `config.text_config.use_cache = model_args.use_cache`


### model.model_utils.checkpointing
#### prepare_model_for_training
1. 将 LayerNorm 层的权重转换为 fp32 精度
2. `_gradient_checkpointing_enable`启用梯度检查点，并禁用模型缓存`model.config.use_cache = False`
3. 通过注册钩子函数`register_forward_hook`设置输出层（lm_head）的精度为 fp32 精度

#### _gradient_checkpointing_enable
1. 设置`gradient_checkpointing_kwargs = {"use_reentrant": True}`
2. 对`checkpoint`使用`get_custom_gradient_checkpointing_func`加工一下，确保只对需要梯度的层应用梯度检查点
3. 检查模型使用的是新格式还是旧格式的梯度检查点
    - 对于旧格式，直接设置 value=True 并启用输入梯度
    - 对于新格式，使用新的设置方式，传入自定义的梯度检查点函数，实际上就是对`checkpoint`函数的加工

梯度检查点的工作原理是：
1. 在前向传播时，不保存所有的中间激活值
2. 在反向传播时，重新计算需要的中间激活值
3. 这样可以用计算时间换取显存空间

#### get_custom_gradient_checkpointing_func
确保只对需要梯度的层应用梯度检查点
1. 获取实际的模块实例
2. 检查模块是否有任何需要梯度的参数，如果模块中有任何参数需要梯度，则
    1. 设置`has_grad = True`
    2. 只处理第一个浮点张量，通常这是隐藏状态，确保隐藏状态需要梯度
3. 如果模块需要梯度，应用梯度检查点；如果模块不需要梯度，直接执行原始函数

### model.model_utils.visual
#### autocast_projector_dtype
如果有`model.quantization_method`，则先获取多模态模块，然后用注册钩子函数的方式对多模态模块进行精度转换

#### get_forbidden_modules
如果是已注册的多模态模型，获取具体的多模态的禁用模块（`freeze_vision_tower`, `freeze_multi_modal_projector`, `freeze_language_model`）


### model.adapter
#### init_adapter
1. 如果模型是量化的且需要训练，只允许使用 LoRA 方式进行微调
2. 不允许在量化模型上使用PiSSA初始化
3. 精度转换
    1. 使用纯bf16或BAdam优化器时保持半精度
    2. 使用DeepSpeed ZeRO3时保持float32
    3. 其他情况将可训练参数转换为float32
4. 设置微调
    1. full: 
        1. `get_forbidden_modules`获取多模态的禁用模块（`freeze_vision_tower`, `freeze_multi_modal_projector`, `freeze_language_model`）
        2. 对非禁用模块，如果需要转换为float32，则把数据类型转为float32
        3. 对于禁用模块，设置`param.requires_grad_(False)`
    2. freeze: 
        1. 获取config，优先用`model.config.text_config`，否则就用`model.config`
        2. 获取num_layers，`num_hidden_layers`>`num_layers`>`n_layer`
        3. 如果`finetuning_args.use_llama_pro=True`，间隔着取`finetuning_args.freeze_trainable_layers`个训练层
        4. 如果`finetuning_args.freeze_trainable_layers > 0`，取指定module的后n层
        5. 如果`finetuning_args.freeze_trainable_layers < 0`，取指定module的前n层
        6. 如果`finetuning_args.freeze_multi_modal_projector=False`并且模型类型是支持的多模态类型，添加`projector_key`层位训练层
        7. `get_forbidden_modules`获取多模态的禁用模块（`freeze_vision_tower`, `freeze_multi_modal_projector`, `freeze_language_model`）
        8. 对于`trainable_layer`且非禁用模块，如果需要转换为float32，则把数据类型转为float32
        9. 对于非`trainable_layer`或禁用模块，设置`param.requires_grad_(False)`
    3. lora:
        1. 如果设置了`quantization_method`或使用DeepSpeed ZeRO-3或使用Unsloth，都只能接受一个adapter
        2. 如果需要继续训练且不创建新adapter，或者不可合并时，最后一个adapter用于恢复训练，其他adapter用于合并
        3. 如果模型不训练或支持合并，所有的adapter都会被合并
        4. 遍历需要合并的适配器，加载每个适配器，将适配器合并到模型中并卸载
        5. 如果有需要恢复训练的adapter，如果使用 Unsloth，使用特殊的加载方法；否则使用标准的 PEFT 加载方法
        6. 如果需要训练且没有需要恢复训练的apapter，则需要创建新的 LoRA 权重
        7. 目标模块不包括：`lm_head`, `Embedding`, chatglm的`output_layer`, internlm2的`output`, vlms的`projector_key`，可选禁用vlms的每个部分
        8. 如果需要重置词表大小，则获取`input_embeddings`和`output_embeddings`，并将它们添加到`modules_to_save`
        9. 初始化 LoRA 配置，并获取添加了 LoRA 的模型
        10. 对需要训练的参数转换为float32精度并返回模型


## data
### data.template
#### get_template_and_fix_tokenizer
从`TEMPLATES`里拿到对应的模板实例（`Template`, `Llama2Template`, `ReasoningTemplate`），`TEMPLATES`是通过`register_template`得到的

#### register_template
根据参数，实例化`template_class`对应的`Template`

### data.loader
#### get_dataset
数据集的加载和处理只在`RANK 0`主进程中执行
1. `_get_merged_dataset`获取一个合并后的、格式转换后的训练数据集
2. `_get_preprocessed_dataset`不同的`stage`进行不同的tokenization，得到`Dataset`
3. `split_dataset`根据`val_size`大小分割`train`和`validation`得到`DatasetDict`
4. `get_dataset_module`将`DatasetDict`重新转成`DatasetModule`格式
```py
{
    "train_dataset": Dataset,
    "eval_dataset": Dataset
}
```



#### _get_preprocessed_dataset
1. `_get_dataset_processor`获取不同`stage`的`DataProcessor`
2. `dataset.map`调用`dataset_processor.preprocess_dataset`执行tokenization，返回`Dataset`
    - pt: `PretrainDatasetProcessor`
        1. 每条样本最后加上`eos_token`
        2. 如果有`packing`，先重亲按照`packing`切分
        3. 如果`tokenizer`有`add_bos_token`，就在每条样本前面插入`bos_token_id`
        4. 返回包含`input_ids`的tokenizer后的结果
    - sft: `SupervisedDatasetProcessor`
        1. 把多模态输入的占位符转换成对用的 special tokens
        2. 历史对话转换乘对话对`encoded_pairs`
        3. 根据`data_args.cutoff_len`智能分配输入和输出的长度
        4. 设置`input_ids`和`labels`，其中`input_ids`就是输入+输出的token ids；`labels`是遮蔽了输入或遮蔽了历史对话的token ids（遮蔽指的是对应位置的token id为-100）
        5. 返回`input_ids`和`labels`

```py
Dataset({
    features: ['input_ids', 'attention_mask'],
    num_rows: 71
})
```



#### _get_merged_dataset
1. `_load_single_dataset`获取单个格式转换后的数据集
2. `merge_dataset`把所有转换后的数据集合并成一个数据集

#### _load_single_dataset
1. 要求数据的文件类型要相同
2. 本地文件先确定文件类型是`FILEEXT2TYPE`支持的
3. 调用 datasets 的`load_dataset`获得数据集
```py
dataset = load_dataset(
    path='json',
    name=None,
    data_dir=None,
    data_files=['/.../demo.json'],
    split='train',
    cache_dir=None,
    token=None,
    num_proc=16,
    trust_remote_code=True,
    streaming=False,
)
# Dataset({
#     features: ['messages', 'label'],
#     num_rows: 300
# })
```
4. 根据`data_args.max_samples`选择最大训练样本数
5. 调用`data.converter.align_dataset`获取格式转换后的训练数据集


### data.converter
#### align_dataset
Align the dataset to a specific format.
1. 调用`get_dataset_converter`获取`DatasetConverter`转换实例
2. 调用`dataset.map`批量执行`DatasetConverter()`得到标准的数据结果
```json
{
    "_prompt": [{"role": "user", "content": "old_prompt"}, {"role": "assistant", "content": "old_response"}, {"role": "user", "content": "prompt\nquery"}],
    "_response": [{"role": "assistant", "content": "response"}],
    "_system": "system",
    "_tools": "tools",
    "_images": null,
    "_videos": null,
    "_audios": null
}
```
kto 的 _response: `[{"role": "assistant", "content": "response"}, {"role": "assistant", "content": ""}]`, `[{"role": "assistant", "content": ""}, {"role": "assistant", "content": "response"}]`
ranking 的 _response: `[{"role": "assistant", "content": "chosen_response"}, {"role": "assistant", "content": "rejected_response"}]`


#### get_dataset_converter
从`DATASET_CONVERTERS`中获取`alpaca`或`sharegpt`对应的`DatasetConverter`


#### SharegptDatasetConverter
1. 把`Role`tag做个映射
2. 检查是否是一问一答



### data.collator
#### MultiModalDataCollatorForSeq2Seq
1. 必须要有`Template`
2. 收集批次中的所有图像、视频和音频数据，记录每种模态的长度信息
3. 当批次中没有图像/音频数据时，创建假的占位数据，在分布式训练（特别是使用 zero3/fsdp 等优化器）时，如果批次中完全没有图像数据，可能会导致处理挂起，通过创建假的图像数据，可以避免这种处理挂起的情况
4. 通过对应多模态模块的`process_messages`把模态占位符转换为 special token 包裹的 token，并组合成 messages，加到batch里的第一个样本
5. 通过`get_mm_inputs`处理多模态输入获取`mm_inputs`，回补`token_type_ids`
6. 把上述处理完的`features`调用父类`DataCollatorForSeq2Seq`得到tokeniz后的`features`
7. 特殊模型的特殊处理
8. 返回`features`


### data.data_utils
#### merge_dataset
1. 只有一个`Dataset`直接返回
2. `data_args.mix_strategy='concat'`策略使用`concatenate_datasets`合并
3. `data_args.mix_strategy='interleave'`策略使用`interleave_datasets`合并


#### split_dataset
1. 如果有`val_size`大小，先用`dataset_dict = dataset.train_test_split(test_size=val_size, seed=seed)`分割
2. 如果指定了`eval_dataset`，再把`eval_dataset`加入
3. 返回`DatasetDict`类
```py
DatasetDict({
    train: Dataset({
        features: ['input_ids', 'attention_mask'],
        num_rows: 71
    })
})
```



## api.app
### create_chat_completion
调用`create_stream_chat_completion_response`返回生成器

## api.chat
### create_stream_chat_completion_response
1. 调用`_process_request`解析请求，分离出`input_messages, system, tools, images, videos, audios`
2. 调用`chat.chat_model.astream_chat`生成`new_token`生成器
3. 调用`_create_stream_chat_completion_chunk`组装返回

### _create_stream_chat_completion_chunk
把`delta`数据组装成 OpenAI API 格式


## chat.chat_model
### astream_chat
调用具体引擎（vllm, sglang, vllm）的`stream_chat`生成`new_toekn`生成器


## VllmEngine
### 初始化
1. `config = load_config(model_args)` 下载模型并返回 AutoConfig 得到的配置 (config.json)
2. `tokenizer_module = load_tokenizer(model_args)` 获取 tokenizer 和 processer
    1. `tokenizer = AutoTokenizer.from_pretrained` 加载 tokenizer (tokenizer_config.json)，注意是 right padding
    2. `processor = AutoProcessor.from_pretrained` 加载 processor (preprocessor_config.json)
3. `self.template = get_template_and_fix_tokenizer(self.tokenizer, data_args)` 获取 template 模板
    1. 根据配置里的`template`参数或`tokenizer.chat_template`得到`Template`实例
    2. 如果配置设置了`tool_format`则构造`template.format_function`和`template.format_tools`的`Formatter`实例
    3. 如果配置里设置了`default_system`则替换系统提示词
    4. 如果配置里`replace_eos=True`则把`eos_token`替换为`stop_words[0]`，再把`stop_words[1:]`加入`additional_special_tokens`
    5. 如果配置了`replace_jinja_template=True`则替换默认的`tokenizer.chat_template`
4. 组装 `engine_args`，然后增加多模态的数量限定，更新`vllm_config`
5. `self.model = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**engine_args))` 获取 vllm 异步模型引擎


### _generate
1. content 前面先加上对应模态的占位符
2. `process_messages`
    1. 验证模型是否支持输入的类型
    2. 验证 message 内容是否匹配
    3. 如果`expand_mm_tokens=True`，则调用`_get_mm_inputs`，分别对每个模态进行正则化，然后调用对应的 processor 处理返回 tensor
    4. 把 content 里对应的占位符换成 special_token 包裹的模板文本
    5. 返回 messages
2. `self.template.encode_oneturn`把一轮对话 encode 成 tensor，把最后一个 token_id 单独拿出来，作为响应 token_id，这里调用`_encode`时按顺序应用：
    1. `format_prefix.apply`
    2. `format_tools.apply`
    3. `format_system.apply`
    4. `format_user.apply`
    5. `format_assistant.apply`
    6. `format_observation.apply`
    7. `format_function.apply`
3. 组装 vllm 生成配置
    1. vllm 不支持`length_penalty`
    2. 如果配置`max_new_tokens`，vllm 的`max_tokens`指的是生成的token，因此`max_tokens = max_new_tokens`
    3. 如果配置了`max_length`，做了一个`max_tokens = max_length - prompt_length`
    4. 设置其他擦样参数
    5. 组装多模态数据
    6. 返回 vllm 生成的 generator


### stream_chat
`async for`异步遍历生成器，返回`delta_text`


## tool_utils
1. function_formatter: 主要就是把 tools 里的内容拼接组装成 tool_text，进而组装成 tools prompts
