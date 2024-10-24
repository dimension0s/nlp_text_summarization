# Qwen/Qwen2.5-72B-Instruct 模型的使用方法示例(from hugging face)
# 注：transformers版本不得低于4.37.0

import transformers
print(transformers.__version__)  # 4.44.1

# 1.导入必要模块，加载模型和分词器
from transformers import AutoModelForCausalLM,AutoTokenizer

model_name = "Qwen/Qwen2.5-72B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype='auto',   # 根据系统自动选择最优的数据类型
        device_map='auto',  # 模型的各部分自动分配到可用的GPU或CPU上
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2.构建对话消息
# prompt:用户输入的请求，要求模型生成一个关于大型语言模型的简短介绍
prompt = "Give me a short introduction to large language model."
# 1.role: system 定义了模型的身份,用于提供对话背景或设定任务情境
# 2.role: user 表示用户发出的请求，即生成大型语言模型的简介
messages = [
    {"role": "system", "content": "You are Qwen ,created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# 3.生成对话模板
text = tokenizer.apply_chat_template(  # 将messages转换成模型可以理解的模板文本格式
        messages,
        tokenize=False,
        add_generation_prompt=True  # 会自动为生成任务添加必要的生成提示信息
)

# 4.编码输入文本并将其转化为模型张量
#  .to(model.device) 将生成的张量移动到与模型相同的设备（CPU或GPU）上
model_inputs = tokenizer([text],return_tensors="pt").to(model.device)

# 5.生成响应
# model.generate(...):使用模型生成响应。输入为前面准备好的模型输入数据
# max_new_tokens=512:指定最多生成512个新token
generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=512
)

# 6.去掉输入文本部分的token
# 在生成的generated_ids中，模型的输出包含了输入的token（因为是自回归模型，生成过程中会将输入包括在内），
# 所以通过这一步去掉模型生成结果中的输入部分，只保留生成的内容

generated_ids = [
    output_ids[len(input_ids):] for input_ids,output_ids in zip(model_inputs,generated_ids)
]

# 7.解码生成的token为文本
# tokenizer.batch_decode(...) 将生成的token序列解码回人类可读的文本
# skip_special_tokens=True 确保跳过任何特殊的控制符（如结束符<|endoftext|>等）
# [0] 取出解码结果的第一项（因为此处只生成了一条响应）
response = tokenizer.batch_decode(generated_ids,skip_special_tokens=True)[0]
