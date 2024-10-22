# 先展示Chinese BART基本用法：
from transformers import BertTokenizer, BartForConditionalGeneration
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = BartForConditionalGeneration.from_pretrained("MODEL_NAME")
print(model)
# 接下来根据以上用法实现具体示例：
# 1.构建数据集：LCSTS，训练集包含200万条数据集，为减少训练时间，将其截取到30万：data1_cutted.txt
# 1.1）加载数据集

import os
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 削减训练数据集，减少运行时间
import json

input_filename = "data1.txt"  # 原文件
output_filename = "data1_cutted.txt"  # 截取新文件

lines_to_extract = 300000

with open(input_filename, 'rt', encoding='utf-8') as in_file, open(output_filename, 'wt', encoding='utf-8') as out_file:
    line_count = 0
    for line in in_file:
        if line_count >= lines_to_extract:
            break
        out_file.write(line)
        line_count += 1

print(f'Extracted {line_count} lines and saved to {output_filename}')



class LCSTS(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, 'rt', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                items = line.strip().split('!=!')
                assert len(items) == 2
                Data[idx] = {
                    'title': items[0],
                    'content': items[1],
                }
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data=LCSTS("data1_cutted.txt")
valid_data=LCSTS("data2.txt")
test_data=LCSTS("data3.txt")

# 1.2)分批，分词，编码
from transformers import BertTokenizer,BartForConditionalGeneration,Text2TextGenerationPipeline
import sentencepiece as spm

model_path = "fnlp/bart-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_path)# ,local_files_only=True
model = BartForConditionalGeneration.from_pretrained(model_path) # ,local_files_only=True
# text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
# text2text_generator("北京是[MASK]的首都", max_length=50, do_sample=False)
max_input_length = 512
max_target_length = 64

# 分批函数
def collote_fn(batch_samples):
    batch_inputs,batch_targets = [],[]
    for sample in batch_samples:
        batch_inputs.append(sample['content'])
        batch_targets.append(sample['title'])
    # 源数据
    batch_data = tokenizer(
        batch_inputs,
        padding=True,
        max_length=max_input_length,
        truncation=True,
        return_tensors='pt')
    # 标签数据
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            padding=True,
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt")['input_ids']
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        end_token_index = torch.where(labels==tokenizer.eos_token_id)[1] # [1]:取列索引
        for idx,end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data

train_dataloader=DataLoader(train_data,batch_size=16,shuffle=True,collate_fn=collote_fn)
valid_dataloader=DataLoader(valid_data,batch_size=16,shuffle=False,collate_fn=collote_fn)
# 注：如果内存不足，batch_size改成16
# 在租卡NVIDIA GeForce RTX 4090 D上出现的错误：
"""
OutOfMemoryError: CUDA out of memory.
Tried to allocate 352.00 MiB (GPU 0; 23.64 GiB total capacity; 22.49 GiB already allocated;
183.75 MiB free; 23.05 GiB reserved in total by PyTorch) If reserved memory is >> allocated memory try setting max_split_size_mb to avoid fragmentation.
See documentation for Memory Management and PYTORCH_CUDA_ALLOC_CONF
"""

# device设置
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"device num:{torch.cuda.device_count()}")
    print(f"device name:{torch.cuda.get_device_name()}")

else:
    device = torch.device("cpu")
    print("No GPU available,using the CPU instead. ")

model = model.to(device)

# 3.模型训练
# 3.1）训练函数

import torch
from tqdm.auto import tqdm
import os


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    model.train()
    model = model.to(device)
    total = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch_data in progress_bar:
        batch_data = batch_data.to(device)
        # 注意：BertTokenizer不支持token_type_ids，因此这里将具体元素添加进去，而不使用**
        # 而且，有了labels,loss损失才能反向计算
        outputs = model(input_ids=batch_data['input_ids'], labels=batch_data['labels'])
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        progress_bar.set_description(f'loss:{avg_loss:>7f}')
    return total_loss


# 3.2) 测试函数：在验证环节中添加评价指标，通常是准确率，召回率，f1,在此任务中使用rouge，包含以上指标
# 注：解码的原因：rouge评价体系所需序列源是文本，而非数字编码
from rouge import Rouge
import random
import numpy as np

rouge = Rouge()


# 0-30：低分
# 30-50：中等
# 50-70：高分
# 70以上：优秀

def test_loop(dataloader, model, mode='Valid'):
    assert mode in ['Valid', 'Test']
    model = model.to(device)
    model.eval()

    preds, labels = [], []
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            generated_tokens = model.generate(  # 1.生成预测
                batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                max_length=max_target_length,
                num_beams=beam_size,  # 使用柱搜索
                no_repeat_ngram_size=no_repeat_ngram_size, ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        # 2.对预测解码
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        label_tokens = batch_data['labels'].cpu().numpy()
        label_tokens = np.where(labels != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        # 用空格连接结果用于匹配rouge格式
        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]

    scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)
    result = {key: value['f'] * 100 for key, value in scores.items()}
    result['avg'] = np.mean(list(result.values()))
    print(f"{mode} Rouge1:{result['rouge-1']:>0.2f} Rouge2:{result['rouge-2']:>0.2f} \
            RougeL:{result['rouge-l']:>0.2f}\n")
    return result


# 3.3) 主循环

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AdamW, get_scheduler
import random


def seed_everything(seed=1029):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(5)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using {device} device')

beam_size = 4
no_repeat_ngram_size = 2
learning_rate = 2e-5
epoch_num = 5

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num * len(train_dataloader))

total_loss = 0.
best_avg_rouge = 0.
for epoch in range(epoch_num):
    print(f'Epoch {epoch + 1}/{epoch_num}\n------------------------------------')
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch + 1, total_loss)
    valid_rouge = test_loop(valid_dataloader, model, mode='Valid')
    rouge_avg = valid_rouge['avg']
    if rouge_avg > best_avg_rouge:
        best_avg_rouge = rouge_avg
        print('saving new weights...\n')
        torch.save(model.state_dict(),
                   f'epoch_{epoch + 1}_valid_rouge_{rouge_avg:0.4f}_model_weights.bin')
        # 打印验证集评价指标
        print(f'rouge_avg:{rouge_avg}')

        # 将验证集指标记录到文件
        with open('rouge_avg.json', 'a') as f:
            json.dump({'epoch': epoch + 1, 'rouge': rouge_avg}, f)
            f.write('\n')  # 确保在文件关闭前执行写入操作

# 4.模型测试
test_data = LCSTS("data/data3.txt")  
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collote_fn)

import json

model.load_state_dict(torch.load('epoch_1_valid_rouge_6.6667_model_weights.bin'))

model.eval()

max_input_length = 512
max_target_length = 64

with torch.no_grad():
    print('evaluating on test set...')
    sources, preds, labels = [], [], []
    for batch_data in test_dataloader:
        batch_data = batch_data.to(device)
        generated_tokens = model.generate(  # 1.生成预测
            batch_data['input_ids'],
            attention_mak=batch_data['attention_mask'],
            max_length=max_target_length,
            num_beams=4,
            no_repeat_ngram_size=2).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        # 2.对预测解码
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        # 转换标签并解码
        label_tokens = batch_data['labels'].cpu().numpy()
        label_tokens = np.where(labels != -100, label_tokens, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

        decoded_sources = tokenizer.batch_decode(
            batch_data['input_ids'].cpu().numpy(),
            skip_special_tokens=True,
            use_source_tokenizer=True)

        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]
        sources += [' '.join(source.strip()) for source in decoded_sources]
    scores = rouge.get_scores(
        hyps=preds, refs=labels, avg=True)
    rouges = {key: value['f'] * 100 for key, value in scores.items()}
    rouges['avg'] = np.mean(list(rouges.values()))
    print(
        f"Test Rouge1: {rouges['rouge-1']:>0.2f} Rouge2: {rouges['rouge-2']:>0.2f} RougeL: {rouges['rouge-l']:>0.2f}\n")
    results = []
    for source, pred, label in zip(sources, preds, labels):
        results.append({
            'document': source,
            'prediction': pred,
            'summarization': label
        })
    with open('test_data_pred.json', 'wt', encoding='utf-8') as f:
        for example_result in results:
            f.write(json.dumps(example_result, ensure_ascii=False) + '\n')




# 显存分配：
# 情况1.如果你是在 Python 脚本或 Jupyter Notebook 中运行的，需要通过 os.environ 来设置环境变量。
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

# 情况2.如果你是在命令行（例如终端或 Anaconda Prompt）中运行 Python 脚本，你可以在运行脚本之前设置这个环境变量：
# bash:
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# python your_script.py

# 情况3.如果你是在 Windows 环境下运行，可以使用 set 命令：
# set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
# python your_script.py

# 训练总结：
# 1.训练5轮，每轮训练约25分钟
# 2.租卡参数：NVIDIA GeForce RTX 4090 D
# 3.缓解内存不足：看上面，削减batch_size至16，如果追求大批次，那么16是该资源前提下的下界
# 模型训练后通常建议手动释放内存，不建议自动释放，手动调节更可控
# del model
# del optimizer
# del dataloader
