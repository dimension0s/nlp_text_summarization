# 1.2) 分批，分词，编码

from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
import sentencepiece as spm
from device import device
import torch
from data_process import train_data,valid_data,test_data
from torch.utils.data import DataLoader

model_path = "csebuetnlpmT5_multilingual_XLSum"
# 这里使用快速分词器
tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path,local_files_only=True)
model = model.to(device)

max_input_length = 512
max_target_length = 64

# 分批函数
def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['content'])
        batch_targets.append(sample['title'])
    batch_data = tokenizer(
        batch_inputs,
        padding=True,
        max_length=max_input_length,
        truncation=True,
        return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            padding=True,
            max_length=max_target_length,
            truncation=True,
            return_tensors="pt")['input_ids']
        batch_data['decoder_input_ids']=model.prepare_decoder_input_ids_from_labels(labels)
        # [1]:取列索引
        end_token_index = torch.where(labels==tokenizer.eos_token_id)[1]
        # 对于非文本内容替换成-100，为了后面的损失计算
        for idx, end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data

train_dataloader=DataLoader(train_data,batch_size=8,shuffle=True,collate_fn=collote_fn)
valid_dataloader=DataLoader(valid_data,batch_size=8,shuffle=False,collate_fn=collote_fn)

# 打印一个批次数据

batch=next(iter(train_dataloader))
print(batch.keys())
print('batch shape:',{k:v.shape for k,v in batch.items()})
print(batch)