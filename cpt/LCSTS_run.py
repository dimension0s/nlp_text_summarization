# 基于LCSTS数据集的应用示例
# 在该示例中，展示了从数据集处理到模型训练等的所有步骤，方便一并阅览
# 注意：在使用cpt模型时，版本要匹配：pytorch==1.8.1，transformers==4.4.1

# 使用cpt模型的演示：
# modeling_cpt.py来源：https://github.com/fastnlp/CPT/blob/master/finetune/modeling_cpt.py
from modeling_cpt import CPTForConditionalGeneration
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("MODEL_NAME")
model = CPTForConditionalGeneration.from_pretrained("MODEL_NAME")
print(model)
# cpt-base和cpt-large均可，使用方法同理

import os
import numpy as np
import json,torch
from torch.utils.data import DataLoader,Dataset

# 1.构建数据集
# 1.1）加载数据集
max_dataset_size = 300000

class LCSTS(Dataset):
    def __init__(self,data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        Data = []  # 原来是字典，这次改成列表,更安全
        with open(data_file,'rt',encoding='utf') as f:
            for idx,line in enumerate(f):
                if idx >= max_dataset_size:
                    break
                items = line.strip().split('!=!')
                assert len(items) == 2
                Data.append({'title':items[0],'content':items[1]})
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

train_data = LCSTS("data1.txt")
valid_data = LCSTS("data2.txt")
test_data = LCSTS("data3.txt")

# 1.2)分批，分词，编码
from transformers import BertTokenizer
from modeling_cpt import CPTForConditionalGeneration

model_path = "fnlp/cpt-base"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = CPTForConditionalGeneration.from_pretrained(model_path)

max_input_length = 1024
max_target_length = 512

# 分批函数
def collote_fn(batch_samples):
    batch_inputs, batch_targets = [], []
    for sample in batch_samples:
        batch_inputs.append(sample['content'])
        batch_targets.append(sample['title'])
    batch_data = tokenizer(
        batch_inputs,
        padding=False,
        max_length=max_input_length,
        truncation=True,
        return_tensors='pt'
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch_targets,
            padding=False,
            max_length=max_target_length,
            truncation=True,
            return_tensors='pt'
        )['input_ids']
        batch_data['decoder_input_ids'] = model.prepare_decoder_input_ids_from_labels(labels)
        # nonzero方法用于获取所有满足条件的索引:
        end_token_index = torch.where(labels==tokenizer.eos_token_id).nonzero(as_tuple=True)[1]
        for idx,end_idx in enumerate(end_token_index):
            labels[idx][end_idx+1:] = -100
        batch_data['labels'] = labels
    return batch_data

train_dataloader = DataLoader(train_data,batch_size=8,shuffle=True,collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data,batch_size=8,shuffle=False,collate_fn=collote_fn)

# 打印一个批次数据用于测试
batch=next(iter(train_dataloader))
print(batch.keys())
print('batch shape:',{k:v.shape for k,v in batch.items()})
print(batch)

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
from tqdm.auto import tqdm
import os


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    model.train()

    total = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch_data in progress_bar:
        batch_data = batch_data.to(device)
        outputs = model(**batch_data)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        progress_bar.set_description(f'loss:{avg_loss:>7f}')
    return avg_loss

# 3.2) 测试函数：在验证环节中添加评价指标，通常是准确率，召回率，f1,在此任务中使用rouge，包含以上指标
# 注：解码的原因：rouge评价体系所需序列源是文本，而非数字编码
from rouge import Rouge
import random
import numpy as np
import numpy

rouge = Rouge()
def test_loop(dataloader,model,mode='Valid'):
    assert mode in ['Valid','Test']
    model.eval()

    preds,labels = [],[]
    for batch_data in dataloader:
        batch_data = batch_data.to(device)
        with torch.no_grad():
            # 1.生成预测
            generated_tokens = model.generate(
                batch_data['input_ids'],
                attention_mask=batch_data['attention_mask'],
                max_length=max_target_length,
                num_beams=4,  # 使用柱搜索
                no_repeat_ngram_size=2,
            ).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        # 2.对预测解码
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        label_tokens = batch_data['labels'].cpu().numpy()
        label_tokens = np.where(labels != -100, label_tokens,tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(label_tokens,skip_special_tokens=True)

        # 用空格连接结果用于匹配rouge格式
        preds += [' '.join(pred.strip()) for pred in decoded_preds]
        labels += [' '.join(label.strip()) for label in decoded_labels]

    scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)
    result = {key: value['f']*100 for key, value in scores.items()}
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







