# 注：以下代码为初步尝试，还没debug，主要用于梳理逻辑，请谨慎使用

# 1.数据处理

from torch.utils.data import Dataset, DataLoader
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

class MyData(Dataset):
    def __init__(self,data_file):
        self.data = self.load_data(data_file)

    def load_data(self,data_file):
        Data = {}
        with open(data_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                # 将行按空格拆分为输入部分和输出部分
                input_str, output_str = line.split('!=!')

                # 分别将两部分解析为 JSON
                inputs = json.loads(input_str)
                outputs = json.loads(output_str)
                # 提取事件内容和摘要
                contents = [event['content'] for event in inputs['events']]
                event_summs = [summary['event-summarization'] for summary in outputs['summarizations']]

            Data[idx] = {
                'doc': inputs['doc'],
                'contents': contents,  # 列表格式
                'summarizations': event_summs,  # 列表格式
            }

        return Data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


train_data = MyData("Qwen/example_data.txt")
valid_data = MyData("Qwen/example_data.txt")
print(train_data[0])
print(len(train_data))

model_name = "Qwen/Qwen2.5-72B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map="auto")

# 预处理：将事件转化为对话输入
def create_event_prompt(doc,event_content):
    prompt = "你是一个擅长文本摘要的专家，你会收到一段文本，它篇幅较长，蕴涵较多信息，我将为你提供相应的辅助片段，\
    请根据原文和辅助片段为该文本生成摘要"
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"这是原文：{doc}，请为它生成摘要"},
        {"role": "assistant", "content": f"这是辅助片段：{ event_content}"}
    ]
    return messages

# 分批处理
def collote_fn(batch_samples,tokenizer):
    batch_inputs,batch_summs = [],[]
    for sample in batch_samples:
        # 原文分批，分词
        doc = sample['doc']

        # 提示分批，分词
        for content, summary in zip(sample['contents'], sample['summarizations']):
            # 生成对话模板
            messages = create_event_prompt(doc,content)
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                )
            model_inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )['input_ids']
            # 添加到输入批次
            batch_inputs.append(model_inputs)

            # 答案分批，分词，后面用来和模型生成做对比
            labels = tokenizer(
                summary,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )['input_ids']
            batch_summs.append(labels)


    return batch_inputs, batch_summs

train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=collote_fn)
# 假设有验证集，它和训练集格式一样，因此采取同样的预处理
valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False, collate_fn=collote_fn)

# 打印测试
data_inputs, summs = next(iter(train_dataloader))

# device设置
import torch

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"device num:{torch.cuda.device_count()}")
    print(f"device name:{torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("No GPU available,using the CPU instead.")

model = model.to(device)

from tqdm.auto import tqdm
import numpy as np
import numpy, os

# 训练函数
def train_loop(dataloader,model,optimizer,lr_scheduler,epoch):
    total_loss =0.
    total = 0

    model.train()

    progress_bar = tqdm(enumerate(dataloader),total=len(dataloader))
    for step, batch_data in progress_bar:
        batch_inputs, batch_summs = batch_data
        batch_inputs = batch_inputs.to(device)
        batch_summs = batch_summs.to(device)

        # 生成摘要预测
        output_pred = model(batch_inputs['input_ids'])

        loss = output_pred.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss/(step+1)
        progress_bar.set_description(f'loss:{avg_loss:>7f}')
        return avg_loss

# 测试函数
from rouge import Rouge
rouge = Rouge()

def test_loop(dataloader,model,mode='Valid'):
    assert mode in ['Valid', 'Test']
    model.eval()

    preds, labels = [], []
    for batch_data in dataloader:
        batch_inputs, batch_summs = batch_data
        batch_inputs = batch_inputs.to(device)
        batch_summs = batch_summs.to(device)

        with torch.no_grad():
            # 1.生成预测
            generated_ids = model.generate(**batch_inputs, max_new_tokens=512)

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(batch_inputs, generated_ids)
            ]

            # 对预测解码
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            # 2.对标签解码(能否这么写看分词器是否允许,假设模型允许这样处理)
            label_tokens = batch_summs.cpu().numpy()
            label_tokens = np.where(labels != -100, label_tokens, tokenizer.pad_token_id)
            decoded_labels = tokenizer.batch_decode(label_tokens, skip_special_tokens=True)

            # 用空格连接结果用于匹配rouge格式
            preds += [' '.join(pred.strip()) for pred in response]
            labels += [' '.join(label.strip()) for label in decoded_labels]

        scores = rouge.get_scores(hyps=preds, refs=labels, avg=True)
        result = {key: value['f']*100 for key, value in scores.items()}
        result['avg'] = np.mean(list(result.values()))
        print(f"{mode} RougeL:{result['rouge-l']:>0.2f}\n")
        return result

# 主循环

from transformers import AdamW,get_scheduler
import random


def seed_everything(seed=1029):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(42)

learning_rate = 2e-4
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
        f.write('\n')









