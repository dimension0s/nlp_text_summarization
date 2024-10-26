# Qwen2-1.5b模型说明：该模型属于自回归生成模型，即Causal Language Model
# 因此没有解码器输入ID，这对数据处理有影响
# 先把代码放上去，初步问题是内存爆了： CUDA out of memory. Tried to allocate 36.00 MiB. GPU
# 后期主要是调参的工作


# 1.数据集处理
# 1.1)加载数据集
import torch,os,random,numpy
from transformers import AutoModelForCausalLM,AutoTokenizer
from torch.utils.data import Dataset,DataLoader

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


train_data = LCSTS("data1_cutted.txt")
valid_data = LCSTS("data2.txt")
test_data = LCSTS("data3.txt")

# 打印测试
print(train_data[0])
print(valid_data[0])
print(test_data[0])
print(len(train_data))
print(len(valid_data))
print(len(test_data))

# device设置
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print(f"device num:{torch.cuda.device_count()}")
    print(f"device name:{torch.cuda.get_device_name()}")
else:
    device = torch.device("cpu")
    print("No GPU available,using the CPU instead.")



# 1.2）调用模型
model_name = "E:/NLP任务/离线模型/Qwen2-1.5b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto")
model = model.to(device)

from torch.nn.utils.rnn import pad_sequence
# 1.3)分批处理
max_length = 128

def collote_fn(batch_samples):
    batch_inputs, batch_targets = [],[]
    for sample in batch_samples:
        text = sample['content']
        # 注意以下提示的设计，它可能是导致输入输出batch_size形状不匹配的主要原因
        messages = [
            {"role": "system", "content": "你是一个文本摘要的专家, 你会接收一段文本, 请将该文本生成摘要。"},
            {"role": "user", "content": text}
        ]
        prompt = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        batch_inputs.append(prompt)
        batch_targets.append(sample['title']+tokenizer.eos_token)

    # 编码：加了提示的原文本
    batch_data = tokenizer(
        batch_inputs,
        max_length=max_length,  # 统一使用相同的长度
        padding='max_length',
        truncation=True,
        return_tensors='pt').to(device)


    # 标签数据
    labels = tokenizer(
        batch_targets,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors="pt")['input_ids'].to(device)

    # 将 <eos> 后的 token 设置为 -100，避免计算损失
    # [1]:取列索引
    # eos_token_id:151645,pad_token_id:151643
    # 在计算交叉熵损失时，pad_token_id不用管，因为CrossEntropyLoss 通常会自动忽略填充标记的损失计算
    # 保险的措施：将 <pad> token 也设置为 -100，计算损失时一并忽略
    # labels[labels == tokenizer.pad_token_id] = -100
    end_token_index = torch.where(labels == tokenizer.eos_token_id )[1]
    for idx, end_idx in enumerate(end_token_index):
        labels[idx][end_idx+1:] = -100
    batch_data['labels'] = labels

    return batch_data

train_dataloader = DataLoader(train_data,batch_size=16,shuffle=True,collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data,batch_size=16,shuffle=False,collate_fn=collote_fn)
test_dataloader = DataLoader(test_data,batch_size=16,shuffle=False,collate_fn=collote_fn)

# 打印测试
batch = next(iter(train_dataloader))
print(batch.keys())
print('batch shape:', {k: v.shape for k, v in batch.items()})
# print(f"eos_token_id: {tokenizer.eos_token_id}")  # eos_token_id:151645
# print(f"pad_token_id:{tokenizer.pad_token_id}")  # 151643
print(batch)

# 3.1）训练函数
from tqdm.auto import tqdm

def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    model.train()
    total = 0

    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for step, batch_data in progress_bar:
        model_inputs = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        labels = batch_data['labels']
        outputs = model(
            input_ids=model_inputs,
            attention_mask=attention_mask,
            labels=labels)
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

# rouge分数标准
# 0-30：低分
# 30-50：中等
# 50-70：高分
# 70以上：优秀

def test_loop(dataloader, model, mode='Valid'):
    assert mode in ['Valid', 'Test']
    model.eval()

    preds, labels = [], []
    for batch_data in dataloader:
        model_inputs = batch_data['input_ids']
        attention_mask = batch_data['attention_mask']
        labels = batch_data['labels']

        with torch.no_grad():
            generated_ids = model.generate(  # 1.生成预测
                model_inputs,
                attention_mask=attention_mask,
                max_new_tokens=512,
                num_beams=4,  # 使用柱搜索
                no_repeat_ngram_size=2, ).cpu().numpy()
        if isinstance(generated_ids, tuple):
            generated_ids = generated_ids[0]
        # 2.对预测解码
        decoded_preds = [
            output_ids[len(input_ids):] for input_ids,output_ids in zip(model_inputs,generated_ids)
        ]
        # 或者：
        # decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        label_tokens = labels.cpu().numpy()
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
import json


def seed_everything(seed=1029):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
seed_everything(42)

learning_rate = 1e-4
epoch_num = 3

optimizer = AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    'linear',
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num * len(train_dataloader))


best_avg_rouge = 0.
for epoch in range(epoch_num):
    print(f'Epoch {epoch + 1}/{epoch_num}\n------------------------------------')
    total_loss =0.  # 重置损失
    total_loss = train_loop(train_dataloader, model, optimizer, lr_scheduler, epoch + 1, total_loss)
    valid_rouge = test_loop(valid_dataloader, model, mode='Valid')
    rouge_avg = valid_rouge['avg']

    # 保存最佳模型权重
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
            f.write('\n')  # 确保在文件关闭前执行写入操作,添加换行符便于文件读取


# 4.模型测试
model.load_state_dict(torch.load('***'))
model.eval()

with torch.no_grad():
    print('evaluating on test set...')
    sources, preds, labels = [], [], []
    for batch_data in test_dataloader:
        batch_data = batch_data.to(device)
        generated_ids = model.generate(  # 1.生成预测
            batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
            max_length=max_length,
            num_beams=4,
            no_repeat_ngram_size=2).cpu().numpy()
        if isinstance(generated_tokens, tuple):
            generated_tokens = generated_tokens[0]
        # 2.对预测解码
        decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

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
