

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


train_data = LCSTS("E:\\NLP任务\\生成式任务\\data\\lcsts_tsv\\data1_cutted.txt")
valid_data = LCSTS("E:\\NLP任务\\生成式任务\\data\\lcsts_tsv\\data2.txt")
test_data = LCSTS("E:\\NLP任务\\生成式任务\\data\\lcsts_tsv\\data3.txt")

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
import torch.nn.functional as F
# 1.3)分批处理
max_input_length = 154
max_target_length = 32

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
        batch_targets.append(sample['title'])
    # 对加了提示的原文本分词，编码
    input_ids = pad_sequence([torch.tensor(tokenizer.encode(
                        x,
                        max_length=max_input_length,
                        truncation=True))
                        for x in batch_inputs],
                        batch_first=True,padding_value=tokenizer.pad_token_id)
    # 创建注意力掩码
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    # 对标签编码和动态填充
    labels = pad_sequence([torch.tensor(tokenizer.encode(
                            x,
                            max_length=max_target_length,
                            truncation=True))
                            for x in batch_targets],
                            batch_first=True, padding_value=tokenizer.pad_token_id)

    # 获取最大长度
    max_len = max(input_ids.size(1),labels.size(1))

    # 填充input_ids和labels到相同的长度
    input_ids = F.pad(input_ids,(0,max_len-input_ids.size(1)),value=tokenizer.pad_token_id).to(device)
    attention_mask = F.pad(attention_mask, (0, max_len - attention_mask.size(1)), value=0).to(device)
    labels = F.pad(labels, (0, max_len - labels.size(1)), value=-100).to(device)

    # 将标签中的填充标记替换为 -100
    labels[labels == tokenizer.pad_token_id] = -100


    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels
    }

train_dataloader = DataLoader(train_data,batch_size=16,shuffle=True,collate_fn=collote_fn)
valid_dataloader = DataLoader(valid_data,batch_size=16,shuffle=False,collate_fn=collote_fn)

# 打印测试
batch = next(iter(train_dataloader))
print(batch.keys())
print('batch shape:', {k: v.shape for k, v in batch.items()})