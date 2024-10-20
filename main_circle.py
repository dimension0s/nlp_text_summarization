# 3.3) 主循环

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import AdamW, get_scheduler
import random,torch,os,json
import numpy as np
from collate_fn import model,train_dataloader,valid_dataloader
from train import train_loop
from test import test_loop




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
learning_rate = 2e-5  # 如果增大学习率，可以3e-5,之后再慢慢涨
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

# 第一次训练总结：
# 大致分数为30出头，即一般到中等的边界，官方参照是：简体中文：均值为30.2015
# 其中ROUGE-1 / ROUGE-2 / ROUGE-L：39.4071 / 17.7913 / 33.406
# Chinese (Traditional)：37.1866 / 17.1432 / 31.6184
# 1.5轮epoch，每轮训练时长约1小时12分钟
# 2.非常容易过拟合，但对比原文示例也是一样的情况
# 3.设置的batch_size较小，学习率也有所增加
# 4.后面还有微调空间
# 5.由于目前市面上的transformer预训练模型多用来处理英文，契合中文的模型(bart-base-chinese)较少，
# 因此可以当做一个探索方向；比如可以从强化学习，prompt等方面出发