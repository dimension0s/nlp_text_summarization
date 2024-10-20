# 4.模型测试
import random,torch,os,json
import numpy as np
from collate_fn import model,train_dataloader,valid_dataloader,tokenizer
from train import train_loop
from test import test_loop
from device import device
from data_process import LCSTS
from torch.utils.data import DataLoader
from test import rouge

test_data = LCSTS("data/data3.txt")  # E:\\NLP任务\\生成式任务\\data\\lcsts_tsv\\data3.txt
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False, collate_fn=collote_fn)


model.load_state_dict(torch.load('epoch_1_valid_rouge_6.6667_model_weights.bin'))

model.eval()

max_input_length = 512
max_target_length = 32  # 之前是64

with torch.no_grad():
    print('evaluating on test set...')
    sources, preds, labels = [], [], []
    for batch_data in test_dataloader:
        batch_data = batch_data.to(device)
        generated_tokens = model.generate(  # 1.生成预测
            batch_data['input_ids'],
            attention_mask=batch_data['attention_mask'],
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

