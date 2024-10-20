# 3.2) 测试函数：在验证环节中添加评价指标，通常是准确率，召回率，f1,在此任务中使用rouge，包含以上指标
# 注：解码的原因：rouge评价体系所需序列源是文本，而非数字编码
from rouge import Rouge
import random,torch
import numpy as np
from device import device
from collate_fn import max_target_length, max_input_length,tokenizer

rouge = Rouge()
# 标准：
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
                num_beams=4,  # 使用柱搜索
                no_repeat_ngram_size=2).cpu().numpy()
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
