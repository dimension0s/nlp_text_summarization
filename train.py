# 3.模型训练
# 3.1）训练函数

import torch
from tqdm.auto import tqdm
import os
from device import device


def train_loop(dataloader, model, optimizer, lr_scheduler, epoch, total_loss):
    total_loss = 0.
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