import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from configs import LocalTrainConfig
from tse.data import TweetDataset
from tse.models import load_model
from tse.utils import create_folds, AverageMeter, EarlyStopping, set_seed, device, batch_jaccard, cuda_num

train_config = LocalTrainConfig()


def train_fn(data_loader_, model_, optimizer, scheduler):
    model_.train()
    losses = AverageMeter()
    jaccards = AverageMeter()

    tk0 = tqdm(data_loader_, total=len(data_loader_))

    device_ = device()

    for bi, d in enumerate(tk0):
        ids = d["ids"].to(device_, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device_, dtype=torch.long)
        mask = d["mask"].to(device_, dtype=torch.long)
        targets_start = d["targets_start"].to(device_, dtype=torch.long)
        targets_end = d["targets_end"].to(device_, dtype=torch.long)

        model_.zero_grad()
        outputs_start_, outputs_end_, loss = model_(ids=ids, mask=mask, token_type_ids=token_type_ids,
                                                    start=targets_start, end=targets_end)

        loss.backward()
        optimizer.step()
        scheduler.step()

        start_positions, end_positions = model_.to_positions(outputs_start_, outputs_end_)
        jaccard_scores, _ = batch_jaccard(start_positions, end_positions, d)

        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)


def eval_fn(data_loader, model_, epoch, fold):
    model_.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    tweets = []
    target_texts = []
    predicted_texts = []
    sentiments = []
    scores_ = []

    device_ = device()

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"].to(device_, dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device_, dtype=torch.long)
            mask = d["mask"].to(device_, dtype=torch.long)
            targets_start = d["targets_start"].to(device_, dtype=torch.long)
            targets_end = d["targets_end"].to(device_, dtype=torch.long)

            outputs_start_, outputs_end_, loss = model_(ids=ids, mask=mask, token_type_ids=token_type_ids,
                                                        start=targets_start, end=targets_end)

            start_positions, end_positions = model_.to_positions(outputs_start_, outputs_end_)
            jaccard_scores, filtered_outputs = batch_jaccard(start_positions, end_positions, d)
            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

            predicted_texts += filtered_outputs
            scores_ += jaccard_scores
            tweets += d['orig_tweet']
            target_texts += d['orig_selected']
            sentiments += d['sentiment']

    df = pd.DataFrame({'tweet': tweets, 'target_text': target_texts,
                       'predicted_text': predicted_texts, 'sentiment': sentiments, 'scores': scores_})
    file_name = f'debug_{epoch}_{fold}.csv'
    df.to_csv(file_name, index=False, encoding='utf-8')
    print(f'save {file_name}')
    return jaccards.avg


def run(fold):
    dfx = pd.read_csv(train_config.TRAINING_FILE)

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)

    train_dataset = TweetDataset(
        train_config=train_config,
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=train_config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = TweetDataset(
        train_config=train_config,
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )

    valid_data_loader = DataLoader(
        valid_dataset,
        batch_size=train_config.VALID_BATCH_SIZE,
        num_workers=2
    )

    model = load_model(train_config, device())

    num_train_steps = int(len(df_train) / train_config.TRAIN_BATCH_SIZE * train_config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]

    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    es = EarlyStopping(patience=2, mode="max", delta=0.001)
    print(f"Training is Starting for fold={fold}")

    jaccard_ = 0
    for epoch in range(train_config.EPOCHS):
        train_fn(train_data_loader, model, optimizer, scheduler=scheduler)
        jaccard_ = eval_fn(valid_data_loader, model, epoch, fold)
        print(f"Jaccard Score = {jaccard_}")
        es(jaccard_, model, model_path=f"{train_config.WEIGHTS_DIR}/{model.prefix}_model_{fold}_{cuda_num()}.bin")
        if es.early_stop:
            print("Early stopping")
            break
    return jaccard_


if __name__ == '__main__':

    start = time.time()
    set_seed(14)

    create_folds(train_config)

    scores = []
    for fold in range(5):
        print(f'fold {fold}')
        j = run(fold=fold)
        scores.append((fold, j))

    sorted_scores = sorted(scores, key=lambda x: -x[1])
    print([(f, round(j, 4)) for f, j in sorted_scores])
    print(np.mean([j for _, j in sorted_scores]))
    print(f'time: {(time.time()-start)/60} m')