from functools import reduce

import numpy as np
import pandas as pd
import tokenizers
import torch
import torch.nn as nn
import transformers
from tqdm.autonotebook import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from tse.data import TweetDataset
from tse.models import TweetModel
from tse.utils import create_folds, jaccard, AverageMeter, bert_output_to_string, EarlyStopping, set_seed, device
from torch.utils.data import DataLoader


# class KaggleTrainConfig:
#     MAX_LEN = 192
#     TRAIN_BATCH_SIZE = 32
#     VALID_BATCH_SIZE = 8
#     EPOCHS = 5
#     BERT_PATH = "../input/roberta-base/"
#     MODEL_PATH = "model.bin"
#     TRAINING_FILE = "train_folds.csv"
#     TEST_FILE = "../input/tweet-sentiment-extraction/test.csv"
#     TRAIN_FILE = "../input/tweet-sentiment-extraction/train.csv"
#     SAMPLE_FILE = "../input/tweet-sentiment-extraction/sample_submission.csv"
#     TOKENIZER = tokenizers.ByteLevelBPETokenizer(
#         vocab_file=f"{BERT_PATH}/vocab.json",
#         merges_file=f"{BERT_PATH}/merges.txt",
#         lowercase=True,
#         add_prefix_space=True
#     )


class LocalTrainConfig:
    MAX_LEN = 192
    TRAIN_BATCH_SIZE = 32
    VALID_BATCH_SIZE = 8
    EPOCHS = 5
    BERT_PATH = "/data/tweet-sentiment-extraction/roberta-base"
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "/data/tweet-sentiment-extraction/train_folds.csv"
    TEST_FILE = "/data/tweet-sentiment-extraction/test.csv"
    TRAIN_FILE = "/data/tweet-sentiment-extraction/train.csv"
    SAMPLE_FILE = "/data/tweet-sentiment-extraction/sample_submission.csv"
    TOKENIZER = tokenizers.ByteLevelBPETokenizer(
        vocab_file=f"{BERT_PATH}/vocab.json",
        merges_file=f"{BERT_PATH}/merges.txt",
        lowercase=True,
        add_prefix_space=True
    )


train_config = LocalTrainConfig


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss


def _batch_jaccard(outputs_start_, outputs_end_, batch):
    orig_tweet = batch['orig_tweet']
    orig_selected = batch['orig_selected']
    sentiment = batch['sentiment']
    offsets = batch['offsets']

    outputs_start_ = torch.softmax(outputs_start_, dim=1).cpu().detach().numpy()
    outputs_end_ = torch.softmax(outputs_end_, dim=1).cpu().detach().numpy()
    jaccard_scores = []
    filtered_outputs_ = []
    for px, tweet in enumerate(orig_tweet):
        selected_tweet = orig_selected[px]
        tweet_sentiment = sentiment[px]
        f_output = bert_output_to_string(tweet, tweet_sentiment, np.argmax(outputs_start_[px, :]),
                                         np.argmax(outputs_end_[px, :]), offsets[px])

        jaccard_score = jaccard(selected_tweet.strip(), f_output.strip())
        jaccard_scores.append(jaccard_score)
        filtered_outputs_.append(f_output)
    return jaccard_scores, filtered_outputs_


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
        outputs_start_, outputs_end_ = model_(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs_start_, outputs_end_, targets_start, targets_end)

        loss.backward()
        optimizer.step()
        scheduler.step()

        jaccard_scores, _ = _batch_jaccard(outputs_start_, outputs_end_, d)

        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)


def eval_fn(data_loader, model, epoch, fold):
    model.eval()
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

            outputs_start_, outputs_end_ = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(outputs_start_, outputs_end_, targets_start, targets_end)

            jaccard_scores, filtered_outputs = _batch_jaccard(outputs_start_, outputs_end_, d)
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

    model_config = transformers.BertConfig.from_pretrained(train_config.BERT_PATH)
    model_config.output_hidden_states = True
    model = TweetModel(train_config, conf=model_config)
    model.to(device())

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
        es(jaccard_, model, model_path=f"model_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break
    return jaccard_


if __name__ == '__main__':

    set_seed(14)

    create_folds(train_config)

    model_config = transformers.BertConfig.from_pretrained(train_config.BERT_PATH)
    model_config.output_hidden_states = True

    scores = []
    for fold in range(5):
        print(f'fold {fold}')
        j = run(fold=fold)
        scores.append((fold, j))

    sorted_scores = sorted(scores, key=lambda x: -x[1])
    print(sorted_scores)

    models = []
    for fold, _ in sorted_scores[:5]:
        print(f"loading model_{fold}.bin")
        model = TweetModel(train_config, conf=model_config)
        model.to(device())
        model.load_state_dict(torch.load(f"model_{fold}.bin"))
        model.eval()
        models.append(model)

    df_test = pd.read_csv(train_config.TEST_FILE)
    df_test.loc[:, "selected_text"] = df_test.text.values

    final_output = []

    test_dataset = TweetDataset(
        train_config=train_config,
        tweet=df_test.text.values,
        sentiment=df_test.sentiment.values,
        selected_text=df_test.selected_text.values
    )

    data_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=train_config.VALID_BATCH_SIZE,
        num_workers=1
    )

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"].to(device(), dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device(), dtype=torch.long)
            mask = d["mask"].to(device(), dtype=torch.long)

            starts_list = []
            ends_list = []

            for model in models:
                outputs_start, outputs_end = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                starts_list.append(outputs_start)
                ends_list.append(outputs_end)

            outputs_start = (starts_list[0] + starts_list[1] + starts_list[2] + starts_list[3] + starts_list[4]) / len(starts_list)
            outputs_end = (ends_list[0] + ends_list[1] + ends_list[2] + ends_list[3] + ends_list[4]) / len(ends_list)

            _, filtered_outputs = _batch_jaccard(outputs_start, outputs_end, d)
            for s in filtered_outputs:
                final_output.append(s)

    # post-process trick:
    # Note: This trick comes from: https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/140942
    # When the LB resets, this trick won't help
    def post_process(selected):
        return " ".join(set(selected.lower().split()))


    sample = pd.read_csv(train_config.SAMPLE_FILE)
    sample.loc[:, 'selected_text'] = final_output
    sample.selected_text = sample.selected_text.map(post_process)
    sample.to_csv("submission.csv", index=False)