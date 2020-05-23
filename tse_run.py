import torch
import pandas as pd
import torch.nn as nn
import numpy as np

import transformers
import tokenizers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm.autonotebook import tqdm
from tse.data import TweetDataset
from tse.models import TweetModel
from tse.utils import create_folds, jaccard, AverageMeter, bert_output_to_string, EarlyStopping

# class KaggleTrainConfig:
#     MAX_LEN = 128
#     TRAIN_BATCH_SIZE = 64
#     VALID_BATCH_SIZE = 16
#     EPOCHS = 5
#     BERT_PATH = "../input/bert-base-uncased/"
#     MODEL_PATH = "model.bin"
#     TRAINING_FILE = "train_folds.csv"
#     TEST_FILE = "../input/tweet-sentiment-extraction/test.csv"
#     TRAIN_FILE = "../input/tweet-sentiment-extraction/train.csv"
#     SAMPLE_FILE = "../input/tweet-sentiment-extraction/sample_submission.csv"
#     TOKENIZER = tokenizers.BertWordPieceTokenizer(
#         f"{BERT_PATH}/vocab.txt",
#         lowercase=True
#     )


class LocalTrainConfig:
    MAX_LEN = 128
    TRAIN_BATCH_SIZE = 64
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    BERT_PATH = "/data/tweet-sentiment-extraction/bert-base-uncased"
    MODEL_PATH = "model.bin"
    TRAINING_FILE = "/data/tweet-sentiment-extraction/train_folds.csv"
    TEST_FILE = "/data/tweet-sentiment-extraction/test.csv"
    TRAIN_FILE = "/data/tweet-sentiment-extraction/train.csv"
    SAMPLE_FILE = "/data/tweet-sentiment-extraction/sample_submission.csv"
    TOKENIZER = tokenizers.BertWordPieceTokenizer(
        f"{BERT_PATH}/vocab.txt",
        lowercase=True
    )


train_config = LocalTrainConfig


def loss_fn(start_logits, end_logits, start_positions, end_positions):
    loss_fct = nn.CrossEntropyLoss()
    start_loss = loss_fct(start_logits, start_positions)
    end_loss = loss_fct(end_logits, end_positions)
    total_loss = (start_loss + end_loss)
    return total_loss


def _batch_jaccard(outputs_start, outputs_end, batch):
    orig_tweet = batch['orig_tweet']
    orig_selected = batch['orig_selected']
    sentiment = batch['sentiment']
    offsets = batch['offsets']

    outputs_start = torch.softmax(outputs_start, dim=1).cpu().detach().numpy()
    outputs_end = torch.softmax(outputs_end, dim=1).cpu().detach().numpy()
    jaccard_scores = []
    filtered_outputs = []
    for px, tweet in enumerate(orig_tweet):
        selected_tweet = orig_selected[px]
        tweet_sentiment = sentiment[px]
        f_output = bert_output_to_string(tweet, tweet_sentiment, np.argmax(outputs_start[px, :]),
                                         np.argmax(outputs_end[px, :]), offsets[px])

        jaccard_score = jaccard(selected_tweet.strip(), f_output.strip())
        jaccard_scores.append(jaccard_score)
        filtered_outputs.append(f_output)
    return jaccard_scores, filtered_outputs


def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    losses = AverageMeter()
    jaccards = AverageMeter()

    tk0 = tqdm(data_loader, total=len(data_loader))

    for bi, d in enumerate(tk0):

        ids = d["ids"].to(device, dtype=torch.long)
        token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
        mask = d["mask"].to(device, dtype=torch.long)
        targets_start = d["targets_start"].to(device, dtype=torch.long)
        targets_end = d["targets_end"].to(device, dtype=torch.long)

        model.zero_grad()
        outputs_start, outputs_end = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)

        loss.backward()
        optimizer.step()
        scheduler.step()

        jaccard_scores, _ = _batch_jaccard(outputs_start, outputs_end, d)

        jaccards.update(np.mean(jaccard_scores), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)


def eval_fn(data_loader, model, device, epoch, fold):
    model.eval()
    losses = AverageMeter()
    jaccards = AverageMeter()

    tweets = []
    target_texts = []
    predicted_texts = []
    sentiments = []
    scores = []

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"].to(device, dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device, dtype=torch.long)
            targets_start = d["targets_start"].to(device, dtype=torch.long)
            targets_end = d["targets_end"].to(device, dtype=torch.long)

            outputs_start, outputs_end = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            loss = loss_fn(outputs_start, outputs_end, targets_start, targets_end)

            jaccard_scores, filtered_outputs = _batch_jaccard(outputs_start, outputs_end, d)
            jaccards.update(np.mean(jaccard_scores), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, jaccard=jaccards.avg)

            predicted_texts += filtered_outputs
            scores += jaccard_scores
            tweets += d['orig_tweet']
            target_texts += d['orig_selected']
            sentiments += d['sentiment']

    df = pd.DataFrame({'tweet': tweets, 'target_text': target_texts,
                       'predicted_text': predicted_texts, 'sentiment': sentiments, 'scores': scores})
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

    train_data_loader = torch.utils.data.DataLoader(
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

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=train_config.VALID_BATCH_SIZE,
        num_workers=2
    )

    device = torch.device("cuda")
    model_config = transformers.BertConfig.from_pretrained(train_config.BERT_PATH)
    model_config.output_hidden_states = True
    model = TweetModel(train_config, conf=model_config)
    model.to(device)

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

    es = EarlyStopping(patience=2, mode="max")
    print(f"Training is Starting for fold={fold}")

    # I'm training only for 3 epochs even though I specified 5!!!
    for epoch in range(3):
        train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)
        jaccard = eval_fn(valid_data_loader, model, device, epoch, fold)
        print(f"Jaccard Score = {jaccard}")
        es(jaccard, model, model_path=f"model_{fold}.bin")
        if es.early_stop:
            print("Early stopping")
            break


if __name__ == '__main__':

    create_folds(train_config)

    print('fold 0')
    run(fold=0)

    print('fold 1')
    run(fold=1)

    print('fold 2')
    run(fold=2)

    print('fold 3')
    run(fold=3)

    print('fold 4')
    run(fold=4)

    df_test = pd.read_csv(train_config.TEST_FILE)
    df_test.loc[:, "selected_text"] = df_test.text.values

    device = torch.device("cuda")
    model_config = transformers.BertConfig.from_pretrained(train_config.BERT_PATH)
    model_config.output_hidden_states = True

    model1 = TweetModel(train_config, conf=model_config)
    model1.to(device)
    model1.load_state_dict(torch.load("model_0.bin"))
    model1.eval()

    model2 = TweetModel(train_config, conf=model_config)
    model2.to(device)
    model2.load_state_dict(torch.load("model_1.bin"))
    model2.eval()

    model3 = TweetModel(train_config, conf=model_config)
    model3.to(device)
    model3.load_state_dict(torch.load("model_2.bin"))
    model3.eval()

    model4 = TweetModel(train_config, conf=model_config)
    model4.to(device)
    model4.load_state_dict(torch.load("model_3.bin"))
    model4.eval()

    model5 = TweetModel(train_config, conf=model_config)
    model5.to(device)
    model5.load_state_dict(torch.load("model_4.bin"))
    model5.eval()

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
            ids = d["ids"].to(device, dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device, dtype=torch.long)
            mask = d["mask"].to(device, dtype=torch.long)

            outputs_start1, outputs_end1 = model1(ids=ids, mask=mask, token_type_ids=token_type_ids)
            outputs_start2, outputs_end2 = model2(ids=ids, mask=mask, token_type_ids=token_type_ids)
            outputs_start3, outputs_end3 = model3(ids=ids, mask=mask, token_type_ids=token_type_ids)
            outputs_start4, outputs_end4 = model4(ids=ids, mask=mask, token_type_ids=token_type_ids)
            outputs_start5, outputs_end5 = model5(ids=ids, mask=mask, token_type_ids=token_type_ids)

            outputs_start = (outputs_start1 + outputs_start2 + outputs_start3 + outputs_start4 + outputs_start5) / 5
            outputs_end = (outputs_end1 + outputs_end2 + outputs_end3 + outputs_end4 + outputs_end5) / 5

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