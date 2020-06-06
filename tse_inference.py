import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import LocalTrainConfig, KaggleTrainConfig
from tse.data import TweetDataset
from tse.models import load_model
from tse.utils import set_seed, device, batch_jaccard, cuda_num

SUBMIT = False


if __name__ == '__main__':

    if SUBMIT:
        train_config = KaggleTrainConfig()
        map_location = {'cuda:0': 'cuda:0', 'cuda:1': 'cuda:0', 'cuda:2': 'cuda:0', 'cuda:3': 'cuda:0'}
        cuda_n = ''
    else:
        train_config = LocalTrainConfig()
        map_location = None
        cuda_n = cuda_num()

    start_time = time.time()
    device_ = device()

    models = []
    for fold in range(5):
        print(f"loading model_{fold}_{cuda_n}.bin")
        model = load_model(train_config, device_)
        model.load_state_dict(
            torch.load(f"{train_config.WEIGHTS_DIR}/{model.prefix}_model_{fold}_{cuda_n}.bin",
                       map_location=map_location)
        )
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

    data_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=train_config.VALID_BATCH_SIZE,
        num_workers=1
    )

    wrong_order_count = 0
    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        for bi, d in enumerate(tk0):
            ids = d["ids"].to(device_, dtype=torch.long)
            token_type_ids = d["token_type_ids"].to(device_, dtype=torch.long)
            mask = d["mask"].to(device_, dtype=torch.long)

            start_probs_list = []
            end_probs_list = []

            for model in models:
                outputs_start, outputs_end, _ = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                start_probs, end_probs = model.softmax(outputs_start, outputs_end)
                start_probs_list.append(start_probs)
                end_probs_list.append(end_probs)

            outputs_start = (start_probs_list[0] + start_probs_list[1] + start_probs_list[2] +
                             start_probs_list[3] + start_probs_list[4]) / len(start_probs_list)

            outputs_end = (end_probs_list[0] + end_probs_list[1] + end_probs_list[2] +
                           end_probs_list[3] + end_probs_list[4]) / len(end_probs_list)

            start_positions, end_positions = models[0].probs_to_positions(outputs_start, outputs_end)

            _, filtered_outputs, wrong_order_ = batch_jaccard(start_positions, end_positions, d)
            wrong_order_count += wrong_order_
            for s in filtered_outputs:
                final_output.append(s)

    # post-process trick:
    # Note: This trick comes from: https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/140942
    # When the LB resets, this trick won't help
    def post_process(selected):
        return " ".join(set(selected.lower().split()))

    print(f'wrong order {wrong_order_count}')

    sample = pd.read_csv(train_config.SAMPLE_FILE)
    sample.loc[:, 'selected_text'] = final_output
    sample.selected_text = sample.selected_text.map(post_process)
    sample.to_csv("submission.csv", index=False)
    print(f'time: {(time.time() - start_time) / 60} m')