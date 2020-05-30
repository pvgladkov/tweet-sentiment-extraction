import torch
import transformers
from torch import nn as nn


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, train_config, conf):
        super(TweetModel, self).__init__(conf)
        self.bert = transformers.RobertaModel.from_pretrained(train_config.BERT_PATH, config=conf)
        self.drop_out = nn.Dropout(0.1)
        self.l0 = nn.Linear(768, 2)
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids):
        _, _, out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = 0.3*out[-1] + 0.2*out[-2] + 0.1*out[-3] + 0.1*out[-4] + 0.1*out[-5] + 0.1*out[-6] + 0.1*out[-7]
        out = self.drop_out(out)
        logits = self.l0(out)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits
