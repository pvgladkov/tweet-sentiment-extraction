import torch
import transformers
from torch import nn as nn


class GELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class TweetModel(transformers.BertPreTrainedModel):
    def __init__(self, train_config, conf):
        super(TweetModel, self).__init__(conf)
        self.bert = transformers.RobertaModel.from_pretrained(train_config.BERT_PATH, config=conf)
        dropout_p = 0.4
        self.l0 = nn.Linear(768, 256)
        self.l1 = nn.Linear(256, 2)
        self.gelu = GELU()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(4)])
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    def forward(self, ids, mask, token_type_ids, start=None, end=None, loss_fn=None):
        _, _, out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = 0.3*out[-1] + 0.2*out[-2] + 0.1*out[-3] + 0.1*out[-4] + 0.1*out[-5] + 0.1*out[-6] + 0.1*out[-7]

        out = self.l0(out)
        out = self.gelu(out)

        start_logits = None
        end_logits = None
        loss = None
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                tmp_out = dropout(out)
                logits = self.l1(tmp_out)
                start_logits, end_logits = logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)
                if loss_fn is not None:
                    loss = loss_fn(start_logits, end_logits, start, end)
            else:
                tmp_out = dropout(out)
                logits = self.l1(tmp_out)
                tmp_start_logits, tmp_end_logits = logits.split(1, dim=-1)

                tmp_start_logits = tmp_start_logits.squeeze(-1)
                tmp_end_logits = tmp_end_logits.squeeze(-1)

                start_logits = start_logits + tmp_start_logits
                end_logits = end_logits + tmp_end_logits
                if loss_fn is not None:
                    loss += loss_fn(tmp_start_logits, tmp_end_logits, start, end)

        start_logits = start_logits / len(self.dropouts)
        end_logits = end_logits / len(self.dropouts)

        if loss_fn is not None:
            loss = loss / len(self.dropouts)

        return start_logits, end_logits, loss
