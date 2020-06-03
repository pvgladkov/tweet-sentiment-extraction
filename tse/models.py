import torch
import transformers
from torch import nn as nn
import numpy as np


def load_model(train_config, device):
    model_config = transformers.RobertaConfig.from_pretrained(train_config.BERT_PATH)
    model_config.output_hidden_states = True
    model = TweetModel(train_config, conf=model_config)
    model.to(device)
    return model


class GELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class TweetModel(transformers.BertPreTrainedModel):

    prefix = 'roberta'

    def __init__(self, train_config, conf):
        super(TweetModel, self).__init__(conf)
        self.bert = transformers.RobertaModel.from_pretrained(train_config.BERT_PATH, config=conf)
        dropout_p = 0.4
        self.l0 = nn.Linear(768, 256)
        self.l1 = nn.Linear(256, 2)
        self.gelu = GELU()
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_p) for _ in range(4)])
        torch.nn.init.normal_(self.l0.weight, std=0.02)

    @staticmethod
    def loss_fn(start_logits, end_logits, start_positions, end_positions):
        loss_fct = nn.CrossEntropyLoss()
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss)
        return total_loss

    def forward(self, ids, mask, token_type_ids, start=None, end=None):
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
                if start is not None:
                    loss = self.loss_fn(start_logits, end_logits, start, end)
            else:
                tmp_out = dropout(out)
                logits = self.l1(tmp_out)
                tmp_start_logits, tmp_end_logits = logits.split(1, dim=-1)

                tmp_start_logits = tmp_start_logits.squeeze(-1)
                tmp_end_logits = tmp_end_logits.squeeze(-1)

                start_logits = start_logits + tmp_start_logits
                end_logits = end_logits + tmp_end_logits
                if start is not None:
                    loss += self.loss_fn(tmp_start_logits, tmp_end_logits, start, end)

        start_logits = start_logits / len(self.dropouts)
        end_logits = end_logits / len(self.dropouts)

        if start is not None:
            loss = loss / len(self.dropouts)

        return start_logits, end_logits, loss

    @staticmethod
    def to_positions(start_logits, end_logits):
        outputs_start_ = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
        outputs_end_ = torch.softmax(end_logits, dim=1).cpu().detach().numpy()
        start_positions = np.argmax(outputs_start_, axis=1)
        end_positions = np.argmax(outputs_end_, axis=1)
        return start_positions, end_positions


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.l0 = nn.Linear(768, 256)
        self.l1 = nn.Linear(256, 1)
        self.gelu = GELU()
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        out = self.dropout(x)
        out = self.l1(out)
        out = out.squeeze(-1)
        return out


class TweetModelTwoHead(TweetModel):
    def __init__(self, train_config, conf):
        super(TweetModelTwoHead, self).__init__(train_config, conf)
        self.start_head = Head()
        self.end_head = Head()

    def forward(self, ids, mask, token_type_ids, start=None, end=None):
        _, _, out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )

        out = 0.3*out[-1] + 0.2*out[-2] + 0.1*out[-3] + 0.1*out[-4] + 0.1*out[-5] + 0.1*out[-6] + 0.1*out[-7]
        out = self.l0(out)
        out = self.gelu(out)

        start_out = self.start_head(out)
        end_out = self.end_head(out)
        loss = None
        if start is not None:
            loss = self.loss_fn(start_out, end_out, start, end)

        return start_out, end_out, loss
