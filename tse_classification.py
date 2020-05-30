import pandas as pd
from transformers import BertTokenizer

from tse.classification.bert_data import df_to_dataset
from tse.classification.bert_trainer import BertTrainer
from tse.utils import get_logger, set_seed


bert_settings = {
    'max_seq_length': 128,
    'num_train_epochs': 4,
    'train_batch_size': 16,
    'eval_batch_size': 16,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'test_size': 0.2,
    'log_dir': '/data/tweet-sentiment-extraction/logs',
    'tb_suffix': 'bert'
}


if __name__ == '__main__':

    logger = get_logger()

    set_seed(3)

    train_df = pd.read_csv('/data/tweet-sentiment-extraction/train.csv', encoding='utf-8')

    train_df = train_df[train_df['sentiment'] != 'neutral']
    train_df['sentence'] = train_df['text']
    train_df['label'] = train_df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_dataset = df_to_dataset(train_df, bert_tokenizer, bert_settings['max_seq_length'])

    trainer = BertTrainer(bert_settings, logger)
    model = trainer.train(train_dataset, bert_tokenizer, '/data/tweet-sentiment-extraction/classification')