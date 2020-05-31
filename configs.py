import tokenizers


class LocalTrainConfig:
    def __init__(self):
        self.MAX_LEN = 192
        self.TRAIN_BATCH_SIZE = 32
        self.VALID_BATCH_SIZE = 8
        self.EPOCHS = 5
        self.WEIGHTS_DIR = 'weights'
        self.BERT_PATH = "/data/tweet-sentiment-extraction/roberta-base"
        self.MODEL_PATH = "model.bin"
        self.TRAINING_FILE = "/data/tweet-sentiment-extraction/train_folds.csv"
        self.TEST_FILE = "/data/tweet-sentiment-extraction/test.csv"
        self.TRAIN_FILE = "/data/tweet-sentiment-extraction/train.csv"
        self.SAMPLE_FILE = "/data/tweet-sentiment-extraction/sample_submission.csv"
        self.TOKENIZER = tokenizers.ByteLevelBPETokenizer(
            vocab_file=f"{self.BERT_PATH}/vocab.json",
            merges_file=f"{self.BERT_PATH}/merges.txt",
            lowercase=True,
            add_prefix_space=True
        )


class KaggleTrainConfig:
    def __init__(self):
        self.MAX_LEN = 192
        self.TRAIN_BATCH_SIZE = 32
        self.VALID_BATCH_SIZE = 8
        self.EPOCHS = 5
        self.WEIGHTS_DIR = '../input/tse-weights'
        self.BERT_PATH = "../input/roberta-base/"
        self.MODEL_PATH = "model.bin"
        self.TRAINING_FILE = "train_folds.csv"
        self.TEST_FILE = "../input/tweet-sentiment-extraction/test.csv"
        self.TRAIN_FILE = "../input/tweet-sentiment-extraction/train.csv"
        self.SAMPLE_FILE = "../input/tweet-sentiment-extraction/sample_submission.csv"
        self.TOKENIZER = tokenizers.ByteLevelBPETokenizer(
            vocab_file=f"{self.BERT_PATH}/vocab.json",
            merges_file=f"{self.BERT_PATH}/merges.txt",
            lowercase=True,
            add_prefix_space=True
        )
