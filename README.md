# Tweet Sentiment Extraction

#### Extract support phrases for sentiment labels

https://www.kaggle.com/c/tweet-sentiment-extraction


### Build & Run

```bash
$ docker build -t tse -f Dockerfile .
$ docker run -v /var/local/pgladkov/data:/data -v /var/local/pgladkov/tweet-sentiment-extraction:/app --runtime nvidia -it tse 
```

### Download data

```bash
$ kaggle competitions download -c tweet-sentiment-extraction -p /data/tweet-sentiment-extraction/
```

### Run

```bash
$ python tse_train.py
```

## Related sources

1. [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692).

2. [Multi-Sample Dropout for Accelerated Training and Better Generalization](https://arxiv.org/abs/1905.09788).

3. [Gaussian Error Linear Units (GELUs)](https://arxiv.org/abs/1606.08415).

