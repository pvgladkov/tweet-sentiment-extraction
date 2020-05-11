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
$ python arc_run.py
```

## Related sources

