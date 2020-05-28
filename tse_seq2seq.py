import time
import torch.nn as nn
from random import random, choice
from torch import optim
from tse.seq2seq.models import train, time_since, tensors_from_pair, Lang, EncoderRNN, AttnDecoderRNN, evaluate
import pandas as pd
from tse.utils import device, set_seed, jaccard
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np


if __name__ == '__main__':

    set_seed(1)

    # read text
    train_df = pd.read_csv('/data/tweet-sentiment-extraction/train.csv', encoding='utf-8')
    train_df = train_df[train_df['sentiment'] == 'negative']
    pairs = [(s1, s2) for s1, s2 in zip(train_df['text'].values, train_df['selected_text'].values)]

    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2)

    print(len(train_pairs))

    lang1 = Lang('input')
    lang2 = Lang('output')
    for pair in pairs:
        lang1.add_sentence(pair[0])
        lang2.add_sentence(pair[1])

    hidden_size = 256
    encoder = EncoderRNN(lang1.n_words, hidden_size).to(device())
    decoder = AttnDecoderRNN(hidden_size, lang2.n_words, dropout_p=0.1).to(device())

    n_iters = 100000
    print_every = 100
    plot_every = 100
    learning_rate = 0.01

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensors_from_pair(choice(train_pairs), lang1, lang2)
                      for i in range(n_iters)]

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor,
                     encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (time_since(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

    jaccard_scores = []
    texts = []
    selected_texts = []
    predicted_texts = []
    for pair in test_pairs:
        output_words, _ = evaluate(encoder, decoder, pair[0], lang1, lang2)
        output_text = ' '.join(output_words)
        j = jaccard(output_text.strip(), pair[1].strip())

        texts.append(pair[0])
        selected_texts.append(pair[1])
        predicted_texts.append(output_text)
        jaccard_scores.append(j)

    pd.DataFrame({'text': texts, 'selected_text': selected_texts,
                  'predicted_text': predicted_texts,
                  'score': jaccard_scores}).to_csv('seq2seq_debug.csv', index=False, encoding='utf-8')
    print(np.mean(jaccard_scores))