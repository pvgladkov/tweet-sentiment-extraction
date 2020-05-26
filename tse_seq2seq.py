import time
import torch.nn as nn
from random import random, choice
from torch import optim
from tse.seq2seq.models import train, time_since, tensors_from_pair, Lang, EncoderRNN, AttnDecoderRNN
import pandas as pd
from tse.utils import device


if __name__ == '__main__':

    # read text
    train_df = pd.read_csv('/data/tweet-sentiment-extraction/train.csv', encoding='utf-8')
    train_df = train_df[train_df['sentiment'] == 'positive']
    pairs = [(s1, s2) for s1, s2 in zip(train_df['text'].values, train_df['selected_text'].values)]

    print(len(pairs))

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
    training_pairs = [tensors_from_pair(choice(pairs), lang1, lang2)
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
