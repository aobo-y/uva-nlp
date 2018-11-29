import os
import random
import time
import torch
from torch import nn

DIR_NAME = os.path.dirname(__file__)

TRN_FILE = 'trn-wiki.txt'
DEV_FILE = 'dev-wiki.txt'
TST_FILE = 'tst-wiki.txt'

INPUT_SIZE = 32
HIDDEN_SIZE = 32
BATCH_SIZE = 1
LAYER_NUM = 1

PRINT_EVERY = 1000


def load_data(file_path):
    path = os.path.join(DIR_NAME, file_path)
    with open(path) as file:
        lines = [l.strip() for l in file.read().split('\n')]

    lines = [l.split(' ') for l in lines if l != '']

    token_num = sum([len(l) for l in lines])

    print(f'{file_path}     #sentences {len(lines)}, #tokens {token_num}')

    return lines

def build_word_map(data):
    tokens = set()
    for line in data:
        for token in line:
            tokens.add(token)
    return {token: idx for idx, token in enumerate(tokens)}

def data_to_idx(data, word_map):
    return [[word_map[token] for token in line] for line in data]


def random_batch(trn, size=1):
    # only support batch 1 for now
    assert size == 1

    sentence = random.choice(trn)
    # exclude <end>
    input_tensor = torch.LongTensor(sentence[:-1])
    input_tensor = input_tensor.view(input_tensor.size(0), 1)

    # exclude <start>
    target_tensor = torch.LongTensor(sentence[1:])

    return input_tensor, target_tensor

class LM(nn.Module):
    ''' Language Model '''

    def __init__(self, token_size, input_size, hidden_size, layer_size):
        super(LM, self).__init__()

        self.embedding = nn.Embedding(token_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size)
        self.out = nn.Linear(hidden_size, token_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    # input shape (length, batch = 1)
    def forward(self, input_tensor):
        embeded = self.embedding(input_tensor)
        lstm_output, _ = self.lstm(embeded)
        output = self.out(lstm_output)
        return self.log_softmax(output)

def train(model, trn, iterations=10000):
    loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    total_loss = 0 # for print

    for i in range(1, iterations + 1):
        optimizer.zero_grad()

        input_tensor, target_tensor = random_batch(trn, BATCH_SIZE)

        output_tensor = model(input_tensor)

        # squeeze batch 1
        output_tensor = output_tensor.squeeze(1)
        output_loss = loss(output_tensor, target_tensor)

        output_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        total_loss += output_loss.item()

        if i % PRINT_EVERY == 0:
            print('%s iter(%d) avg-loss: %.4f' % (time.strftime('%x %X'), i, total_loss / PRINT_EVERY))
            total_loss = 0


def main():
    trn_data = load_data(TRN_FILE)
    dev_data = load_data(DEV_FILE)
    tst_data = load_data(TST_FILE)

    word_map = build_word_map(trn_data)
    print('number of tokens:', len(word_map))

    model = LM(len(word_map), INPUT_SIZE, HIDDEN_SIZE, LAYER_NUM)

    trn_idx = data_to_idx(trn_data, word_map)
    dev_idx = data_to_idx(dev_data, word_map)
    tst_idx = data_to_idx(tst_data, word_map)

    iter_num = 10000
    print(f'start training of {iter_num} iterations')
    train(model, trn_idx, iter_num)


if __name__ == '__main__':
    main()
