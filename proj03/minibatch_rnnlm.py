import os
import random
import time
import argparse
import torch
from torch import nn

MODEL_NAME = 'minibatch_rnnlm'

DIR_NAME = os.path.dirname(__file__)

TRN_FILE = 'trn-wiki.txt'

INPUT_SIZE = 32
HIDDEN_SIZE = 32
BATCH_SIZE = 16
LAYER_NUM = 1

PRINT_EVERY = 500
SAVE_EVERY = 20000

CHECKPOINTS_FOLDER = os.path.join(DIR_NAME, 'checkpoints', MODEL_NAME)

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

def load_data(file_path):
    path = os.path.join(DIR_NAME, file_path)
    with open(path, 'r', encoding='utf-8') as file:
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

    tokens = sorted(tokens) # sort the set in order to stable the word index
    return {token: idx for idx, token in enumerate(tokens)}

def data_to_idx(data, word_map):
    return [[word_map[token] for token in line] for line in data]


def sentence_to_tensors(sentence):
    # exclude <end>
    input_tensor = torch.LongTensor(sentence[:-1])

    # exclude <start>
    target_tensor = torch.LongTensor(sentence[1:])

    return input_tensor, target_tensor


def random_batch(trn, size=1):
    sentences = [random.choice(trn) for _ in range(size)]
    # sort to pack
    sentences.sort(key=len, reverse=True)

    sentence_tensors = [sentence_to_tensors(s) for s in sentences]

    # len includes the <start> <stop>, so need -1
    lengths = [len(s) - 1 for s in sentences]

    input_tensors_seq = [tensors[0] for tensors in sentence_tensors]
    target_tensors_seq = [tensors[1] for tensors in sentence_tensors]

    # any padding value is fine, coz it will packed with actual length in the model
    input_tensor = nn.utils.rnn.pad_sequence(input_tensors_seq)
    # shape (token_length) sum of each sentence length
    target_tensor = nn.utils.rnn.pack_sequence(target_tensors_seq).data
    return input_tensor, target_tensor, torch.tensor(lengths)

class LM(nn.Module):
    ''' Language Model '''

    def __init__(self, token_size, input_size, hidden_size, layer_size):
        super(LM, self).__init__()

        self.embedding = nn.Embedding(token_size, input_size)
        self.embedding_dropout = nn.Dropout(p=0.2)
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size)
        self.out = nn.Linear(hidden_size, token_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    # input shape (length, batch = 1)
    def forward(self, input_tensor, length_tensor):
        embedded = self.embedding(input_tensor)
        embedded = self.embedding_dropout(embedded)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, length_tensor)

        lstm_output_pack, _ = self.lstm(packed)

        lstm_output, _ = torch.nn.utils.rnn.pad_packed_sequence(lstm_output_pack)

        output = self.out(lstm_output)
        return self.log_softmax(output)

def train(model, trn, iterations=10000, checkpoints=None):
    loss = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    start_iter = 0

    if checkpoints:
        optimizer.load_state_dict(checkpoints['opt'])
        start_iter = checkpoints['iteration']

    total_loss = 0 # for print

    for i in range(start_iter + 1, iterations + 1):
        optimizer.zero_grad()

        input_tensor, target_tensor, length_tensor = random_batch(trn, BATCH_SIZE)
        input_tensor = input_tensor.to(DEVICE)
        target_tensor = target_tensor.to(DEVICE)
        length_tensor = length_tensor.to(DEVICE)

        output_tensor = model(input_tensor, length_tensor)

        # shape (token_length) sum of each sentence length
        output_pack_tensor = nn.utils.rnn.pack_padded_sequence(output_tensor, length_tensor).data
        output_pack_tensor = output_pack_tensor.to(DEVICE)

        output_loss = loss(output_pack_tensor, target_tensor)

        output_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        optimizer.step()

        total_loss += output_loss.item()

        if i % PRINT_EVERY == 0:
            print('%s iter(%d) avg-loss: %.4f' % (time.strftime('%x %X'), i, total_loss / PRINT_EVERY))
            total_loss = 0

        # Save checkpoint
        if i % SAVE_EVERY == 0:
            if not os.path.exists(CHECKPOINTS_FOLDER):
                os.makedirs(CHECKPOINTS_FOLDER)

            torch.save({
                'iteration': i,
                'loss': output_loss.item(),
                'lm': model.state_dict(),
                'opt': optimizer.state_dict()
            }, os.path.join(CHECKPOINTS_FOLDER, f'{i}.tar'))

    print('training ends')

# init everything, export to perplexity to use
def init(checkpoint_file):
    print('device:', DEVICE)

    checkpoint = None
    if checkpoint_file and checkpoint_file != '':
        cp_file = os.path.join(CHECKPOINTS_FOLDER, checkpoint_file)

        if not os.path.exists(cp_file):
            print('no checkpoint file', cp_file)
            quit()

        print('load checkpoint', cp_file)
        checkpoint = torch.load(cp_file, map_location=DEVICE)

    trn_data = load_data(TRN_FILE)


    word_map = build_word_map(trn_data)
    print('number of tokens:', len(word_map))

    print(f'create model: input size {INPUT_SIZE}, hidden size {HIDDEN_SIZE}, layer number {LAYER_NUM}')
    model = LM(len(word_map), INPUT_SIZE, HIDDEN_SIZE, LAYER_NUM)
    model.to(DEVICE)
    if checkpoint:
        model.load_state_dict(checkpoint['lm'])

    return model, word_map, trn_data, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint')
    args = parser.parse_args()
    checkpoint_file = args.checkpoint

    model, word_map, trn_data, checkpoint = init(checkpoint_file)

    # dev_data = load_data(DEV_FILE)
    # tst_data = load_data(TST_FILE)

    trn_idx = data_to_idx(trn_data, word_map)
    # dev_idx = data_to_idx(dev_data, word_map)
    # tst_idx = data_to_idx(tst_data, word_map)

    iter_num = 100 * 1000
    print(f'start training of {iter_num} iterations')
    train(model, trn_idx, iter_num, checkpoint)


if __name__ == '__main__':
    main()
