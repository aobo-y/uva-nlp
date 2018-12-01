import os
import math
import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint')
parser.add_argument('--model', choices={'1', '2', '3', '4', '5'})
args = parser.parse_args()

if args.model == '1':
    from simple_rnnlm import MODEL_NAME, DEVICE, init, load_data, data_to_idx, sentence_to_tensors
elif args.model == '2':
    from stackedlstm_rnnlm import MODEL_NAME, DEVICE, init, load_data, data_to_idx, sentence_to_tensors
elif args.model == '3':
    from opt_rnnlm import MODEL_NAME, DEVICE, init, load_data, data_to_idx, sentence_to_tensors
elif args.model == '4':
    from model_rnnlm import MODEL_NAME, DEVICE, init, load_data, data_to_idx, sentence_to_tensors
elif args.model == '5':
    from minibatch_rnnlm import MODEL_NAME, DEVICE, init, load_data, data_to_idx, sentence_to_tensors

DIR_NAME = os.path.dirname(__file__)


DEV_FILE = 'dev-wiki.txt'
TST_FILE = 'tst-wiki.txt'

PERPLEXITY_FOLDER = os.path.join(DIR_NAME, 'perplexity')
PERPLEXITY_FILE = os.path.join(PERPLEXITY_FOLDER, MODEL_NAME + '-tst-logprob.txt')

PRINT_EVERY = 1000

## ll for log-likelihood and coz we use LogSoftmax, the output is ll
def perplexity(model, data):
    data_ll = []
    for idx, line in enumerate(data):
        input_tensor, target_tensor = sentence_to_tensors(line)
        input_tensor = input_tensor.to(DEVICE)

        if model != 5:
            output_tensor = model(input_tensor)
        else:
            length_tensor = torch.tensor([input_tensor.size(0)])
            output_tensor = model(input_tensor, length_tensor)

        output_tensor = output_tensor.squeeze(1)

        sentence_ll = []
        for i in range(output_tensor.size(0)):
            target_index = target_tensor[i].item()
            token_ll = output_tensor[i, target_index].item()
            sentence_ll.append(token_ll)

        data_ll.append(sentence_ll)
        if (idx + 1) % PRINT_EVERY == 0:
            print('number of sentences:', idx + 1)

    size = sum([len(sentence_ll) for sentence_ll in data_ll])
    sum_nll = sum([sum(sentence_ll) for sentence_ll in data_ll])

    avg_nll = sum_nll / size
    return math.exp(-avg_nll), data_ll


def main():
    print('calculate perplexity for model:', MODEL_NAME)

    model, word_map, trn_data, _ = init(args.checkpoint)

    dev_data = load_data(DEV_FILE)
    tst_data = load_data(TST_FILE)

    trn_idx = data_to_idx(trn_data, word_map)
    dev_idx = data_to_idx(dev_data, word_map)
    tst_idx = data_to_idx(tst_data, word_map)

    print('calculate perplexity for training')
    trn_perplexity, _ = perplexity(model, trn_idx)
    print('training data perplexity:', trn_perplexity)

    print('calculate perplexity for devlopment')
    dev_perplexity, _ = perplexity(model, dev_idx)
    print('development data perplexity:', dev_perplexity)

    print('calculate perplexity for testing')
    tst_perplexity, tst_ll = perplexity(model, tst_idx)
    print('development testing perplexity:', tst_perplexity)

    if args.model == '1':
        print('output testing log-likelihood')
        if not os.path.exists(PERPLEXITY_FOLDER):
            os.mkdir(PERPLEXITY_FOLDER)

        with open(PERPLEXITY_FILE, 'w', encoding='utf-8') as file:
            output = []
            for line, line_ll in zip(tst_data, tst_ll):
                ll_output = [f'{token}\t{token_ll}' for token, token_ll in zip(line[1:], line_ll)]
                output.append('\n'.join(ll_output))

            file.write('\n'.join(output))

if __name__ == '__main__':
    main()
