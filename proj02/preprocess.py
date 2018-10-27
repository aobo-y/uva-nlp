from decimal import Decimal
from math import log2

from file import load_data, write_data

MIN_FREQ = 2
tags = {'A', 'C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W'}


def get_vocabulary(data, min_freq = 0):
  word_counts = {}
  for line in data:
    for token in line:
      word = token[0]
      if word not in word_counts:
        word_counts[word] = 0
      word_counts[word] += 1

  vocab = {w for w, c in word_counts.items() if c >= min_freq}
  vocab.add('Unk')
  return vocab


def get_transition_probability(data, beta = 0):
  t_count = {t1: {t2: 0 for t2 in tags | {'END'}} for t1 in tags | {'START'}}

  for line in data:
    line_tags = [t[1] for t in line]
    line_tags.append('END')
    prev_tag = 'START'

    for tag in line_tags:
      t_count[prev_tag][tag] += 1
      prev_tag = tag

  tag_size = len(tags) + 1 # include END
  tag_sum = {t1: sum(v1.values()) for t1, v1 in t_count.items()}
  to_prob = lambda count, tag: (Decimal(count) + Decimal(beta)) / (Decimal(tag_sum[tag]) + tag_size * Decimal(beta))

  t_prob = {t1: {t2: to_prob(v2, t1) for t2, v2 in v1.items()} for t1, v1 in t_count.items()}

  # print([f'{t1} {sum(probs.values())}' for t1, probs in t_prob.items()])

  return t_prob


def get_emission_probability(data, vocabulary, alpha = 0):
  e_count = {t: {w: 0 for w in vocabulary} for t in tags}

  for line in data:
    for [word, tag] in line:
      if (word not in vocabulary):
        word = 'Unk'
      e_count[tag][word] += 1

  vocab_size = len(vocabulary)
  tag_sum = {t: sum(v1.values()) for t, v1 in e_count.items()}
  to_prob = lambda count, tag: (Decimal(count) + Decimal(alpha)) / (Decimal(tag_sum[tag]) + vocab_size * Decimal(alpha))

  e_prob = {t: {w: to_prob(v2, t) for w, v2 in v1.items()} for t, v1 in e_count.items()}

  # print([f'{t} {sum(probs.values())}' for t, probs in e_prob.items()])

  return e_prob

def prob_to_log(data):
  return {k1: {k2: log2(v2) for k2, v2 in v1.items()} for k1, v1 in data.items()}


def format_probs(data):
  return '\n'.join(['\n'.join([f'{t1} {t2} {v2}' for t2, v2 in v1.items()]) for t1, v1 in data.items()])


def preprocess(alpha = 0, beta = 0):
  data = load_data('trn.pos')

  vocabulary = get_vocabulary(data, MIN_FREQ)

  transition_probability = get_transition_probability(data, alpha)
  emission_probability = get_emission_probability(data, vocabulary, beta)

  t_logs = prob_to_log(transition_probability)
  e_logs = prob_to_log(emission_probability)

  return t_logs, e_logs

def main():
  print('minimum word frequency is:', MIN_FREQ)

  trn_data = load_data('trn.pos')

  vocabulary = get_vocabulary(trn_data, MIN_FREQ)
  print('vocabulary size is:', len(vocabulary))

  transition_probability = get_transition_probability(trn_data)
  write_data('tprob.txt', format_probs(transition_probability))

  emission_probability = get_emission_probability(trn_data, vocabulary)
  write_data('eprob.txt', format_probs(emission_probability))

if __name__ == '__main__':
   main()
