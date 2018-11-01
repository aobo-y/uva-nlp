from math import inf

from preprocess import preprocess
from file import load_data, write_data

tags = {'A', 'C', 'D', 'M', 'N', 'O', 'P', 'R', 'V', 'W'}

def viterbi(seqs, score):
  v_scores = [{t: -inf for t in tags} for _ in range(len(seqs))]
  b_tags = [{t: None for t in tags} for _ in range(len(seqs))]

  for index, token in enumerate(seqs):
    for tag in tags:
      if (index == 0):
        v_scores[index][tag] = score(None, tag, token)
      else:
        for prev_tag in tags:
          new_score = v_scores[index - 1][prev_tag] + score(prev_tag, tag, token)
          if new_score > v_scores[index][tag]:
            v_scores[index][tag] = new_score
            b_tags[index][tag] = prev_tag

  end_score = -inf
  last_tag = None
  for prev_tag, prev_score in v_scores[len(seqs) - 1].items():
    new_score = prev_score + score(prev_tag, None, None)
    if new_score > end_score:
      end_score = new_score
      last_tag = prev_tag

  result = [None] * len(seqs)
  result[-1] = last_tag
  for i in range(len(seqs) - 1, 0, -1):
    result[i - 1] = b_tags[i][result[i]]

  return result


def main():
  ALPHA = 0.005
  BETA = 300

  print('alpha:', ALPHA)
  print('beta:', BETA)

  t_logs, e_logs = preprocess('trn.pos', ALPHA, BETA)
  dev_data = load_data('dev.pos')
  tst_data = load_data('tst.word')

  dev_words = [[token[0] for token in line] for line in dev_data]
  dev_labels = [[token[1] for token in line] for line in dev_data]
  tst_words = [[token[0] for token in line] for line in tst_data]

  def score(prev_tag, tag, word):
    if tag is None:
      return t_logs[prev_tag]['END']

    if prev_tag is None:
      prev_tag = 'START'

    if (word not in e_logs[tag]):
      word = 'Unk'

    return t_logs[prev_tag][tag] + e_logs[tag][word]

  predict_labels = [viterbi(line, score) for line in dev_words]

  correct_no = len([1 for i in range(len(dev_labels)) if dev_labels[i] == predict_labels[i]])
  print('accuracy:', correct_no / len(dev_labels))

  # err_res = [[dev_labels[i], predict_labels[i]] for i in range(len(dev_labels)) if dev_labels[i] != predict_labels[i]]

  # for r in err_res:
  #   print(r[0])
  #   print(r[1])
  #   print('\n')

  predict_tst_labels = [viterbi(line, score) for line in tst_words]
  output = [' '.join([f'{word}/{label}' for word, label in zip(words, labels)]) for words, labels in zip(tst_words, predict_tst_labels)]
  write_data('viterbi-tuned.txt', '\n'.join(output))

if __name__ == '__main__':
   main()
