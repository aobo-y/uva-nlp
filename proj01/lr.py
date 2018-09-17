from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

from time import time
from utils import print_cells

trn_texts= open("trn.data").read().strip().split("\n")
trn_labels = open("trn.label").read().strip().split("\n")
dev_texts= open("dev.data").read().strip().split("\n")
dev_labels = open("dev.label").read().strip().split("\n")

def run_vectorize(name, v):
  print(name, 'LogisticRegression')
  print('-' * 40)

  print('transform feature')

  trn_data = v.fit_transform(trn_texts)
  dev_data = v.transform(dev_texts)

  print('feature length:', len(v.get_feature_names()))

  return trn_data, dev_data


def run_training(name, lr, trn_data, dev_data):
  print(name, 'LogisticRegression')
  print('-' * 40)

  print('training start')
  start = time()

  lr.fit(trn_data, trn_labels)

  print('training end')
  print('duration:', round(time() - start))

  print('accurary')
  print_cells(['trn', 'dev'], 9)
  print('-' * 20)

  trn_score = lr.score(trn_data, trn_labels)
  dev_score = lr.score(dev_data, dev_labels)

  print_cells([round(s, 4) for s in [trn_score, dev_score]], 9)

  print('\n')



# 3.1 default
vectorizer = CountVectorizer()
linear_regression = LogisticRegression()

trn_data, dev_data = run_vectorize('default', vectorizer)
run_training('default', linear_regression, trn_data, dev_data)

# 3.2 2-gram
vectorizer = CountVectorizer(ngram_range = (1, 2))
linear_regression = LogisticRegression()

trn_data, dev_data = run_vectorize('2-gram', vectorizer)
# run_training('default', linear_regression, trn_data, dev_data)

# 3.3
# reuse the vectorizer in step 2
# for l in [0.0001, 0.001, 0.01, 0.1, 1, 10, 100]:
for l in [5, 6, 7, 8]: # 6
  linear_regression = LogisticRegression(C = 1 / l)

  run_training(f'L2 lamda={l}', linear_regression, trn_data, dev_data)

# 3.4
# reuse the vectorizer in step 2
for l in [0.002, 0.003, 0.004, 0.005, 2, 3, 4]:
  linear_regression = LogisticRegression(penalty = 'l1', C = 1 / l)

  run_training(f'L1 lamda={l}', linear_regression, trn_data, dev_data)
