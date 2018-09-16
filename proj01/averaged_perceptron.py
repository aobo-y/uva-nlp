import numpy as np
from time import time
from utils import print_cells
from bag_of_words import BagOfWords


class AvgPerceptron:
  """averaged perceptron classification"""

  avg_weights = None

  def __init__(self, bag_of_words):
    self.bag_of_words = bag_of_words

  # convert_instance_to_feature_vectors
  def __cvt_ins_to_fv(self, instance):
    return np.array([self.bag_of_words.get_feature_vector(instance, l) for l in self.bag_of_words.label_keys])

  def __predict_label_index(self, feature_vectors, weights):
    # shape(label_size * feature_size) dot shape(feature_size, 1) = vector of scores in label_size
    scores =  feature_vectors @ weights
    return np.argmax(scores)

  def fit(self, data, labels, epoch = 5, shuffle = False, epoch_callback = None):
    feature_vectors_ary = np.array([self.__cvt_ins_to_fv(instance) for instance in data])
    labels = np.array(labels)

    # length of weights matches the length of feature vector
    weights = np.zeros(len(feature_vectors_ary[0][0]))
    sum_weights = np.zeros(len(feature_vectors_ary[0][0]))

    # create looping indexes to trieve data in order to shuffle it to change the training order
    idx = np.arange(len(data))

    t = 0
    # start = time()
    for i in range(0, epoch):
      for looping_index in idx:
        feature_vectors = feature_vectors_ary[looping_index]
        label = labels[looping_index]

        predict_index = self.__predict_label_index(feature_vectors, weights)

        correct_index = self.bag_of_words.label_indexes[label]

        if (predict_index != correct_index):
          weights = weights - feature_vectors[predict_index] + feature_vectors[correct_index]

        # sum the total weights
        sum_weights += weights

        t += 1
        # if t % 5000 == 0:
        #   print('time:', time() - start)
        #   start = time()

      # after each epoch update the avg_weights & run epoch_callback if any
      self.avg_weights = sum_weights / t

      if (epoch_callback is not None):
        epoch_callback(i)

      # shuffle by gen random permutation
      if (shuffle):
        idx = np.random.permutation(len(data))

  def predict(self, data):
    feature_vectors_ary = [self.__cvt_ins_to_fv(instance) for instance in data]

    predict_indexes = [self.__predict_label_index(feature_vectors, self.avg_weights) for feature_vectors in feature_vectors_ary]
    return [self.bag_of_words.label_keys[i] for i in predict_indexes]

  def accuracy(self, data, labels):
    predict_labels = self.predict(data)
    return round(sum(1 for pl, l in zip(predict_labels, labels) if pl == l) / len(labels), 4)


def main():
  EPOCH = 10
  MIN_FREQ = 5
  SHUFFLE = True

  trn_texts= open("trn.data").read().strip().split("\n")
  trn_labels = open("trn.label").read().strip().split("\n")
  dev_texts= open("dev.data").read().strip().split("\n")
  dev_labels = open("dev.label").read().strip().split("\n")
  tst_texts= open("tst.data").read().strip().split("\n")

  print('averaged perceptron')
  print('-' * 40)
  print('trn data size:', len(trn_texts))
  print('dev data size:', len(dev_texts))

  bag_of_words = BagOfWords(True, True, MIN_FREQ)
  trn_data = bag_of_words.fit_transform(trn_texts, trn_labels)
  dev_data = bag_of_words.transform(dev_texts)
  tst_data = bag_of_words.transform(tst_texts)

  print('min vocabulary freq:', MIN_FREQ)
  print('vocabulary size:', len(trn_data[0]))
  print('shuffle after epoch:', SHUFFLE)

  perceptron = AvgPerceptron(bag_of_words)

  print('training start\n')
  start = time()
  print('data accurary')
  print_cells(['epoch', 'trn', 'dev'], 9)
  print('-' * 30)

  print_accuracy = lambda i: print_cells([i, perceptron.accuracy(trn_data, trn_labels), perceptron.accuracy(dev_data, dev_labels)], 9)
  perceptron.fit(trn_data, trn_labels, EPOCH, SHUFFLE, print_accuracy)

  print('\ntraining end')
  print('duration:', round(time() - start))

  print('write predict result')
  pred_tst_labels = perceptron.predict(tst_data)
  open('averaged-perceptron-test.pred', 'w+').write('\n'.join(pred_tst_labels))

if __name__ == '__main__':
  main()
