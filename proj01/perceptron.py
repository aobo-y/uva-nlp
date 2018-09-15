import numpy as np
from time import time
from bag_of_words import BagOfWords


class Perceptron:
  """perceptron classification"""

  weights = None

  def __init__(self, bag_of_words):
    self.bag_of_words = bag_of_words

  def fit(self, data, labels, epoch = 5, epoch_callback = None):
    for i in range(0, epoch):
      for instance, label in zip(data, labels):
        feature_vectors = np.array([self.bag_of_words.get_feature_vector(instance, l) for l in self.bag_of_words.label_keys])

        if self.weights is None:
          # length of weights matches the length of feature vector
          self.weights = np.zeros(len(feature_vectors[0]))

        scores = self.weights @ feature_vectors.T
        predict_index = np.argmax(scores)

        correct_index = self.bag_of_words.label_indexes[label]

        if (predict_index != correct_index):
          self.weights = self.weights - feature_vectors[predict_index] + feature_vectors[correct_index]

      # after each epoch run epoch_callback if any
      if (epoch_callback is not None):
        epoch_callback(i)

  def __predict_label_index(self, instance):
    feature_vectors = np.array([self.bag_of_words.get_feature_vector(instance, l) for l in self.bag_of_words.label_keys])
    scores = self.weights @ feature_vectors.T
    return np.argmax(scores)

  def accuracy(self, data, labels):
    predict_indexes = [self.__predict_label_index(instance) for instance in data]
    predict_labels = [self.bag_of_words.label_keys[i] for i in predict_indexes]
    return round(sum(1 for pl, l in zip(predict_labels, labels) if pl == l) / len(labels), 4)



def main():
  trn_texts= open("trn.data").read().strip().split("\n")
  trn_labels = open("trn.label").read().strip().split("\n")
  dev_texts= open("dev.data").read().strip().split("\n")
  dev_labels = open("dev.label").read().strip().split("\n")

  print('perceptron\n----------------------------------------------')
  print('trn data size:', len(trn_texts))
  print('dev data size:', len(dev_texts))

  bag_of_words = BagOfWords(True, True, 30)
  trn_data = bag_of_words.fit_transform(trn_texts, trn_labels)
  dev_data = bag_of_words.transform(dev_texts)

  print('vocabulary size:', len(trn_data[0]))

  perceptron = Perceptron(bag_of_words)

  print('training start')

  print_accuracy = lambda i: print(f'epoch {i}: trn {perceptron.accuracy(trn_data, trn_labels)} | dev {perceptron.accuracy(dev_data, dev_labels)}')
  perceptron.fit(trn_data, trn_labels, 5, print_accuracy)

  print('training end')

if __name__ == '__main__':
  main()
