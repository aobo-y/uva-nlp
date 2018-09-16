import re
import numpy as np
# from nltk.tokenize import word_tokenize

class BagOfWords:
  """build bag of words features"""

  wordcount_keys = None # array of feature keys ['the', 'a'...]
  wordcount_indexes = None # dict of feature keys with corresponding index {'the': 1, 'a': 2....}
  label_keys = None # array of labels keys ['spam', 'non-spam'...]
  label_indexes = None # array of labels keys with corresponding index {'spam': 1, 'non-spam': 2....}

  def __init__(self, lower_case = False, filter_letter = False, min_freq = 1):
    self.lower_case = lower_case
    self.min_freq = min_freq
    self.filter_letter = filter_letter

  # construct wordcount_indexes with distinct feature keys, index and freq
  def __build_wordcount_indexes(self, token_contents):
    # group feature counts by labels
    wordcounts = {}

    for tokens in token_contents:
      for token in tokens:
        if (token not in wordcounts):
          wordcounts[token] = 0

        wordcounts[token] += 1

    # remove low frequency words and add UNK & OFFSET
    wordcount_keys = [k for k, v in wordcounts.items() if v >= self.min_freq]
    wordcount_keys.extend(['UNK', 'OFFSET'])
    # assign index to features
    wordcount_indexes = {v: i for i, v in enumerate(wordcount_keys)}

    self.wordcount_keys = wordcount_keys
    self.wordcount_indexes = wordcount_indexes

  # construct label_indexes with distinct label and index
  def __build_label_indexes(self, labels):
    label_keys = list(set(labels))
    # sort the label to ensure feature vector sequence, since set has no sequence
    label_keys.sort()
    self.label_keys = label_keys
    self.label_indexes = {k: i for i, k in enumerate(label_keys)}

  # feature label function to combine feature vector with label
  def get_feature_vector(self, x, y):
    features_len = len(x)
    return np.array([x if l == y else np.zeros(features_len) for l in self.label_keys]).flatten()

  # feature extraction function to get vector out of tokens
  def get_wordcount_vector(self, tokens):
    features_len = len(self.wordcount_indexes)

    wordcount_vector = np.zeros(features_len)
    for token in tokens:
      if token in self.wordcount_indexes:
        wordcount_vector[self.wordcount_indexes[token]] += 1
      else:
        wordcount_vector[self.wordcount_indexes['UNK']] += 1

    # set offset
    wordcount_vector[features_len - 1] = 1

    return wordcount_vector

  def __tokenize_texts(self, texts):
    # lowercase
    if (self.lower_case):
      texts = [s.lower() for s in texts]

    token_texts = [s.split(' ') for s in texts]

    # filter out tokens without any letter [a-zA-Z]
    if (self.filter_letter):
      filterLetter = lambda tokens: [t for t in tokens if re.search(r'[a-zA-Z]', t) is not None]
      token_texts = [filterLetter(tokens) for tokens in token_texts]

    return token_texts

  # fit with the training data and return the wordcount vectors
  def fit_transform(self, texts, labels):
    token_texts = self.__tokenize_texts(texts)

    self.__build_wordcount_indexes(token_texts)
    self.__build_label_indexes(labels)

    return np.array([self.get_wordcount_vector(tokens) for tokens in token_texts])

  # transform texts into the wordcount vectors
  def transform(self, texts):
    token_texts = self.__tokenize_texts(texts)
    return np.array([self.get_wordcount_vector(tokens) for tokens in token_texts])

  # a help util to convert feature back to words in testing
  def words_by_feature(self, feature, label):
    return [self.wordcount_keys[i - self.label_indexes[label] * len(self.wordcount_keys)] for i, v in enumerate(feature) if v != 0]


def main():
  trn_texts= open("trn.data").read().strip().split("\n")
  trn_labels = open("trn.label").read().strip().split("\n")

  bag_of_words = BagOfWords(True, True, 5)

  print('test bag of words with training data & label\n----------------------------------------------')
  trn_data = bag_of_words.fit_transform(trn_texts, trn_labels)

  print('wordcounts vector length:', len(bag_of_words.wordcount_indexes))
  print('labels length:', len(bag_of_words.label_indexes))

  sample_feature = bag_of_words.get_feature_vector(trn_data[0], trn_labels[0])
  print('feature vector length', len(sample_feature))
  # print(bag_of_words.words_by_feature(sample_feature, trn_labels[0]))

if __name__ == "__main__":
   main()
