import re
# from nltk.tokenize import word_tokenize

class BagOfWords:
  """build bag of words features"""

  feature_keys = None # array of feature keys ['the', 'a'...]
  feature_indexes = None # dict of feature keys with corresponding index {'the': 1, 'a': 2....}
  label_keys = None # array of labels keys ['spam', 'non-spam'...]
  label_indexes = None # array of labels keys with corresponding index {'spam': 1, 'non-spam': 2....}

  def __init__(self, lower_case = False, filter_letter = False, min_freq = 1):
    self.lower_case = lower_case
    self.min_freq = min_freq
    self.filter_letter = filter_letter

  # construct feature_indexes with distinct feature keys, index and freq
  def __build_feature_indexes(self, token_contents):
    # group feature counts by labels
    features_freq = {}

    for tokens in token_contents:
      for token in tokens:
        if (token not in features_freq):
          features_freq[token] = 0

        features_freq[token] += 1

    # remove low frequency words and add UKN & OFFSET
    feature_keys = [k for k, v in features_freq.items() if v >= self.min_freq]
    feature_keys.extend(['UKN', 'OFFSET'])
    # assign index to features
    feature_indexes = {v: i for i, v in enumerate(feature_keys)}

    self.feature_keys = feature_keys
    self.feature_indexes = feature_indexes

  # construct label_indexes with distinct label and index
  def __build_label_indexes(self, labels):
    label_keys = list(set(labels))
    self.label_keys = label_keys
    self.label_indexes = {k: i for i, k in enumerate(label_keys)}

  # feature label function to combine feature vector with label
  def feature_label_vector(self, x, y):
    features_len = len(x)
    labels_len = len(self.label_indexes)

    feature_label_vector = [0] * features_len * labels_len

    y_index = self.label_indexes[y]
    for i, v in enumerate(x):
        feature_label_vector[y_index * features_len + i] = v

    return feature_label_vector

  # feature extraction function to get vector out of tokens
  def feature_data(self, tokens):
    features_len = len(self.feature_indexes)

    features_vector = [0] * features_len
    for token in tokens:
      if token in self.feature_indexes:
        features_vector[self.feature_indexes[token]] += 1
      else:
        features_vector[self.feature_indexes['UKN']] += 1

    # set offset
    features_vector[features_len - 1] = 1

    return features_vector

  # fit with the training data and return the feature vectors
  def fit_transform(self, contents, labels):
    # lowercase
    if (self.lower_case):
      contents = [s.lower() for s in contents]

    token_contents = [s.split(' ') for s in contents]

    # filter out tokens without any letter [a-zA-Z]
    if (self.filter_letter):
      filterLetter = lambda tokens: [t for t in tokens if re.search(r'[a-zA-Z]', t) is not None]
      token_contents = [filterLetter(tokens) for tokens in token_contents]

    self.__build_feature_indexes(token_contents)
    self.__build_label_indexes(labels)

    return [self.feature_data(tokens) for tokens in token_contents]

  # a help util to convert feature back to words in testing
  def words_by_feature(self, feature):
    return [self.feature_keys[i - self.label_indexes[trn_labels[4]] * len(self.feature_keys)] for i, v in enumerate(feature) if v != 0]


trn_texts= open("trn.data").read().strip().split("\n")
trn_labels = open("trn.label").read().strip().split("\n")

bag_of_words = BagOfWords(True, True, 5)

trn_data = bag_of_words.fit_transform(trn_texts, trn_labels)

print(len(bag_of_words.feature_indexes))
print(bag_of_words.label_indexes)
print(len(trn_data[0]))

sample_feature = bag_of_words.feature_label_vector(trn_data[4], trn_labels[4])

# print(bag_of_words.words_by_feature(sample_feature))
