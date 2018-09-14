import re
# from nltk.tokenize import word_tokenize

class BagOfWords:
  """build bag of words features"""

  features_desc = None
  labels_desc = None

  def __init__(self, lower_case = False, filter_letter = False, min_freq = 1):
    self.lower_case = lower_case
    self.min_freq = min_freq
    self.filter_letter = filter_letter

  # construct features_desc with distinct feature keys, index and freq
  def __build_features_desc(self, token_contents):
    # group feature counts by labels
    features_desc = {}

    for tokens in token_contents:
      for token in tokens:
        if (token not in features_desc):
          features_desc[token] = 0

        features_desc[token] += 1

    # map low frequency words to UKN
    features_desc['UKN'] = sum(v for v in features_desc.values() if v < self.min_freq)
    features_desc = {k: v for k, v in features_desc.items() if v >= self.min_freq}

    # assign index to features
    features_desc = {k: {'index': i, 'freq': features_desc[k]} for i, k in enumerate(features_desc)}

    self.features_desc = features_desc

  # construct labels_desc with distinct label and index
  def __build_labels_desc(self, labels):
    self.labels_desc = {k: {'index': i} for i, k in enumerate(list(set(labels)))}

  # feature label function to combine feature vector with label
  def feature_label_vector(self, x, y):
    features_len = len(x)
    labels_len = len(self.labels_desc)

    feature_label_vector = [0] * features_len * labels_len

    y_index = self.labels_desc[y]['index']
    for i, v in enumerate(x):
        feature_label_vector[y_index * features_len + i] = v

    return feature_label_vector

  # feature extraction function to get vector out of tokens
  def feature_data(self, tokens):
    features_len = len(self.features_desc) + 1 # give it an offset

    features_vector = [0] * features_len
    for token in tokens:
      if token in self.features_desc:
        features_vector[self.features_desc[token]['index']] += 1
      else:
        features_vector[self.features_desc['UKN']['index']] += 1

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

    self.__build_features_desc(token_contents)
    self.__build_labels_desc(labels)

    return [self.feature_data(tokens) for tokens in token_contents]


trn_data = open("trn.data").read().strip().split("\n")
trn_labels = open("trn.label").read().strip().split("\n")

bag_of_words = BagOfWords(True, True, 5)

features_vectors = bag_of_words.fit_transform(trn_data, trn_labels)

print(bag_of_words.features_desc)
print(len(bag_of_words.features_desc))
print(bag_of_words.labels_desc)
print(len(features_vectors[0]))
