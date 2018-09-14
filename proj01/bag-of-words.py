import re
# from nltk.tokenize import word_tokenize

class BagOfWords:
  """build bag of words features"""

  features_desc = None

  def __init__(self, lower_case = False, filter_letter = False, min_freq = 1):
    self.lower_case = lower_case
    self.min_freq = min_freq
    self.filter_letter = filter_letter

  # fit with the training data and return the features
  def fit_features(self, contents, labels):
    # lowercase
    if (self.lower_case):
      contents = [s.lower() for s in contents]

    token_contents = [s.split(' ') for s in contents]

    # filter out tokens without any letter [a-zA-Z]
    if (self.filter_letter):
      filterLetter = lambda tokens: [t for t in tokens if re.search(r'[a-zA-Z]', t) is not None]
      token_contents = [filterLetter(tokens) for tokens in token_contents]

    # group feature counts by labels
    features_desc = {}

    for tokens in token_contents:
      for token in tokens:
        if (token not in features_desc):
          features_desc[token] = 0

        features_desc[token] += 1

    # map low frequency words to UKN
    features_desc['UKN'] = sum(v for v in features_desc.values() if v < self.min_freq)
    features_desc = {k:v for k, v in features_desc.items() if v >= self.min_freq}

    self.features_desc = features_desc


trn_data = open("trn.data").read().strip().split("\n")
trn_labels = map(str, open("trn.label").read().strip().split("\n"))

bag_of_words = BagOfWords(True, True, 5)

bag_of_words.fit_features(trn_data, trn_labels)
print(bag_of_words.features_desc)
print(len(bag_of_words.features_desc))
