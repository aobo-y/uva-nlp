## crf.py
## Author: CS 6501-005 NLP @ UVa
## Time-stamp: <yangfeng 10/14/2018 16:14:05>

from util import load_data
import sklearn_crfsuite as crfsuite
from sklearn_crfsuite import metrics

class CRF(object):
    def __init__(self, trnfile, devfile):
        self.trn_text = load_data(trnfile)
        self.dev_text = load_data(devfile)
        #
        print('Extracting features on training data ...')
        self.trn_feats, self.trn_tags = self.build_features(self.trn_text)
        print('Extracting features on dev data ...')
        self.dev_feats, self.dev_tags = self.build_features(self.dev_text)
        #
        self.model, self.labels = None, None

    def build_features(self, text):
        feats, tags = [], []
        for sent in text:
            N = len(sent.tokens)
            sent_feats = []
            for i in range(N):
                word_feats = self.get_word_features(sent, i)
                sent_feats.append(word_feats)
            feats.append(sent_feats)
            tags.append(sent.tags)
        return (feats, tags)


    def train(self):
        print('Training CRF ...')
        self.model = crfsuite.CRF(
            # algorithm='lbfgs',
            algorithm='ap',
            max_iterations=5)
        self.model.fit(self.trn_feats, self.trn_tags)
        trn_tags_pred = self.model.predict(self.trn_feats)
        self.eval(trn_tags_pred, self.trn_tags)
        dev_tags_pred = self.model.predict(self.dev_feats)
        self.eval(dev_tags_pred, self.dev_tags)


    def eval(self, pred_tags, gold_tags):
        if self.model is None:
            raise ValueError('No trained model')
        print(self.model.classes_)
        print('Acc =', metrics.flat_accuracy_score(pred_tags, gold_tags))


    def get_word_features(self, sent, i):
        ''' Extract features with respect to time step i
        '''

        # the i-th token
        tok = sent.tokens[i]
        word_feats = {'tok': tok}

        # TODO for question 1
        # the i-th tag
        # word_feats['pos'] = sent.tags[i]

        # TODO for question 2
        # add more features here

        word_feats['is_digit'] = tok.isdigit()
        word_feats['is_upper'] = tok.isupper()

        word_feats['first_letter'] = tok[:1]

        if (len(tok) > 2):
            word_feats['first_2_letters'] = tok[:2]
            word_feats['last_2_letters'] = tok[-2:]

        if (i > 0):
            prev_tok = sent.tokens[i - 1]
            word_feats['prev_tok'] = prev_tok
            word_feats['prev_tok_is_digit'] = prev_tok.isdigit()
            # word_feats['prev_tok_first_letter'] = prev_tok[:1]

            if (len(prev_tok) > 2):
                word_feats['prev_tok_first_2_letters'] = prev_tok[:2]
                word_feats['prev_tok_last_2_letters'] = prev_tok[-2:]

        if (i < len(sent.tokens) - 1):
            next_tok = sent.tokens[i + 1]
            word_feats['next_tok'] = next_tok
            word_feats['next_tok_is_digit'] = next_tok.isdigit()
            # word_feats['next_tok_first_letter'] = next_tok[:1]

            if (len(next_tok) > 2):
                word_feats['next_tok_first_2_letters'] = next_tok[:2]
                word_feats['next_tok_last_2_letters'] = next_tok[-2:]


        return word_feats


if __name__ == '__main__':
    trnfile = 'trn-tweet.pos'
    devfile = 'dev-tweet.pos'
    crf = CRF(trnfile, devfile)
    crf.train()


