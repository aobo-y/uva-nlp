# Sequence Labeling

## Hidden Markov Models

### 1.1 A Toy Problem

#### 1

Covert Transition Probability to Transition Weight by `log`

| | H | L | END
|-|---|---|-----
| Start | -0.74 | -1.32 | −∞
| H | -1.32 | -1.32 | -2.32
| L | -2.32 | -1 | -1.74

Covert Emission Probability to Emission Weight by `log`

| | A | C | G | T
|-|---|---|---|---
| H | -2.32 | -1.74 | -1.74 | -2.32
| L | -1.74 | -2.32 | -2.32 | -1.74

The trellis table is as following

| | START | G | C | A | C | T | G | END
|---|---|---|---|---|---|---|---|---
| H | 0 | -2.48, START | -5.54, H | -9.18, H | -12.24, H | -15.88, H | -18.72, L | -19.72, L
| L |   | -3.64, START | -6.12, H | -8.6, H | -11.92, L | -14.66, L | -17.98, L |

#### 2

Based on above table, the sequence of `GCACTG` should be `HHLLLL`.

### 1.2 POS Tagging

The codes consists of three files:
- `preprocess.py` Codes to form the vocabulary and further return the log space of the transition & emission probabilities using the given `α` & `β`
- `viterbi.py` The core logic implementation of the Viterbi algorithm. It uses the log probabilities returned from `preprocess.py` in the score function.
- `file.py` The util to read & write file

#### 1

**K** is set to `2`. The corresponding vocabulary size **V** is `24509`

#### 5

While **α** and **β** are both `1`, the token accuracy of the dev data is `94.75%` and sentence accuracy is `43.75%`

#### 7

When **α** is `0.005` and **β** is `300`, the accuracy of the dev data can achieve the best, where the the token accuracy is `95.65%` and the sentence accuracy is `49.86%`

## Conditional Random Fields

### 1

Below is the list of features added. The accuracy on dev data is `71.14%`

feature name | description
----|----
is_digit  | is the token a digit number
is_upper  | is the token in uppercase
first_letter  | the first letter of the token
first_2_letters  | the first 2 letters of the token
last_2_letters  | the last 2 letters of the token
prev_tok  | previous token
prev_tok_is_digit  | is the previous token a digit number
prev_tok_first_2_letters  | the first 2 letters of the previous token
prev_tok_last_2_letters  | the last 2 letters of the previous token
next_tok  | next token
next_tok_is_digit  | is the next token a digit number
next_tok_first_2_letters  | the last 2 letters of the next token
next_tok_last_2_letters  | the last 2 letters of the next token

### 2

After switching the algorithm from `lbfgs` to `averaged perceptron`, the accuracy on dev data is `85.62%`


Not like CRF built on logistic regression, CRF with average perceptron does not use softmax to normalize the scores and so has no loss function to compute the gradient to update the weights which will further require the forward-backward algorithm. Instead, like perceptron, it will predict a label `yˆ` for the training data. However, while the perceptron enumerates all possible labels to find the one with the max score, it uses the Viterbi algorithm here to efficiently search the tag sequence `yˆ`. If the predicted label is inccorect, it updates the weights by adding the features of correct label and subtracting the feature of the predicted label. Like average perceptron, it maintain the sum of such weights and compute the average to use at the end of training.
