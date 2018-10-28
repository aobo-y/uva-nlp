# Sequence Labeling

## Hidden Markov Models

### 1.1 A Toy Problem

#### 1

Covert Transition Probability to Transition Weight

| | H | L | END
|-|---|---|-----
| Start | -0.74 | -1.32 | −∞
| H | -1.32 | -1.32 | -2.32
| L | -2.32 | -1 | -1.74

Covert Emission Probability to Emission Weight

| | A | C | G | T
|-|---|---|---|---
| H | -2.32 | -1.74 | -1.74 | -2.32
| L | -1.74 | -2.32 | -2.32 | -1.74

The trellis table is as following

| | START | G | C | A | C | T | G | END
|---|---|---|---|---|---|---|---|---
| H | 0 | -2.48, START | -5.54, H | -9.18, H | -12.24, H | -15.88, H | -18.72, L | -19.72, L
| L |   | -3.64, START | -6.12, H | -8.6, H | -11.92, L | -14.66, L | -17.98, L |

### 2

Based on above table, the sequence of `GCACTG` should be `HHLLLL`.

### 1.2 POS Tagging

#### 1

**K** is set to `2`. The corresponding vocabulary size **V** is `24509`

#### 5

While **α** and **β** are both `1`, the accuracy of the dev data is `43.75%`

#### 7

When **α** is `300` and **β** is `0.005`, the accuracy of the dev data can achieve the best, `49.86%`

## Conditional Random Fields

### 1


### 2

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

### 3

After switching the algorithm from `lbfgs` to `averaged perceptron`, the accuracy on dev data is `85.45%`
