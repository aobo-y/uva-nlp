# Language Modeling

file | desc
-|-
simple_rnnlm.py | base model
stackedlstm_rnnlm | source is copied from simple_rnnlm with only changing layer number config
opt_rnnlm | source is copied from simple_rnnlm with only changing optimizer
model_rnnlm | source is copied from simple_rnnlm with only changing input hidden size config
minibatch_rnnlm | source is copied from simple_rnnlm but changed the logic how to get batch

All support starting with checkpoint argument `--checkpoint=xxx.tar`

## Recurrent Neural Network Language Models

### 1

The `simple_rnn` follows the requirements strictly, but I add an extra Dropout layer with ratio 0.2 for the input.

### 2

The script supports 2 arguments, `--model=x` to specify which model to calculate and `--checkpoint=xxx.tar` to load specific checkpoints.

### 3

Perplexity of the simple_rnnlm

trn | dev
-|-
558.97 | 485.60

### 4

The best Perplexity of the stackedlstm_rnnlm, which contains `3` stacked layers

trn | dev
-|-
517.82 | 436.10

### 5

The best Perplexity of the opt_rnnlm, which adopts the `Adam` algorithm

trn | dev
-|-
273.01 | 344.47

### 6

The best Perplexity of the model_rnnlm, which set both input & hidden size as `256`

trn | dev
-|-
423.09 | 383.12

### 7

I am only able to train the model in batch size `16`, because any larger sizes give me `CUDA memory error` because limited GPU resource available for now.

Considering each iteration now go through `16` times more data, so I lowered the overall training iterations to around the half of the other models, but seems the result are not as good as them.

trn | dev
-|-
853.82 | 663.15

