# Text Classification

## Perceptron

### 2.1
I did the following to reduce the feature size
- Unify all texts to lower case
- Purge tokens without any alphabetic letters, like numbers & punctuations, through regex `[A-Za-z]`
- Map low frequency words to a special token `UKN`

The vocabulary size is reduced to `13400` including the `UKN` & `OFFSET` token, while the minimum frequency is set to `5`, which means a word needs to appear at leat 5 times to be kept. This vocabulary set is used in the following problems.

### 2.2
Below is my Perceptron's accurary of 10 epochs, with shuffling data after each epoch.

<img width="575" alt="perceptron-plot" src="https://raw.githubusercontent.com/aobo-y/uva-nlp-projects/master/proj01/perceptron_plot.png">

epoch|   trn   |   dev
-----|---------|----------
0    |  0.759  |  0.7496
1    |  0.861  |  0.8358
2    |  0.8351 |  0.8195
3    |  0.8961 |  0.8647
4    |  0.9021 |  0.863
5    |  0.896  |  0.8608
6    |  0.9066 |  0.8702
7    |  0.8981 |  0.8536
8    |  0.8749 |  0.8405
9    |  0.9122 |  0.8679

### 2.3
Below is my Averaged Perceptron's accurary of 10 epochs, with shuffling data after each epoch.

<img width="575" alt="averaged-perceptron-plot" src="https://raw.githubusercontent.com/aobo-y/uva-nlp-projects/master/proj01/averaged_perceptron_plot.png">

epoch|   trn    |   dev
-----|----------|-----------
0    |  0.8733  |  0.8565
1    |  0.8889  |  0.8672
2    |  0.8992  |  0.8744
3    |  0.9042  |  0.8761
4    |  0.9085  |  0.8788
5    |  0.9122  |  0.8797
6    |  0.915   |  0.8816
7    |  0.9175  |  0.8819
8    |  0.9201  |  0.8818
9    |  0.9217  |  0.8821

## Logistic Regression

### 3.1
The size of the feature set is `47963`. Below is the accuracy.

   trn   |   dev
---------|----------
  0.9782 |  0.8802

### 3.2
The size of the feature set is `776704`. Below is the accuracy.

   trn   |   dev
---------|----------
  0.9999 |  0.9044

### 3.3
With regularization L2, after several tries, the `λ` is narrowed into range `[5, 9]`. Below is accuracy.

λ |    trn   |   dev
--|----------|----------
5 |  0.9999  |  0.9044
6 |  0.9971  |  0.9059
7 |  0.9962  |  0.9054
8 |  0.9955  |  0.9053
9 |  0.9946  |  0.9051

### 3.4

<img width="425" alt="l1_desc" src="https://raw.githubusercontent.com/aobo-y/uva-nlp-projects/master/proj01/l1_desc.png">

Based on the definition of 1-norm, the illustration of L1's shape is the square diamond. Compared with L2' shape, which is a spherical, it has higher possiblities to meet the ellipses, which stands for the original cost function, at its angles. The angles of L1 are all on the axises, which means a feature coefficient θ is zero. Therefore, L1 is more possible to make more features has no impacts, which means L1 prefers sparse solutions.

With regularization L1, after several tries, the `λ` is narrowed into ranges `[0.002, 0.005]` and `[2, 4]`

λ     |    trn   |   dev
------|----------|----------
0.002 |    1.0   |  0.899
0.003 |    1.0   |  0.9016
0.004 |    1.0   |  0.8996
0.005 |    1.0   |  0.899
2     |  0.9674  |  0.8984
3     |  0.9512  |  0.9013
4     |  0.9405  |  0.8998

### 3.5

   trn   |   dev
---------|----------
  0.9929 |  0.9063

The above is the best I could achieve. In this model, the texts are vectorized in n-gram ranging from 1 to 2. From the previous results, I have already seen it performs better than mere unigram. I also tried to be more greedy by including the additional trigram, but the accuracy dropped. Then, since it generated more than 700,000 features, I give the vectroizer a minimum document frequency of 2, which can largely reduce the feature size without any noticable impacts on the results. Increasing the frequency or giving another maximum frequency ended with worse accuracy. However, the new vectorizer did not work well with the above narrowed λ range and the corresponding regularization algorithms. Bringing in optimization algorithms made the results even stray further. So I had to retune the classification model by nested-looping different combination of λ, regularization and optimaztion algorithms. After keeping narrowing the range of λ and filtering out badly-performed combinations, the final Logistic Regression adopts L2 regularization with λ of 8.065 (c = 0.124) and the lbfgs optimization algorithm. It gives a dev accuracy of 90.63% which is higher than all my other models.
