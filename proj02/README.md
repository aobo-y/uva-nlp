# Sequence Labeling

## Hidden Markov Models

### 1.1 - 1

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

### 1.1 - 2

Based on above table, the sequence of `GCACTG` should be `HHLLLL`.

### 1.2 - 1

**K** is set to `2`. The corresponding vocabulary size **V** is `24509`

### 1.2 - 5

While **α** and **β** are both `1`, the accuracy of the dev data is `43.75%`

### 1.2 - 7

When **α** is `300` and **β** is `0.005`, the accuracy of the dev data can achieve the best, `49.86%`
