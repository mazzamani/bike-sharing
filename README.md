# bike-sharing
predicting bike sharing count per hour

|              Baseline Method              |   MAE  |   STD  |
|:-----------------------------------------:|:------:|:------:|
|            Previous Hour Count            |  85.06 |  97.70 |
|          Previous Day, Same Hour          |  82.21 |        |
|              Keep Hourly Pace             | 89.21  | 125.17 |
| Linear Regression, without regularization | 142.21 | 115.59 |


|             RNN-GRU input (output is Y(t))             |    MAE   |    STD   |
|:------------------------------------------------------:|:--------:|:--------:|
|                          X(t)                          |  85.7497 | 146.7500 |
|                     [X(t-1), X(t)]                     | 76.2733  | 123.4682 |
|                 [X(t-2), X(t-1), X(t)]                 |  69.6842 | 104.5788 |
|                     [X(t), Y(t-1)]                     |  46.4130 |  77.1480 |
|           [[X(t-1), Y(t-2)], [X(t), Y(t-1)]]           |  12.1994 |  19.0140 |
|                     [X(t), Y(t-24)]                    |  64.7710 | 109.2530 |
|                [X(t), Y(t-24), Y(t-48)]                | 15.7937  |  37.0861 |
|          [[X(t-1), Y(t-25)], [X(t), Y(t-24)]]          |  58.8632 |  85.9595 |
| [[X(t-1), Y(t-25), Y(t-49)], [X(t), Y(t-24), Y(t-48)]] |  16.2291 |  35.1439 |
