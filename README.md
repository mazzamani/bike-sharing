# bike-sharing
predicting bike sharing count per hour 

Here are the simple baseline methods:

|              Baseline Method              |   MAE  |   STD  |
|:-----------------------------------------:|:------:|:------:|
|            Previous Hour Count            |  85.06 |  97.70 |
|          Previous Day, Same Hour          |  82.21 |        |
|              Keep Hourly Pace             | 89.21  | 125.17 |
| Linear Regression, without regularization | 142.21 | 115.59 |

## Method
In this repository I have used RNN-GRU with different configurations. First, the dataset is divided into 70% training set, 10% validation set and 20% test set. Then, the bike count is normalized respect to the maximum bike count number in the training set. The network is trained with the relative count respect to the previous day (same hour) or the previous hour. It means in this approach we have one of these two assumptions: either (1) we already know the count of the previous hour and we need to predict the next hour or (2) we know the previous day data (which is more realistic). The proposed architecture calculates the relative change and its value is converted into the absolute value for getting the absolute count number. Also, the relative change was trained by converting it into discrete levels (here: [-1]+[-.75:0.05:0.45]+[0.5:0.25:2.5])
 
Here is the results with Gated Recurrent Unit (GRU): 

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
