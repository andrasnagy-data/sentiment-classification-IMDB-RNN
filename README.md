### Sentiment Classification with RNN (many-to-one) architecture, GloVe 50-d embedding layer

This project is a follow-up of the "simpler" approach in the ![sentiment-classification-IMDB](https://github.com/andrasnagy-data/sentiment-classification-IMDB).
The words are represented in continuous space, and the architecture is based on pytorch's GRU. (Reasons: addresses the "vanishing gradient problem, faster to train, than LSTM))

The best point to stop the training is after the 20th epoch (lowest testing error).From that point, the testing loss stops decreasing.

![image](training_evaluation.png)

