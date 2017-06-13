# yrtgu
## Out-of-Core Online Logistic Regression
The drastic improvement of computer processors has paved the way for the use of increasingly complex and powerful data analysis tools. Similarly, data collection has exploded in size allowing practictioners to take advantage of these new capabilities. However, physical memory has lagged behind its counterparts making it difficult to reliably read large data sets at once. One way to work around this problem is with out-of-core techniques. 

## Description
This simple script, inspired by [VowpalWabbit](http://hunch.net/~vw/) and this [innovative Kaggle user](https://www.kaggle.com/c/tradeshift-text-classification/discussion/10537), implements an out-of-core online logistic regression. Here, we discuss the components of this algorithm. 

To start, it is out-of-core in the sense that it does not need to read an entire data set into memory at once. Instead, it uses an adaptive algorithm, Follow the Regularized Leader - Proximal (FTRL-P), to sequentally train a logistic regression row-by-row. FTRL-P can be seen as a version of online stochastic gradient descent except with L<sub>1</sub> and L<sub>2</sub> regularization. This is the same strategy used by Google in [ad click-through rate prediction](https://research.google.com/pubs/pub41159.html). 

Next, we move on to the actual machinery to see how we can estimate an intricate model while only using minimal memory. It achieves this by using the so-called hash trick. Essentially, the hash trick takes every pair of column and value and converts it to its hash value. Then, these hashes are all converted into a sparse array through one-hot encoding.  Thus, we can run a "large" regression with hundreds of thousands of explanatory variables, each corresponding to one particular pair. Computation is efficient however, because the feature vectors are sparse with most entries equal to zero. It is clear then why FTRL-P is used, as it ensures the sparseness of the model update. The updating occurs by applying this method to small batches of data at a time, where we have set batch size to one.

This script is customized to be able to perform a wide variety of tasks from regression to classification. It is also able to easily generate interactions between all of the features in a "kitchen sink" style. In the case of multinomial classification, a One-Against-All model is trained which outputs a separate prediction for each label. As a result, multilabel classification is also supported. 

It is recommended that yrtgu is used with [PyPy](https://pypy.org/) as this leaner implementation runs much faster. 

## Example 1: Otto Group Product Classification Challenge
We provide here a quick application to the [Otto classification competition](https://www.kaggle.com/c/otto-group-product-classification-challenge). 

Directions: Download `yrtgu.py`, `trainOH.csv` and run `yrtgu-otto.py`.

With PyPy, this should only take a few seconds. This single model scores a 0.70950 on the private leaderboard putting it in 2617th place putting it in the bottom 25 percentile. While this may seem dismal at first, note that this is a relatively small data set (only 12MB) so there is not enough data to overcome the sparse nature of the algorithm. Thus, it is included simply as a demonstration. Note further that this example is run with only one epoch, and increasing this number steadily improves the model. 


## Example 2: Azavu
The [Azavu Click-Through Rate Prediction](https://www.kaggle.com/c/avazu-ctr-prediction) competition provides a better setting for showcasing the true capabilities of yrtgu as it is a similar setting as used by Google with great success. 

Directions: Download 'yrtgu.py', training and test data from [here](https://www.kaggle.com/c/avazu-ctr-prediction/data), and `yrtgu-avazu.py`. Note that the training data is nearly a whopping 6GB when uncompressed! While it is certainly possible to read this into memory, model processing and usage will overall be slower. This is where out-of-core techniques really shine.

With PyPy, the runtime is about 10 minutes as opposed to roughly two hours with a standard python implementation. This achieves a private leaderboard score of 0.3936633 attaining 633rd place and top 41 percentile! Not bad especially considering it is a single model with no parameter tuning and only a single pass epoch. 



