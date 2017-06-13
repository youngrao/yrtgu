# yrtgu
## Out-of-Core Online Logistic Regression
The drastic improvement of computer processors has paved the way for the use of increasingly complex and powerful data analysis tools. Similarly, data collection has exploded in size allowing practictioners to take advantage of these new capabilities. However, physical memory has lagged behind its counterparts making it difficult to reliably read large data sets at once. One way to work around this problem is with out-of-core techniques. 

## Description
This simple script, inspired by [VowpalWabbit](http://hunch.net/~vw/) and this [innovative Kaggle user](https://www.kaggle.com/c/tradeshift-text-classification/discussion/10537), implements an out-of-core online logistic regression. Here, we discuss the components of this algorithm. 

To start, it is out-of-core in the sense that it does not need to read an entire data set into memory at once. Instead, it uses an adaptive algorithm, Follow the Regularized Leader - Proximal (FTRL-P), to sequentally train a logistic regression row-by-row. FTRL-P can be seen as a version of online stochastic gradient descent except with L<sub>1</sub> and L<sub>2</sub> regularization. This is the same strategy used by Google in [ad click-through rate prediction](https://research.google.com/pubs/pub41159.html). 

It achieves this by using the so-called hash trick. Essentially, the hash trick takes every pair of column and value and converts it into its hash. Then, these hashes are all converted into a sparse array through one-hot encoding.  Thus, we can run a "large" regression with hundreds of thousands of explanatory variables, each corresponding to one particular pair. Computation is efficient however, because the feature vectors are sparse with most entries equal to zero. It is clear then why FTRL-P is used, as it ensures the sparseness of the model. Theoretically, model updates can be performed in small batches as well.

This script is customized to be able to perform a wide variety of tasks from regression to classification. In the case of multinomial classification, a One-Against-All model is trained which means that we create a separate prediction for each label. As a result, multilabel classification is also supported. 

## Example 1: Otto

## Example 2: Azavu



