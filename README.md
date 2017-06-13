# yrtgu
## Out-of-Core Online Logistic Regression
The drastic improvement of computer processors has paved the way for the use of increasingly complex and powerful data analysis tools. Similarly, data collection has exploded in size allowing practictioners to take advantage of these new capabilities. However, physical memory has lagged behind its counterparts making it difficult to reliably read large data sets at once. One way to work around this problem is with out-of-core techniques. 

## Description
This simple script, inspired by [VowpalWabbit](http://hunch.net/~vw/) and this [innovative Kaggle user](https://www.kaggle.com/c/tradeshift-text-classification/discussion/10537) implements an out-of-core online logistic regression. Let us break down what that means.

First, it is out-of-core in the sense that it does not need to read an entire data set into memory at once. Instead, it uses an adaptive algorithm, Follow the Regularized Leader - Proximal (FTRL-P), to sequentally train a logistic regression row by row. FTRL-P can be seen as a version of online stochastic gradient descent except with additional L<sub>1</sub> This is the a similar strategy used by Google in [ad click-through rate prediction](https://research.google.com/pubs/pub41159.html). 

It achieves this by using the so-called hash trick. Essentially, the hash trick takes every pair of column and value and converts it into its hash. Then, we can run a "large" regression with hundreds of thousands of explanatory varibles, each corresponding to one particular pair. the FTRL-P algorithm ensures that model coefficients are adjusted at a rate that allows for quick convergence, but with 
