# Cross Validation in Time Series
## $k$-Fold Cross Validation
$k$-Fold cross validation is a technique used to evaluate the performance of a
model. $k$-Fold cross validation works as follows:

1. Split the dataset $\mathcal{D}$ into $k$ folds or non-overlapping subsets.
   $\mathcal{D} = \{\mathcal{D}_i | i \in [1..k]\}$
2. For $i = 1 \rightarrow k$, train the model on $\mathcal{D}_j$ where $j\neq i$
   and evaluate the model on $\mathcal{D}_i$.
3. Report the performance of the model as its average performance on the $k$
   evaluation/test folds.

This kind of cross validation would not be effective for time series forecasting
because time series data may be non-stationary, so using randomly sampled events
to train a model would not help it learn the trend underlying the data.

## Rolling Basis Cross Validation
In rolling basis cross validation, your test set always follows your training
set in time, and you continually move your test set along in time.

### Time Series Split Cross-Validation
### Blocked Cross-Validation

