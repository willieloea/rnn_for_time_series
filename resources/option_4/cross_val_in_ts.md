# Cross Validation in Time Series
Notes from reading: https://medium.com/@soumyachess1496/cross-validation-in-time-series-566ae4981ce4

## Traditional $k$-Fold Cross Validation
$k$-Fold cross validation is a technique used to evaluate the performance of a
model. $k$-Fold cross validation works as follows:

1. Split the dataset $\mathcal{D}$ into $k$ folds or non-overlapping subsets.
   $\mathcal{D} = \{\mathcal{D}_i | i \in [1..k]\}$
2. For $i = 1 \rightarrow k$, train the model on $\mathcal{D}_j$ where $j\neq i$
   and evaluate the model on $\mathcal{D}_i$.
3. Report the performance of the model as its average performance on the $k$
   evaluation/test folds.

This kind of cross validation would not be effective for time series forecasting
because it relies on the assumption that data points are independent and
identically distributed.

In time series data:
- Temporal Dependency: Observations are sequentially correlated (autocorrelation).
- Lookahead Bias (Data Leakage): Shuffling or randomly sampling folds allows
  values from the future to be present in the training set while predicting the 
  past, leading to overly optimistic and invalid evaluation metrics.

## Cross-Validation on a Rolling Basis (Forward Chaining)

To respect temporal order, time series evaluation uses a rolling basis.

### Rules for Splitting:
1. Every test set contains strictly unique observations.
2. Training observations must always occur before their corresponding test
   observations.

### Example Walkthrough ($N = 5$ observations, 4 splits)
Given data: `[1, 2, 3, 4, 5]`
- **Fold 1:** Train: `[1]` | Test: `[2]`
- **Fold 2:** Train: `[1, 2]` | Test: `[3]`
- **Fold 3:** Train: `[1, 2, 3]` | Test: `[4]`
- **Fold 4:** Train: `[1, 2, 3, 4]` | Test: `[5]`


## Blocked Cross-Validation

Even with walk-forward splits, subtle forms of data leakage can occur:
- Lag Feature Leakage: If lag features are used, a lag value at the boundary
  might bridge the gap between training and validation sets.
- Pattern Memorization: High autocorrelation across adjacent time steps can
  cause adjacent validation sets to mirror the immediate past too closely.

### The Solution: Adding Margins / Gaps
Blocked Cross-Validation introduces margins in two places:
1. Between Training and Validation Folds: Prevents observations used as lag
   features from overlapping into target responses.
2. Between Consecutive Iterations: Prevents the model from memorizing cyclical
   or high-frequency transition patterns between folds.

## Nested Cross-Validation for Single Time Series

Nested CV is used when hyperparameter tuning (inner loop) must be evaluated
without biasing the final performance estimate (outer loop).

### 1. Predict Second Half
- Mechanism: The time series is split temporally in two: the first half is
  allocated to training and hyperparameter selection, and the second half acts
  as the test set.
- Pros: Fast and simple to implement (single split).
- Cons: High variance and potential bias due to the arbitrary selection of a
  single hold-out test period.

### 2. Day Forward-Chaining
- Mechanism: Data is divided into discrete time units (e.g., days). The test set
  rolls forward one day at a time, using all preceding days as the
  training/validation pool.
- Pros: Produces multiple independent error measurements, yielding an unbiased
  and robust error estimate.
- Cons: Computationally expensive due to many re-training iterations.

## Nested Cross-Validation with Multiple Time Series

When working with multiple independent entities (e.g., multiple medical
patients, retail stores, or IoT sensors):

### A. Regular Nested Cross-Validation
- Strategy: Applies identical temporal cutoffs across all entities
  simultaneously.
- Example: For Subjects A and B, the training set uses Days 1–10 of both Subject
  A and Subject B, while testing uses Days 11–20 of both.
- Constraint: Maintains strict temporal alignment across all entities.

### B. Population-Informed Nested Cross-Validation
- Strategy: Capitalizes on the statistical independence between distinct
  entities.
- Mechanism:
  - The validation and test sets come only from the target entity (e.g.,
    Participant A on Day 18).
  - The training set includes all historical data of Participant A up to Day 17,
    along with all available data (past, present, and future) from independent
    participants (B, C, D, etc.).
- Why it works without leakage: Because participant processes are strictly
  independent, using Participant B's "future" observations does not provide any
  lookahead signal regarding Participant A's timeline.

## Trade-off Summary Matrix

| Method | Number of Splits ($k$ participants, $d$ days) | Computational Cost | Risk of Bias / Overfitting |
|---|---|---|---|
| Predict Second Half | 1 | Low | High (sensitive to test-period anomalies) |
| Day Forward-Chaining | $d - \text{initial window}$ | Moderate to High | Low (robust across time horizons) |
| Regular Multiple TS | Depends on base strategy | Moderate | Low across cohorts, sensitive to global shocks |
| Population-Informed | $k \times (d - \text{window})$ | High | Lowest (maximizes cross-individual transfer) |

## Summary

- Standard random $K$-fold cross-validation must never be applied directly to
  time-dependent observations.
- Always enforce chronological direction: Train $\rightarrow$ Validate
  $\rightarrow$ Test.
- Use margins/blocking to combat lag-variable leakage.
- For independent multi-entity panels, Population-Informed CV maximizes sample
  size while preventing target-entity leakage.
