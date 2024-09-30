## Test CCA with third-order shear stats

Goal: Understand how the CCA is efficient for third-order shear stats

Procedure

1. Make a chain of 2pt -> we want to have equaly weighted chain.
2. For each sample of chain, we make a prediction of map3.
    1. Make param file from chain
    2. source to cosmosis using list sampler
3. Apply CCA to get the compression matrix.
