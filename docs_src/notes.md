# Notes on Benchmark Data


### Mean absolute percentage error (`mape*`) scores
Mean absolute percentage error is only valid on sets of data without any true values of zero.
Also, small true values can result in very large MAPE for samples with even very small predicted error.
A threshold of 1e-5 is applied to mask samples with true absolute values smaller than this from the
MAPE calculation. The reported MAPE is the decimal (not percentage) among these masked samples; i.e., 
a MAPE of 11% corresponds to `mape*=0.11`, and a MAPE of 11,000% corresponds to `mape*=110`.

**Please use the given MAPE scores with a grain of salt, as they are not complete for the reason given above**.