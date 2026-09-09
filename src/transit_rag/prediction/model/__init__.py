"""Delay prediction: the naive-persistence baseline and the XGBoost model.

Split into small modules because the interesting failures here are silent. A
model that scores well because a feature leaked the answer looks exactly like a
model that works, so :mod:`dataset` owns the feature contract and is tested
against it directly, separately from anything that fits.

The baseline is not a formality. Naive persistence -- "this trip's delay at the
next stop equals its last observed delay" -- is a strong predictor on a network
where most trains are on time, and it is the number the model has to beat for
the prediction layer to be worth including at all.
"""
