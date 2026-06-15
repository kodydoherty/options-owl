"""Shared feature modules — single source of truth for ML feature math.

A feature is defined exactly ONCE here so the trainer
(``scripts/train_ml_models_v3.py``) and the live serving path
(``options_owl/sourcing/ml_pipeline.py``) compute identical features by
construction, eliminating train/serve skew.
"""

from options_owl.sourcing.features.regime_features import (
    REGIME_FEATURE_ORDER,
    compute_regime_feature_vector,
    load_serving_inputs,
    load_training_inputs,
)

__all__ = [
    "REGIME_FEATURE_ORDER",
    "compute_regime_feature_vector",
    "load_serving_inputs",
    "load_training_inputs",
]
