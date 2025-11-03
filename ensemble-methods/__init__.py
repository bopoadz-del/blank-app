"""
Ensemble Methods Framework

A comprehensive framework for ensemble learning methods including:
- Bagging (Bootstrap Aggregating)
- Boosting (AdaBoost, Gradient Boosting, XGBoost, LightGBM, CatBoost)
- Stacking (Stacked Generalization)
- Voting (Hard and Soft Voting)
- Blending (Holdout-based Stacking)

Author: ML Framework Team
Version: 1.0.0
"""

# Bagging methods
from .bagging import (
    BaggingEnsemble,
    RandomForestEnsemble,
    ExtraTreesEnsemble
)

# Boosting methods
from .boosting import (
    AdaBoostEnsemble,
    GradientBoostingEnsemble,
    XGBoostEnsemble,
    LightGBMEnsemble,
    CatBoostEnsemble,
    XGBOOST_AVAILABLE,
    LIGHTGBM_AVAILABLE,
    CATBOOST_AVAILABLE
)

# Stacking methods
from .stacking import (
    StackingEnsemble,
    MultiLevelStacking
)

# Voting methods
from .voting import (
    VotingEnsemble,
    WeightedVotingEnsemble
)

# Blending methods
from .blending import (
    BlendingEnsemble,
    MultiLayerBlending
)

__all__ = [
    # Bagging
    'BaggingEnsemble',
    'RandomForestEnsemble',
    'ExtraTreesEnsemble',

    # Boosting
    'AdaBoostEnsemble',
    'GradientBoostingEnsemble',
    'XGBoostEnsemble',
    'LightGBMEnsemble',
    'CatBoostEnsemble',
    'XGBOOST_AVAILABLE',
    'LIGHTGBM_AVAILABLE',
    'CATBOOST_AVAILABLE',

    # Stacking
    'StackingEnsemble',
    'MultiLevelStacking',

    # Voting
    'VotingEnsemble',
    'WeightedVotingEnsemble',

    # Blending
    'BlendingEnsemble',
    'MultiLayerBlending',
]

__version__ = '1.0.0'
__author__ = 'ML Framework Team'
