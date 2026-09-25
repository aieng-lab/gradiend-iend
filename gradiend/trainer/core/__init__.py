"""
Core training components for GRADIEND models.
"""

from .training import train
from .callbacks import (
    TrainingCallback,
    EvaluationCallback,
    CheckpointCallback,
    NormalizationCallback,
    LoggingCallback,
    get_default_callbacks,
)
from .transition_selection import TransitionSpec, pair, identity, expand_transition_selection
from .signals import (
    Signal,
    SignalSet,
    SignalScope,
    SignalSpace,
    SignalBatch,
    GradientSignalExtractor,
    ActivationSignalExtractor,
)
from .dataset import SignalTrainingDatasetBase
from .decoder_lr import AutoDecoderLearningRate, AUTO_DECODER_LR
from .lr_search import LRSearch, LRSearchResult, RunState, tune_learning_rate
from .signal_checks import SignalDiversityCallback, SignalNotDiverseError, assert_signal_diverse

__all__ = [
    'train',
    'TrainingCallback',
    'EvaluationCallback',
    'CheckpointCallback',
    'NormalizationCallback',
    'LoggingCallback',
    'get_default_callbacks',
    'TransitionSpec',
    'pair',
    'identity',
    'expand_transition_selection',
    'Signal',
    'SignalSet',
    'SignalScope',
    'SignalSpace',
    'SignalBatch',
    'GradientSignalExtractor',
    'ActivationSignalExtractor',
    'SignalTrainingDatasetBase',
    'AutoDecoderLearningRate',
    'AUTO_DECODER_LR',
    'LRSearch',
    'LRSearchResult',
    'RunState',
    'tune_learning_rate',
    'assert_signal_diverse',
    'SignalDiversityCallback',
    'SignalNotDiverseError',
]
