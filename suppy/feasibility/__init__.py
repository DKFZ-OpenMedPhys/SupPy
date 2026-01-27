"""Module for linear feasibility and split feasibility algorithms."""
from suppy.utils import Bounds
from ._bands._ams_algorithms import (
    SequentialAMSHyperslab,
    SequentialWeightedAMSHyperslab,
    SimultaneousAMSHyperslab,
    StringAveragedAMSHyperslab,
    BlockIterativeAMSHyperslab,
)


from ._bands._ams_extrapolations import (
    ExtrapolatedLandweberHyperslab,
    AdaptiveStepLandweberHyperslab,
)
from ._bands._arm_algorithms import SequentialARM, SimultaneousARM, StringAveragedARM
from ._bands._art3_algorithms import SequentialART3plus


from ._halfspaces._ams_algorithms import (
    SequentialAMSHalfspace,
    SequentialWeightedAMSHalfspace,
    SimultaneousAMSHalfspace,
    StringAveragedAMSHalfspace,
    BlockIterativeAMSHalfspace,
)

from ._halfspaces._ams_extrapolations import (
    ExtrapolatedLandweberHalfspace,
    AdaptiveStepLandweberHalfspace,
)

from ._hyperplanes._kaczmarz_algorithms import (
    KaczmarzMethod,
    SequentialWeightedKaczmarz,
    SimultaneousKaczmarzMethod,
    StringAveragedKaczmarz,
    BlockIterativeKaczmarz,
)

from ._hyperplanes._kaczmarz_extrapolations import (
    ExtrapolatedLandweberHyperplane,
    AdaptiveStepLandweberHyperplane,
)

from ._split_algorithms import CQAlgorithm

__all__ = [
    "SequentialAMSHyperslab",
    "SequentialWeightedAMSHyperslab",
    "SimultaneousAMSHyperslab",
    "StringAveragedAMSHyperslab",
    "BlockIterativeAMSHyperslab",
    "ExtrapolatedLandweberHyperslab",
    "AdaptiveStepLandweberHyperslab",
    "SequentialAMSHalfspace",
    "SequentialWeightedAMSHalfspace",
    "SimultaneousAMSHalfspace",
    "StringAveragedAMSHalfspace",
    "BlockIterativeAMSHalfspace",
    "ExtrapolatedLandweberHalfspace",
    "AdaptiveStepLandweberHalfspace",
    "KaczmarzMethod",
    "SequentialWeightedKaczmarz",
    "SimultaneousKaczmarzMethod",
    "StringAveragedKaczmarz",
    "BlockIterativeKaczmarz",
    "ExtrapolatedLandweberHyperplane",
    "AdaptiveStepLandweberHyperplane",
    "SequentialARM",
    "SimultaneousARM",
    "StringAveragedARM",
    "SequentialART3plus",
    "CQAlgorithm",
]
