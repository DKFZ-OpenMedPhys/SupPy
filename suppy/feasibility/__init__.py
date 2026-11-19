"""Module for linear feasibility and split feasibility algorithms."""
from suppy.utils import Bounds
from ._bands._ams_algorithms import (
    SequentialAMSHyperslab,
    SequentialWeightedAMSHyperslab,
    SimultaneousAMSHyperslab,
    StringAveragedAMSHyperslab,
    BlockIterativeAMSHyperslab,
    SARTHyperslab,
)


from ._bands._ams_extrapolations import (
    LandweberHyperslab,
    ExtrapolatedLandweberHyperslab,
    AdaptiveStepLandweberHyperslab,
    BlockIterativeExtrapolatedLandweberHyperslab,
    BlockIterativeAdaptiveStepLandweberHyperslab,
)
from ._bands._fista import FISTAHyperslab
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
    LandweberHalfspace,
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

from ._hyperplanes._variants import DROPHyperplane

from ._hyperplanes._kaczmarz_extrapolations import (
    LandweberHyperplane,
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
    "LandweberHyperslab",
    "ExtrapolatedLandweberHyperslab",
    "AdaptiveStepLandweberHyperslab",
    "BlockIterativeExtrapolatedLandweberHyperslab",
    "BlockIterativeAdaptiveStepLandweberHyperslab",
    "FISTAHyperslab",
    "SequentialAMSHalfspace",
    "SequentialWeightedAMSHalfspace",
    "SimultaneousAMSHalfspace",
    "StringAveragedAMSHalfspace",
    "BlockIterativeAMSHalfspace",
    "LandweberHalfspace",
    "ExtrapolatedLandweberHalfspace",
    "AdaptiveStepLandweberHalfspace",
    "KaczmarzMethod",
    "SequentialWeightedKaczmarz",
    "SimultaneousKaczmarzMethod",
    "StringAveragedKaczmarz",
    "BlockIterativeKaczmarz",
    "LandweberHyperplane",
    "ExtrapolatedLandweberHyperplane",
    "AdaptiveStepLandweberHyperplane",
    "SequentialARM",
    "SimultaneousARM",
    "StringAveragedARM",
    "SequentialART3plus",
    "CQAlgorithm",
    "DROPHyperplane",
    "SARTHyperslab",
]
