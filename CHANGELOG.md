# Changelog
All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-5

### Added

- Initial release

## [0.2.1] - 2025-02-07

### Added

- Added linear algorithms for Halfspace and Hyperplane problems
- Better GPU/CPU differentiaton
- More proximity measure options
- More example files
- General code cleanup

### Removed


## [0.3.0] - 2026-02-03

### Added

- Added various new algorithms for feasibility seeking
- Aded option for custom projection algorithms
- Added perturbation strategies with adaptive step sizes
- Added option for custom stopping criteria
- Removed old superiorization module
- Proper volume based projections
- Better documentation

### Removed


## [0.4.0] - 2026-05-08

### Added

- General Code/Docstring cleanup
- Runtime improvements for perturbations
- Added more options to store intermediate results

### Removed

## [Unreleased]

### Added

- New `suppy.optimization` module with `FISTA` and `FISTABacktracking` as reference
  constrained-optimization algorithms (original Beck-Teboulle settings, optional restarts)
- `FISTAHyperslab` feasibility algorithm
- Dedicated Landweber implementations: `LandweberHyperslab`, `LandweberHalfspace`,
  `LandweberHyperplane`
- `BlockIterativeAdaptiveStepLandweberHyperslab` (block-iterative Landweber with adaptive step)
- `ExtrapolatedSimultaneousProjectionHyperslab`: the row-normalized extrapolated algorithm
  (the `ExtrapolatedLandweberHyperslab` of 0.4.0, see Changed)
- Experimental `BarzilaiBorweinPerturbation` perturbation strategy (private, in
  `suppy.perturbations._variants`; not part of the public API)
- Options to limit (`max_restarts`) and customize (`restart_value`) the restarts of perturbations
- `callback` option for feasibility, projection and superiorization solvers
- Optional storage of values after the projection step (`store_function_reduction`)
- Slice plotting utilities for radiotherapy: `plot_slice`, `scroll_slices`, `matrad_to_zyx`,
  `matrad_indices_to_mask`
- `Bounds.violation()` helper for proximity calculations
- Extended test suite for projections, feasibility algorithms, perturbations and superiorization

### Changed

- Fewer GPU/host synchronizations in feasibility algorithms, projections, perturbations,
  stopping criteria and superiorization drivers; large problems run faster on GPU and CPU
  (results agree with the previous version up to ~1e-13 relative)
- Radiotherapy objectives cache the dose and skip the matrix multiplication for repeated
  evaluations at the same point
- `FuncWrapper` and residual calculations avoid redundant device syncs
- **Behavior change:** `ExtrapolatedLandweberHyperslab` (and its block-iterative variant) now
  uses plain constraint weights (gradient of the weighted squared residual) instead of
  row-normalized weights; results differ from 0.4.0. The previous behavior is available as
  `ExtrapolatedSimultaneousProjectionHyperslab`
- Updated documentation

### Fixed

- `ExtrapolatedLandweberHalfspace`: a degenerate step (0/0) now results in no step instead of
  a NaN error, consistent with the hyperslab variant
- `SimultaneousART3plus`: element-wise negation (`not` replaced by `~`)
- Several minor bugs in the code and tests
