# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`SupPy` is a pure-Python library for building and running **superiorization** and **feasibility-seeking** algorithms (published on PyPI as `suppy`). Feasibility-seeking finds a point in the intersection of convex sets; superiorization perturbs such an algorithm to also reduce an objective function without solving a full constrained optimization. A primary application is IMRT radiotherapy treatment planning.

## Commands

```bash
pip install -e .[dev]          # install with dev tooling (pytest, coverage, pre-commit, sphinx, ...)
pre-commit install             # set up black + pydocstringformatter hooks

pytest tests                   # run the full unit-test suite
pytest tests/superiorization/test_standard_superiorization_new.py   # single file
pytest tests/... -k test_name  # single test by name
pytest examples --nbmake       # execute the example notebooks as tests (CI does this)

pytest tests --cov=suppy --cov-report term   # with coverage (CI uses xml/html reports too)

black --check .                # formatting check (line-length 99)
flake8 --toml-config ./pyproject.toml        # lint
pylint suppy                   # lint
sphinx-build docs .doc         # build documentation
```

CI (`.gitlab-ci.yml`) runs lint (black/flake8/pylint), tests + notebook execution, and the docs build (docs/pages only on `main`).

## Architecture

### `Projection` is the universal base class
Almost everything is a `Projection` ([suppy/projections/_projections.py](suppy/projections/_projections.py)). This includes simple set projections, *combinations* of projections, **and** the linear feasibility algorithms. The contract:
- Subclasses implement `_project(x)` (the raw step) and `_proximity(x, measures)`; the public `project()`/`proximity()` wrap them (applying `relaxation` and the `proximity_flag`).
- `solve(x, max_iter, ...)` runs `project` iteratively until a proximity-based stopping criterion is met, optionally storing iterates (`storage`/`storage_iters`) and proximity history.

Because combinations are themselves `Projection`s, they nest arbitrarily.

### Layered modules (each layer consumes the one above)
- **`suppy.projections`** — concrete set projections ([_basic_projections.py](suppy/projections/_basic_projections.py): `BallProjection`, `HalfspaceProjection`, `BoxProjection`, DVH projections, ...), the combination methods ([_projection_methods.py](suppy/projections/_projection_methods.py): `SequentialProjection`, `SimultaneousProjection`, `StringAveragedProjection`, `BlockIterativeProjection`), and subgradient projections. The four combination patterns (sequential / simultaneous / string-averaged / block-iterative) recur throughout the codebase.
- **`suppy.feasibility`** — linear-system algorithms for `lb ≤ Ax ≤ ub`. `Feasibility` ([_linear_algorithms.py](suppy/feasibility/_linear_algorithms.py)) subclasses `Projection`. Implementations are grouped by constraint geometry: `_bands/` (hyperslabs: AMS, ARM, ART3+), `_halfspaces/`, `_hyperplanes/` (Kaczmarz, DROP). Each geometry mirrors the same sequential/simultaneous/string-averaged/block-iterative variants. `_split_algorithms.py` holds the split-feasibility `CQAlgorithm`.
- **`suppy.perturbations`** — perturbation schemes for superiorization ([_base.py](suppy/perturbations/_base.py)). `ObjectivePerturbation` → `GradientPerturbation` → concrete schemes (`PowerSeriesGradientPerturbation`, `AdaptiveStepGradientPerturbation`). Implement `_function_reduction_step(x)`; optional `pre_step`/`post_step`/`reset` hooks are called by the superiorization driver each iteration.
- **`suppy.superiorization`** — the driver. `Superiorization(basic, perturbation_scheme)` ([_standard_sup.py](suppy/superiorization/_standard_sup.py)) alternates `perturbation_scheme.perturbation_step` with `basic.step` each iteration, tracking objective values + proximities (with separate `*_basic` / `*_function_reduction` histories). `_split_sup.py` is the split-feasibility variant.
- **`suppy.optimization`** — reference constrained-optimization algorithms (e.g. FISTA) for comparison, not part of the superiorization path.

### Cross-cutting conventions
- **Optional GPU.** Every numerical file uses the pattern `try: import cupy as cp; NO_GPU = False; except ImportError: cp = np; NO_GPU = True`, then picks the backend at runtime with `xp = cp if isinstance(x, cp.ndarray) else np`. New numerical code must stay backend-agnostic this way and avoid hard numpy assumptions. Arrays are converted back to numpy for storage via `.get()`.
- **`LinearMapping`** ([suppy/utils/_array_helper.py](suppy/utils/_array_helper.py)) wraps the system matrix `A` to unify dense/sparse × numpy/scipy/cupy behind one interface (`single_map`, `row_norm`, `update_step`, ...). Use it instead of touching `A` directly in linear algorithms.
- **`@ensure_float_array`** ([suppy/utils/_decorators.py](suppy/utils/_decorators.py)) decorates every public `solve` entry point to coerce inputs to float64.
- **`FuncWrapper`** binds objective/gradient callables with their extra args so perturbations can call `self.func(x)` / `self.grad(x)`.
- **Proximity measures** are passed as a list like `[("p_norm", 2)]` or `"max_norm"`; only the first measure drives the stopping criterion.

## Conventions
- Style: PEP 8 naming, **NumPy-style docstrings** (enforced by `pydocstringformatter` + rendered by `numpydoc`). Never use Google/reST docstring style.
- Line length 99 for code; docstrings wrap at 79.
- Public API per submodule is curated in each `__init__.py` via `__all__`; new public classes must be added there (and to `docs/api/` for documentation).
- Module files are underscore-prefixed (`_basic_projections.py`); names are re-exported from the package `__init__`.
