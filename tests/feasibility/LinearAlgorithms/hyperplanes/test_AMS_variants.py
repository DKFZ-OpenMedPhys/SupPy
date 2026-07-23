import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import DROPHyperplane


@pytest.fixture
def get_sparse_variables():
    A = np.array([[1, 0], [1 / 2, -1 / 2], [0, 1]])
    A = sparse.csr_matrix(A)
    b = np.array([1, 0, 1])
    return A, b


@pytest.fixture
def get_DROPHyperplane_input_sparse(get_sparse_variables):
    A, b = get_sparse_variables
    return DROPHyperplane(A, b), A, b


def test_DROPHyperplane_project_sparse(get_DROPHyperplane_input_sparse):
    alg, A, b = get_DROPHyperplane_input_sparse
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([1.5, 1.5]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.5, 0.5]))


# ---------------------------------------------------------------------------
# DROPHyperplane: options and branches
# ---------------------------------------------------------------------------


def test_DROPHyperplane_constructor_state(get_sparse_variables):
    A, b = get_sparse_variables
    alg = DROPHyperplane(A, b)
    # both columns have 2 nonzero entries -> inverse counts are 1/2
    assert np.allclose(alg.inv_nnz_counts, np.array([0.5, 0.5]))


def test_DROPHyperplane_dense_input_unsupported(get_sparse_variables):
    # per the docstring A must support getnnz(axis=0); a dense matrix does
    # not (current contract)
    A, b = get_sparse_variables
    with pytest.raises(AttributeError):
        DROPHyperplane(A.toarray(), b)


def test_DROPHyperplane_algorithmic_relaxation(get_sparse_variables):
    A, b = get_sparse_variables
    alg = DROPHyperplane(A, b, algorithmic_relaxation=2.0)
    # x = [2, 2]: residuals [-1, 0, -1], row updates sum to [-1, -1];
    # x += 1/2 * 2.0 * [-1, -1] = [1, 1]
    proj = alg.project(np.array([2.0, 2.0]))
    assert np.allclose(proj, np.array([1.0, 1.0]))


def test_DROPHyperplane_relaxation(get_sparse_variables):
    A, b = get_sparse_variables
    alg = DROPHyperplane(A, b, relaxation=0.5)
    # the unrelaxed step is [2, 2] -> [1.5, 1.5]; the outer relaxation
    # halves it: [2, 2] + 0.5 * ([1.5, 1.5] - [2, 2]) = [1.75, 1.75]
    proj = alg.project(np.array([2.0, 2.0]))
    assert np.allclose(proj, np.array([1.75, 1.75]))


def test_DROPHyperplane_proximity(get_DROPHyperplane_input_sparse):
    # proximity is inherited from SimultaneousKaczmarzMethod (weights 1/3):
    # x = [2, 2] -> |residuals| = [1, 0, 1] -> 1/3 * 2 = 2/3
    alg, _, _ = get_DROPHyperplane_input_sparse
    prox = alg.proximity(np.array([2.0, 2.0]), [("p_norm", 2), "max_norm"])
    assert np.abs(prox[0] - 2 / 3) < 1e-10
    assert np.abs(prox[1] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_DROPHyperplane_solve_converges_to_solution(get_sparse_variables):
    # the system is consistent with unique solution [1, 1]
    A, b = get_sparse_variables
    rng = np.random.default_rng(42)
    for _ in range(3):
        alg = DROPHyperplane(A, b)
        x = 4 * rng.standard_normal(2)
        x_n = alg.solve(x, max_iter=1000)
        # prox_tol=1e-6 with weights 1/3 only guarantees residuals ~1.7e-3
        assert np.allclose(x_n, np.array([1.0, 1.0]), atol=1e-2)
        assert alg.proximities[-1][0] < 1e-6


def test_DROPHyperplane_project_idempotent_on_solution(get_sparse_variables):
    A, b = get_sparse_variables
    alg = DROPHyperplane(A, b)
    x = np.array([1.0, 1.0])
    assert np.allclose(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_DROPHyperplane_solve_int_input(get_sparse_variables):
    A, b = get_sparse_variables
    alg = DROPHyperplane(A, b)
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_DROPHyperplane_cupy_backend(get_sparse_variables):
    cp = pytest.importorskip("cupy")
    csp = pytest.importorskip("cupyx.scipy.sparse")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, b = get_sparse_variables
    x_np = DROPHyperplane(A, b).project(np.array([2.0, 2.0]))

    alg_cp = DROPHyperplane(csp.csr_matrix(A), cp.asarray(b))
    assert isinstance(alg_cp.inv_nnz_counts, cp.ndarray)  # stored on device
    x_cp = alg_cp.project(cp.asarray([2.0, 2.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
