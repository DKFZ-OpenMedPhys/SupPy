import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SequentialWeightedAMSHyperslab
from suppy.utils import LinearMapping


@pytest.fixture
def get_full_variables():
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    lb = np.array([-2, -2, -3 / 2, -3 / 2])
    ub = -1 * lb
    return A, lb, ub


@pytest.fixture
def get_sparse_variables():
    A = sparse.csr_matrix([[1, 1], [-1, 1], [1, 0], [0, 1]])
    lb = np.array([-2, -2, -3 / 2, -3 / 2])
    ub = -1 * lb
    return A, lb, ub


@pytest.fixture
def get_SequentialWeightedAMSHyperslab_input(get_full_variables):
    A, lb, ub = get_full_variables
    return SequentialWeightedAMSHyperslab(A, lb, ub, algorithmic_relaxation=1.5), A, lb, ub


@pytest.fixture
def get_SequentialWeightedAMSHyperslab_input_sparse(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    return SequentialWeightedAMSHyperslab(A, lb, ub, algorithmic_relaxation=1.5), A, lb, ub


def test_SequentialWeightedAMSHyperslab_constructor_no_weights_full(
    get_SequentialWeightedAMSHyperslab_input,
):
    """
    Test the SequentialWeightedAMSHyperslab constructor with no weights and
    full
    matrix.
    """
    alg, A, lb, ub = get_SequentialWeightedAMSHyperslab_input

    alg = SequentialWeightedAMSHyperslab(A, lb, ub, algorithmic_relaxation=1.5)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0
    assert np.all(alg.weights == np.ones(len(A)))


def test_SequentialWeightedAMSHyperslab_constructor_no_weights_sparse(
    get_SequentialWeightedAMSHyperslab_input_sparse,
):
    """
    Test the SequentialWeightedAMSHyperslab constructor with no weights and
    sparse
    matrix.
    """
    alg, A, lb, ub = get_SequentialWeightedAMSHyperslab_input_sparse

    alg = SequentialWeightedAMSHyperslab(A, lb, ub, algorithmic_relaxation=1.5)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(A.shape[0]))
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0
    assert np.all(alg.weights == np.ones(A.shape[0]))


def test_SequentialWeightedAMSHyperslab_constructor_custom_weights(get_full_variables):
    """Test the SequentialWeightedAMSHyperslab constructor with custom weights."""
    A, lb, ub = get_full_variables
    alg = SequentialWeightedAMSHyperslab(
        A, lb, ub, algorithmic_relaxation=1.5, weights=np.ones(len(A))
    )

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0
    assert np.all(alg.weights == np.ones(len(A)))
    assert alg.temp_weight_decay == 1.0

    # does weight decay work?


def test_SequentialWeightedAMSHyperslab_weight_decay_full(get_full_variables):
    """Test the weight decay of the SequentialWeightedAMSHyperslab class."""
    A, lb, ub = get_full_variables
    alg = SequentialWeightedAMSHyperslab(A, lb, ub, weights=np.ones(len(A)), weight_decay=0.5)

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)
    assert alg.temp_weight_decay == 0.5


def test_SequentialWeightedAMSHyperslab_weight_decay_sparse(get_sparse_variables):
    """Test the weight decay of the SequentialWeightedAMSHyperslab class."""
    A, lb, ub = get_sparse_variables
    alg = SequentialWeightedAMSHyperslab(A, lb, ub, weights=np.ones(A.shape[0]), weight_decay=0.5)

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)
    assert alg.temp_weight_decay == 0.5


def test_SequentialWeightedAMSHyperslab_weight_decay_step_full(get_full_variables):
    """Test the weight decay of the SequentialWeightedAMSHyperslab class."""
    A, lb, ub = get_full_variables
    alg = SequentialWeightedAMSHyperslab(
        A, lb, ub, algorithmic_relaxation=1.5, weights=np.ones(len(A))
    )
    alg.temp_weight_decay = 2 / 3
    assert alg.temp_weight_decay == 2 / 3

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-1, 1]) < 1e-10))
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([-1.5, 0]) < 1e-10))
    assert np.array_equal(x_n, x_5)


def test_SequentialWeightedAMSHyperslab_weight_decay_step_sparse(get_sparse_variables):
    """Test the weight decay of the SequentialWeightedAMSHyperslab class."""
    A, lb, ub = get_sparse_variables
    alg = SequentialWeightedAMSHyperslab(
        A, lb, ub, algorithmic_relaxation=1.5, weights=np.ones(A.shape[0])
    )
    alg.temp_weight_decay = 2 / 3
    assert alg.temp_weight_decay == 2 / 3

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-1, 1]) < 1e-10))
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([-1.5, 0]) < 1e-10))
    assert np.array_equal(x_n, x_5)
