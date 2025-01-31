import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SequentialWeightedAMSHyperplane
from suppy.utils import LinearMapping


@pytest.fixture
def get_full_variables():
    A = np.array([[1, 0], [1 / 2, -1 / 2], [0, 1]])
    b = np.array([1, 0, 1])
    return A, b


@pytest.fixture
def get_sparse_variables():
    A = np.array([[1, 0], [1 / 2, -1 / 2], [0, 1]])
    A = sparse.csr_matrix(A)
    b = np.array([1, 0, 1])
    return A, b


@pytest.fixture
def get_SequentialWeightedAMSHyperplane_input(get_full_variables):
    A, b = get_full_variables
    return SequentialWeightedAMSHyperplane(A, b, algorithmic_relaxation=1.5), A, b


@pytest.fixture
def get_SequentialWeightedAMSHyperplane_input_sparse(get_sparse_variables):
    A, b = get_sparse_variables
    return SequentialWeightedAMSHyperplane(A, b, algorithmic_relaxation=1.5), A, b


def test_SequentialWeightedAMSHyperplane_constructor_no_weights_full(
    get_SequentialWeightedAMSHyperplane_input,
):
    """
    Test the SequentialWeightedAMSHyperplane constructor with no weights and
    full
    matrix.
    """
    alg, A, b = get_SequentialWeightedAMSHyperplane_input

    alg = SequentialWeightedAMSHyperplane(A, b, algorithmic_relaxation=1.5)
    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0
    assert np.all(alg.weights == np.ones(len(A)))


def test_SequentialWeightedAMSHyperplane_constructor_no_weights_sparse(
    get_SequentialWeightedAMSHyperplane_input_sparse,
):
    """
    Test the SequentialWeightedAMSHyperplane constructor with no weights and
    sparse
    matrix.
    """
    alg, A, b = get_SequentialWeightedAMSHyperplane_input_sparse

    alg = SequentialWeightedAMSHyperplane(A, b, algorithmic_relaxation=1.5)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.cs, np.arange(A.shape[0]))
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0
    assert np.all(alg.weights == np.ones(A.shape[0]))


def test_SequentialWeightedAMSHyperplane_constructor_custom_weights(get_full_variables):
    """Test the SequentialWeightedAMSHyperplane constructor with custom
    weights.
    """
    A, b = get_full_variables
    alg = SequentialWeightedAMSHyperplane(
        A, b, algorithmic_relaxation=1.5, weights=np.ones(len(A))
    )

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0
    assert np.all(alg.weights == np.ones(len(A)))
    assert alg.temp_weight_decay == 1.0

    # does weight decay work?


def test_SequentialWeightedAMSHyperplane_weight_decay_full(get_full_variables):
    """Test the weight decay of the SequentialWeightedAMSHyperplane class."""
    A, b = get_full_variables
    alg = SequentialWeightedAMSHyperplane(A, b, weights=np.ones(len(A)), weight_decay=0.5)

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)
    assert alg.temp_weight_decay == 0.5


def test_SequentialWeightedAMSHyperplane_weight_decay_sparse(get_sparse_variables):
    """Test the weight decay of the SequentialWeightedAMSHyperplane class."""
    A, b = get_sparse_variables
    alg = SequentialWeightedAMSHyperplane(A, b, weights=np.ones(A.shape[0]), weight_decay=0.5)

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)
    assert alg.temp_weight_decay == 0.5


def test_SequentialWeightedAMSHyperplane_weight_decay_step_full(get_full_variables):
    """Test the weight decay of the SequentialWeightedAMSHyperplane class."""
    A, b = get_full_variables
    alg = SequentialWeightedAMSHyperplane(
        A, b, algorithmic_relaxation=1.5, weights=np.ones(len(A))
    )
    alg.temp_weight_decay = 2 / 3
    assert alg.temp_weight_decay == 2 / 3

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([2, 1]) < 1e-10))
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([2, 1]) < 1e-10))
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([1.5, 1]) < 1e-10))
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([0.5, 1]) < 1e-10))
    assert np.array_equal(x_n, x_5)


def test_SequentialWeightedAMSHyperplane_weight_decay_step_sparse(get_sparse_variables):
    """Test the weight decay of the SequentialWeightedAMSHyperplane class."""
    A, b = get_sparse_variables
    alg = SequentialWeightedAMSHyperplane(
        A, b, algorithmic_relaxation=1.5, weights=np.ones(A.shape[0])
    )
    alg.temp_weight_decay = 2 / 3
    assert alg.temp_weight_decay == 2 / 3

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([2, 1]) < 1e-10))
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([2, 1]) < 1e-10))
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([1.5, 1]) < 1e-10))
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([0.5, 1]) < 1e-10))
    assert np.array_equal(x_n, x_5)
