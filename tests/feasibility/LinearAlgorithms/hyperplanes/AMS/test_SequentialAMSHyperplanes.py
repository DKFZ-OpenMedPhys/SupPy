import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SequentialAMSHyperplane
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
def get_SequentialAMSHyperplane_input_full(get_full_variables):
    A, b = get_full_variables
    return SequentialAMSHyperplane(A, b), A, b


@pytest.fixture
def get_SequentialAMSHyperplane_input_sparse(get_sparse_variables):
    A, b = get_sparse_variables
    return SequentialAMSHyperplane(A, b), A, b


def test_SequentialAMSHyperplane_no_relaxation_constructor_full(
    get_SequentialAMSHyperplane_input_full,
):
    """
    Test the SequentialAMSHyperplane constructor with no relaxation and a
    full
    matrix.
    """
    alg, A, b = get_SequentialAMSHyperplane_input_full

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.algorithmic_relaxation == 1.0
    assert alg.relaxation == 1.0


def test_SequentialAMSHyperplane_no_relaxation_constructor_sparse(
    get_SequentialAMSHyperplane_input_sparse,
):
    """
    Test the SequentialAMSHyperplane constructor with no relaxation and a
    sparse
    matrix.
    """
    alg, A, b = get_SequentialAMSHyperplane_input_sparse

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.cs, np.arange(A.shape[0]))
    assert alg.algorithmic_relaxation == 1.0
    assert alg.relaxation == 1.0


def test_SequentialAMSHyperplane_constructor_wrong_bounds_shape(get_full_variables):
    """Test the SequentialAMSHyperslab constructor with wrong bounds shape."""
    A, _ = get_full_variables
    ub = np.array([1, 2])
    with pytest.raises(ValueError):
        SequentialAMSHyperplane(A, ub)


def test_SequentialAMSHyperplane_constructor_scalar_bounds(get_full_variables):
    """Test the SequentialAMSHyperslab constructor with scalar bounds."""
    A, _ = get_full_variables
    b = 1
    alg = SequentialAMSHyperplane(A, b)
    print(alg.b)
    assert (alg.b == np.array([1, 1, 1])).all()


def test_SequentialAMSHyperplane_map_full(get_SequentialAMSHyperplane_input_full):
    """Test the map function of the SequentialAMSHyperplane class with full
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHyperplane_input_full

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([1, 0, 1]))


def test_SequentialAMSHyperplane_map_sparse(get_SequentialAMSHyperplane_input_sparse):
    """Test the map function of the SequentialAMSHyperplane class with sparse
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHyperplane_input_sparse

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([1, 0, 1]))


def test_SequentialAMSHyperplane_indexed_map_full(get_SequentialAMSHyperplane_input_full):
    """
    Test the indexed_map function of the SequentialAMSHyperplane class with
    full
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHyperplane_input_full
    idx = [0, 2]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([1, 1]))


def test_SequentialAMSHyperplane_indexed_map_sparse(get_SequentialAMSHyperplane_input_sparse):
    """
    Test the indexed_map function of the SequentialAMSHyperplane class with
    sparse
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHyperplane_input_sparse
    idx = [0, 2]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([1, 1]))


def test_SequentialAMSHyperplane_step_full(get_SequentialAMSHyperplane_input_full):
    """Test the step function of the SequentialAMSHyperplane class with full
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHyperplane_input_full

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


def test_SequentialAMSHyperplane_step_sparse(get_SequentialAMSHyperplane_input_sparse):
    """Test the step function of the SequentialAMSHyperplane class with
    sparse.
    """
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHyperplane_input_sparse

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


def test_SequentialAMSHyperplane_step_full_algoritimic_relaxation(get_full_variables):
    """Test the step function of the SequentialAMSHyperplane class with
    relaxation.
    """
    A, b = get_full_variables
    # test with relaxation
    alg = SequentialAMSHyperplane(A, b, algorithmic_relaxation=1.5)
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.625, 1.0625])) < 1e-10)
    assert np.array_equal(x_n, x_1)


def test_SequentialAMSHyperplane_step_sparse_algoritimic_relaxation(get_sparse_variables):
    """Test the step function of the SequentialAMSHyperplane class with
    relaxation.
    """
    A, b = get_sparse_variables
    # test with relaxation
    alg = SequentialAMSHyperplane(A, b, algorithmic_relaxation=1.5)
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.625, 1.0625])) < 1e-10)
    assert np.array_equal(x_n, x_1)


def test_SequentialAMSHyperplane_step_full_relaxation(get_full_variables):
    """Test the step function of the SequentialAMSHyperplane class with
    relaxation.
    """
    A, b = get_full_variables
    # test with relaxation
    alg = SequentialAMSHyperplane(A, b, relaxation=1.5)
    assert alg.relaxation == 1.5
    assert alg.algorithmic_relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.25, 0.5])) < 1e-10)


def test_SequentialAMSHyperplane_step_sparse_relaxation(get_sparse_variables):
    """Test the step function of the SequentialAMSHyperplane class with
    relaxation.
    """
    A, b = get_sparse_variables
    # test with relaxation
    alg = SequentialAMSHyperplane(A, b, relaxation=1.5)
    assert alg.relaxation == 1.5
    assert alg.algorithmic_relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.25, 0.5])) < 1e-10)


def test_SequentialAMSHyperplane_custom_cs(get_SequentialAMSHyperplane_input_full):
    """Test the step function of the SequentialAMSHyperplane class with
    custom.
    """
    _, A, b = get_SequentialAMSHyperplane_input_full

    # test with custom cs
    alg = SequentialAMSHyperplane(A, b, cs=[2, 1, 0])

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1, 1.5])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([1, 2]) < 1e-10))
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([1, 2]) < 1e-10))
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([1, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([1, 0.5]) < 1e-10))
    assert np.array_equal(x_n, x_5)


def test_SequentialAMSHyperplane_proximity(get_SequentialAMSHyperplane_input_full):
    """Test the proximity function of the SequentialAMSHyperplane class."""
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHyperplane_input_full

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    prox_measures = []
    no_prox = alg.proximity(x_1, prox_measures)
    assert no_prox.size == 0

    prox_measures = [("p_norm", 2), "max_norm"]
    prox_1 = alg.proximity(x_1, prox_measures)
    prox_2 = alg.proximity(x_2, prox_measures)
    prox_3 = alg.proximity(x_3, prox_measures)
    prox_4 = alg.proximity(x_4, prox_measures)
    prox_5 = alg.proximity(x_5, prox_measures)

    assert np.abs(prox_1[0] - 2 / 3) < 1e-10
    assert np.abs(prox_1[1] - 1) < 1e-10

    assert np.abs(prox_2[0] - 5 / 3) < 1e-10
    assert np.abs(prox_2[1] - 2) < 1e-10

    assert np.abs(prox_3[0] - 29 / 12) < 1e-10
    assert np.abs(prox_3[1] - 2) < 1e-10

    assert np.abs(prox_4[0] - 14 / 3) < 1e-10
    assert np.abs(prox_4[1] - 3) < 1e-10

    assert np.abs(prox_5[0] - 77 / 12) < 1e-10
    assert np.abs(prox_5[1] - 4) < 1e-10
