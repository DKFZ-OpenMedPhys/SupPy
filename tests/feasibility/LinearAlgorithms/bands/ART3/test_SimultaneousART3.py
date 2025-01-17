# import numpy as np
# import pytest
# import scipy.sparse as sparse
# from suppy.feasibility import SimultaneousART3plus
# from suppy.utils import LinearMapping


# @pytest.fixture
# def get_ART3_variables_full():
#     A = np.array([[1, 0], [0, 1]])
#     lb = np.array([-1, -1])
#     ub = np.array([1, 1])
#     return A, lb, ub


# @pytest.fixture
# def get_ART3_variables_sparse():
#     A = sparse.csr_matrix([[1, 0], [0, 1]])
#     lb = np.array([-1, -1])
#     ub = np.array([1, 1])
#     return A, lb, ub


# def test_SimultaneousART3plus_constructor_full(get_ART3_variables_full):
#     """Test the SimultaneousART3plus constructor."""
#     A, lb, ub = get_ART3_variables_full
#     alg = SimultaneousART3plus(A, lb, ub, weights=np.array([1 / 3, 2 / 3]))
#     assert np.array_equal(alg.weights, np.array([1 / 3, 2 / 3]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     assert isinstance(alg.A, LinearMapping)
#     assert np.array_equal(alg.A, A)
#     assert np.array_equal(alg.Bounds.l, lb)
#     assert np.array_equal(alg.Bounds.u, ub)
#     assert np.array_equal(alg.weights, [1 / 2, 1 / 2])
#     assert alg._feasible == True


# def test_SimultaneousART3plus_constructor_sparse(get_ART3_variables_sparse):
#     """Test the SimultaneousART3plus constructor."""
#     A, lb, ub = get_ART3_variables_sparse
#     alg = SimultaneousART3plus(A, lb, ub)

#     assert isinstance(alg.A, LinearMapping)
#     assert np.array_equal(alg.A.todense(), A.todense())
#     assert np.array_equal(alg.Bounds.l, lb)
#     assert np.array_equal(alg.Bounds.u, ub)
#     assert np.array_equal(alg.weights, [1 / 2, 1 / 2])
#     assert alg._feasible == True


# def test_SimultaneousART3plus_step_full(get_ART3_variables_full):
#     """Test the step function of the SimultaneousART3plus class."""
#     A, lb, ub = get_ART3_variables_full
#     alg = SimultaneousART3plus(A, lb, ub)

#     x_1 = np.array([-3.0, 0.0])
#     x_2 = np.array([-1.5, 0.0])
#     x_3 = np.array([0.5, 0.5])
#     x_4 = np.array([1.5, 0.0])
#     x_5 = np.array([3.0, 0.0])
#     x_6 = np.array([-3.0, -3.0])
#     x_7 = np.array([-3.0, 3.0])
#     x_8 = np.array([-3, 1.5])

#     assert np.all(alg._not_met == np.array([0, 1]))
#     x_n = alg.step(x_1)
#     assert np.array_equal(x_1, x_n)
#     assert np.array_equal(x_n, np.array([-1.5, 0]))
#     assert np.all(alg._not_met == np.array([0]))
#     # next step is a reflection
#     x_n = alg.step(x_1)
#     assert np.array_equal(x_1, x_n)
#     assert np.array_equal(x_n, np.array([-1, 0]))
#     assert np.all(alg._not_met == np.array([0]))
#     # next step should remove the last constraint
#     x_n = alg.step(x_1)
#     assert np.array_equal(x_1, x_n)
#     assert np.array_equal(x_n, np.array([-1, 0]))
#     assert np.all(alg._not_met == np.array([]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_2)
#     assert np.array_equal(x_2, x_n)
#     assert np.array_equal(x_n, np.array([-1, 0]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_3)
#     assert np.array_equal(x_3, x_n)
#     assert np.array_equal(x_n, np.array([0.5, 0.5]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_4)
#     assert np.array_equal(x_4, x_n)
#     assert np.array_equal(x_n, np.array([1, 0]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_5)
#     assert np.array_equal(x_5, x_n)
#     assert np.array_equal(x_n, np.array([1.5, 0]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_6)
#     assert np.array_equal(x_6, x_n)
#     assert np.array_equal(x_n, np.array([-1.5, -1.5]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_7)
#     assert np.array_equal(x_7, x_n)
#     assert np.array_equal(x_n, np.array([-1.5, 1.5]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_8)
#     assert np.array_equal(x_8, x_n)
#     assert np.array_equal(x_n, np.array([-1.5, 1]))


# def test_SimultaneousART3plus_step_sparse(get_ART3_variables_sparse):
#     """
#     Test the step function of the SimultaneousART3plus class for sparse
#     input.
#     """
#     A, lb, ub = get_ART3_variables_sparse
#     alg = SimultaneousART3plus(A, lb, ub)

#     x_1 = np.array([-3.0, 0.0])
#     x_2 = np.array([-1.5, 0.0])
#     x_3 = np.array([0.5, 0.5])
#     x_4 = np.array([1.5, 0.0])
#     x_5 = np.array([3.0, 0.0])
#     x_6 = np.array([-3.0, -3.0])
#     x_7 = np.array([-3.0, 3.0])
#     x_8 = np.array([-3, 1.5])

#     x_n = alg.step(x_1)
#     assert np.array_equal(x_1, x_n)
#     assert np.array_equal(x_n, np.array([-1.5, 0]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_2)
#     assert np.array_equal(x_2, x_n)
#     assert np.array_equal(x_n, np.array([-1, 0]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_3)
#     assert np.array_equal(x_3, x_n)
#     assert np.array_equal(x_n, np.array([0.5, 0.5]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_4)
#     assert np.array_equal(x_4, x_n)
#     assert np.array_equal(x_n, np.array([1, 0]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_5)
#     assert np.array_equal(x_5, x_n)
#     assert np.array_equal(x_n, np.array([1.5, 0]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_6)
#     assert np.array_equal(x_6, x_n)
#     assert np.array_equal(x_n, np.array([-1.5, -1.5]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_7)
#     assert np.array_equal(x_7, x_n)
#     assert np.array_equal(x_n, np.array([-1.5, 1.5]))

#     alg = SimultaneousART3plus(A, lb, ub)
#     x_n = alg.step(x_8)
#     assert np.array_equal(x_8, x_n)
#     assert np.array_equal(x_n, np.array([-1.5, 1]))
