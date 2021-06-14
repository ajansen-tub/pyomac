# %%
from scipy.signal import lti, lsim
from scipy.linalg import eigh
import numpy as np
from typing import Tuple


def get_model() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate system matrices for mdof-system

    :return: System matrices, natural frequencies [Hz] and mode shapes for mdof-system
    """

    # %% base matrices
    m_diag = np.array([1000, 1001, 1002, 1003, 1004])  # masses in kg
    m = np.diag(m_diag)
    k0 = 100 * np.array([8000, 7000, 7000, 7000, 7000])
    n_dof = 5

    # %% stiffness matrix
    k_raw = np.zeros((6, 6))
    elem_matrix = np.array([[1, -1], [-1, 1]])

    for idx_element in range(n_dof):
        k_elem = k0[idx_element] * elem_matrix
        k_raw[idx_element: idx_element + 2, idx_element: idx_element + 2] += k_elem

    k = k_raw[:-1, :-1]

    # %% eigenfrequencies
    eigvals, ms = eigh(k, m)
    omega = np.sqrt(eigvals)
    f = omega / 2 / np.pi

    # %% damping
    xi = 0.005

    beta = 2 * xi / (omega[0] + omega[1])
    alpha = omega[0] * omega[1] * beta

    zeta = alpha / 2 / omega + beta / 2 * omega
    c_damp = alpha * m + beta * k

    return m, k, c_damp, f, ms, zeta


def solver(m: np.ndarray, k: np.ndarray, c_damp: np.ndarray,
           force: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    """
    Solve state-space model subjected to external force

    :param m: mass matrix (n x n)
    :param k: stiffness matrix (n x n)
    :param c_damp: damping matrix (n x n)
    :param force: force vector (m x n)
    :param t: time vector (m x 0)

    :return: Acceleration response of mdof-system to external force
    """

    # dim of state-space system
    n = m.shape[0]

    # allocate system matrices
    a = np.zeros((2 * n, 2 * n))
    b = np.zeros((2 * n, n))
    c_sys = np.zeros((n, 2 * n))
    d = np.zeros((n, n))

    # only minor time difference compared to pseudo-inverse
    m_inv = np.linalg.inv(m)

    # fill system matrices (@ better than np.dot)
    a[:n, n:] = np.identity(n)
    a[n:, :n] = -m_inv @ k
    a[n:, n:] = -m_inv @ c_damp

    b[n:, :] = m_inv

    c_sys[:n, :n] = -m_inv @ k
    c_sys[:n, n:] = -m_inv @ c_damp

    d = m_inv

    # lti-system
    system = lti(a, b, c_sys, d)

    # solver
    t, acc, _ = lsim(system, force, t)

    return t, acc
