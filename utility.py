import numpy as np
from numba import jit, prange
from scipy.ndimage import gaussian_filter


def inner_prod(psi1, psi2, dx, dy):
    """
    Calculate the inner product of two wavefunctions.

    Parameters
    ----------
    psi1 : np.ndarray
        Wavefunction 1.
    psi2 : np.ndarray
        Wavefunction 2.
    dx : float
        x grid spacing.
    dy : float
        y grid spacing.

    Returns
    -------
    inner_prod : float
        Inner product.
    """
    if psi1.shape != psi2.shape:
        raise ValueError("Wavefunctions must have the same shape.")
    if np.any(np.isnan(psi1)):
        raise ValueError("psi1 contains NaN values.")
    if np.any(np.isnan(psi2)):
        raise ValueError("psi2 contains NaN values.")

    product = psi1.conj() * psi2
    integral_x = np.trapz(product, dx=dx, axis=1)
    integral_xy = np.trapz(integral_x, dx=dy)
    # integral_xy = np.sum(product) * dx * dy

    return integral_xy


    

def generate_fructuation(potential, sigma, scale, seed=None):
    """
    Generate fluctuation for given potential grid shape.

    Parameters
    ----------
    potential : Potential
        Potential object. The potential grid shape is used to generate the fluctuation.
    sigma : float
        Standard deviation of the Gaussian distribution.
    scale : tuple
        The scale of the fluctuation in unit of m.
    """
    np.random.seed(seed)

    dx = potential.x[0][1] - potential.x[0][0]
    dy = potential.y[1][0] - potential.y[0][0]
    scale_x = scale / dx; scale_y = scale / dy

    noise = np.random.normal(0, sigma, potential.x.shape)
    smoothed_noise = gaussian_filter(noise, sigma=(scale_x, scale_y), mode='wrap')
    scale_factor = sigma / np.std(smoothed_noise)
    smoothed_noise *= scale_factor

    return smoothed_noise

@jit(nopython=True, parallel=True, cache=True)
def partial_matrix_prod(A, B):
    """
    Calculate the product of an operator and a wavefunction, partially element-wise and partially matrix multiplication.

    Parameters
    ----------
    A : np.ndarray
        Array of matrices.
    B : np.ndarray
        Array of matrices.

    Returns
    -------
    prod : np.ndarray
        The result of partially matrix product.
    """
    s_A = A.shape; s_B = B.shape

    if s_A[:-2] != s_B[:-2]:
        raise ValueError("The arrays must have the same shape except for the last two dimensions.")
    if s_A[-1] != s_B[-2]:
        raise ValueError("The last two dimensions must match to perform matrix multiplication.")
    if len(s_A) != 4 or len(s_B) != 4:
        raise ValueError("4 dimension arrays are only supported.")

    product_shape = (*s_A[:-2], s_A[-2], s_B[-1])
    product = np.zeros(product_shape, dtype=np.complex128)
    for i in prange(s_A[0]):
        for j in prange(s_A[1]):
            product[i, j] = A[i, j] @ B[i, j]

    return product

            