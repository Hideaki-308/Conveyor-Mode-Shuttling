import numpy as np
from numba import jit
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
    scale_x = scale[0] / dx; scale_y = scale[1] / dy

    noise = np.random.normal(0, sigma, potential.shape)
    smoothed_noise = gaussian_filter(noise, sigma=(scale_x, scale_y), mode='wrap')
    scale_factor = sigma / np.std(smoothed_noise)
    smoothed_noise *= scale_factor

    return smoothed_noise