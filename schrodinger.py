import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import eigsh
from numba import jit
from tqdm import tqdm
from pulse_potentials import Potential


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

    return integral_xy

def get_instantaneous_eigenfunction(potential, t, xrange=None, yrange=None):
    """
    Get the instantaneous electron eigenfunction for a given time.

    Parameters
    ----------
    potential : Potential
        Potential object.
    t : float
        Time.
    xrange : tuple, optional
        The range of x coordinate where the electron should exist.
        If None, the whole x range is considered. The default is None.
    yrange : tuple, optional
        The range of y coordinate where the electron should exist.
        If None, the whole y range is considered. The default is None.

    Returns
    -------
    eigenfunction : np.ndarray
        Eigenfunction.
    """
    # get constants
    hbar = potential.consts.hbar
    me = potential.consts.me


    # get the grid parameters
    X = potential.x; Y = potential.y
    x = X[0,:]; y = Y[:,0]
    Nx = len(x); Ny = len(y)
    dx = x[1] - x[0]; dy = y[1] - y[0]

    # get the potential at time t and convert to hamiltonian
    pot = potential.get_potential(t) * potential.consts.e  # convert to J

    # truncate the potential and grid if xrange and yrange are specified
    if xrange is not None:
        xidx = np.where((x >= xrange[0]) & (x <= xrange[1]))[0]
        x = x[xidx]
        pot = pot[:, xidx]
        nx = len(x)
    else:
        xidx = np.arange(Nx)
        nx = Nx
    if yrange is not None:
        yidx = np.where((y >= yrange[0]) & (y <= yrange[1]))[0]
        y = y[yidx]
        pot = pot[yidx, :]
        ny = len(y)
    else:
        yidx = np.arange(Ny)
        ny = Ny
    
    # build the potential energy hamiltonian term
    pot_flat = pot.T.reshape((nx*ny,), order='F')
    PE_2D = sparse.diags(pot_flat)

    # build the kinetic energy hamiltonian term
    KE_X = sparse.diags([-2/(dx**2), 1/(dx**2), 1/(dx**2)], [0, -1, 1], shape=(nx, nx))
    KE_Y = sparse.diags([-2/(dy**2), 1/(dy**2), 1/(dy**2)], [0, -1, 1], shape=(ny, ny))
    KE_2D = sparse.kron(sparse.eye(ny), KE_X) + sparse.kron(KE_Y, sparse.eye(nx))

    KE_2D = -hbar**2/(2*me)*KE_2D

    H = PE_2D + KE_2D

    # get the eigenvalues and eigenvectors
    eigvals, eigvecs = eigsh(H.tocsc(), k=1, sigma=pot.min())

    # reshape the eigenfunction
    eigenfunction = eigvecs[:,0].reshape((ny, nx))

    # convert to original grid
    if xrange is not None or yrange is not None:
        eigenfunction = np.pad(eigenfunction, ((yidx[0], Ny-yidx[-1]-1), (xidx[0], Nx-xidx[-1]-1)))

    # normalize the eigenfunction
    eigenfunction = eigenfunction / np.sqrt(inner_prod(eigenfunction, eigenfunction, dx, dy))
    eigenfunction = eigenfunction.astype(np.complex128)

    return eigvals, eigenfunction

def TE_solver(potential, xrange, yrange, dt=0.1e-9, pulse_reso=10, num_evals=50):
    """
    Solve the time-dependent Schrodinger equation using the Split-Operator method.

    Parameters
    ----------
    potential : Potential
        Potential object.
            xrange : list of tuple
        The range of x coordinate where the electron initially and finally should exist.
    yrange : list of tuple
        The range of y coordinate where the electron initially and finally should exist.
    dt : float
        Time step.
    pulse_reso : int, optional
        The time resolution of the pulse. The pulse keeps its voltage for dt*pulse_reso.
        The default is 10.
    num_evals : int, optional
        How many times to evaluate the shuttling. The default is 50.

    Returns
    -------
    None.
    """

    # get constants
    ct = potential.consts
    hbar = ct.hbar

    # get the grid parameters
    X = potential.x; Y = potential.y
    x = X[0,:]; y = Y[:,0]
    Nx = len(x); Ny = len(y)
    dx = x[1] - x[0]; dy = y[1] - y[0]
    
    # define momentum coordinates
    dpx = 2*np.pi*hbar/(Nx*dx); dpy = 2*np.pi*hbar/(Ny*dy)
    px = np.arange(-Nx/2, Nx/2) * dpx
    py = np.arange(-Ny/2, Ny/2) * dpy
    PX, PY = np.meshgrid(px, py)

    # time evolution operator in momentum space
    exp_k = np.exp(-1j*(PX**2 + PY**2) /(2*ct.me*hbar) *dt/2)
    exp_kk = exp_k * exp_k
    exp_k = fft.ifftshift(exp_k)
    exp_kk = fft.ifftshift(exp_kk) 

    # time variables
    tlist = np.arange(0, potential.pulse['length'], dt)
    # eval_steps = np.linspace(0, len(tlist)-1, num_evals, dtype=int)
    eval_step = len(tlist) // num_evals
    xr = np.linspace(xrange[0], xrange[1], num_evals)
    yr = np.linspace(yrange[0], yrange[1], num_evals)

    # get the voltages of the pulse
    pls = potential.pulse.resolve(dt*pulse_reso)

    # initialize the fidelity array
    F = np.zeros((num_evals, 2))

    # initial wavefunction
    _, psi_r = get_instantaneous_eigenfunction(potential, 0, xrange[0], yrange[0])

    # define the function of split operator method
    @jit(nopython=True, cache=True)
    def split_operator(exp_r, eval_step, psi_r, exp_kk, pot, dt, hbar):
        for i_step in range(eval_step):
            # time evolution (sprit operator method)

            exp_r = np.exp(-1j*pot*dt/hbar/2)
            psi_r = exp_r * psi_r
            psi_p = fft.fft2(psi_r)
            psi_p = exp_kk * psi_p
            psi_r = fft.ifft2(psi_p)
            psi_r = exp_r * psi_r

        return psi_r


    pot = np.zeros((eval_step, Ny, Nx))
    for i_eval in tqdm(range(0, num_evals), total=num_evals):
        # get potential in the range
        pls_slice = pls[i_eval*eval_step: (i_eval+1)*eval_step]
        print(pls_slice.shape)
        pot = Potential._get_potential(pls_slice, potential.gpot)
        print(pot.shape)
        exp_r = np.exp(-1j*pot*dt/hbar/2)
        print(exp_r.shape)
        if np.any(np.isnan(exp_r[0])):
            raise ValueError(f"exp_r contains NaN at step {i_eval}")
        # shuttling
        psi_r = split_operator(exp_r, eval_step, psi_r, exp_kk, pulse_reso)
        # evaluation
        t = tlist[i_eval*eval_step + eval_step - 1]
        _, psi_r_inst = get_instantaneous_eigenfunction(potential, t, xr[i_eval], yr[i_eval])
        F[i_eval][0] = t
        F[i_eval][1] = np.abs(inner_prod(psi_r, psi_r_inst, dx, dy))**2
        print(F[i_eval][1])
        raise ValueError("stop")
    
    return F