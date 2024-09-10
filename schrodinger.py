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
    # integral_xy = np.sum(product) * dx * dy

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
    F = np.zeros((2, num_evals))

    # initial wavefunction
    _, psi_r = get_instantaneous_eigenfunction(potential, 0, xrange[0], yrange[0])

    # define the function of split operator method
    @jit(nopython=True, cache=True)
    def split_operator(exp_r, eval_step, psi_r, exp_kk, pulse_reso):
        for i_step in range(eval_step):
            # time evolution (sprit operator method)
            i_pulse = i_step // pulse_reso
            psi_r = exp_r[i_pulse] * psi_r
            psi_p = fft.fft2(psi_r)
            psi_p = exp_kk * psi_p
            psi_r = fft.ifft2(psi_p)
            psi_r = exp_r[i_pulse] * psi_r

        return psi_r


    pot = np.zeros((eval_step, Ny, Nx))
    for i_eval in tqdm(range(0, num_evals), total=num_evals):
        ### Aquire the potential for the current evaluation step
        pls_slice = pls[i_eval*eval_step//pulse_reso: (i_eval+1)*eval_step//pulse_reso+1]
        pot = Potential._get_potential(pls_slice, potential.gpot)

        exp_r = np.exp(-1j*pot*dt/hbar/2)
        
        # shuttling
        psi_r = split_operator(exp_r, eval_step, psi_r, exp_kk, pulse_reso)
        
        # evaluation
        t = tlist[i_eval*eval_step + eval_step - 1]
        _, psi_r_inst = get_instantaneous_eigenfunction(potential, t, xr[i_eval], yr[i_eval])
        F[0][i_eval] = t
        F[1][i_eval] = np.abs(inner_prod(psi_r, psi_r_inst, dx, dy))**2  # fidelity
    
    return F

# %%
if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    import pathlib
    import pulse_potentials as pp
    import schrodinger as sch
    from constants import Constants
    from pulse_potentials import Potential

    script_dir = pathlib.Path().resolve()
    data_dir = script_dir / 'output' / 'pot'

    consts = Constants('Si/SiGe')
    gate_names = ['C1', 'C2', 'B1', 'B2', 'B3', 'B4', 'B5', 'P1', 'P2', 'P3', 'P4']
    ppot = Potential(data_dir, gate_names, consts)

    T = 1 / 1000e6  # period of the pulse (s)
    pulse_length = 1.5 * T

    A = 0.05  # amplitude of the pulse (V)
    B = 0.73  # offset of the pulse (V)
    dB = 0.07  # offset of the pulse for different layers (V)
    C1 = -0.0  # lateral gate voltage (V)
    C2 = -0.0  # lateral gate voltage (V)

    def V1(t): return A * np.cos(-2*np.pi*t/T + 0  *np.pi) + B
    def V2(t): return A * np.cos(-2*np.pi*t/T + 0.5*np.pi) + B + dB
    def V3(t): return A * np.cos(-2*np.pi*t/T + 1  *np.pi) + B
    def V4(t): return A * np.cos(-2*np.pi*t/T + 1.5*np.pi) + B + dB

    V_list = [V1, V2, V3, V4]

    voltages = {'C1': C1,
                'C2': C2,
                'B1': V4,
                'P1': V1,
                'B2': V2,
                'P2': V3,
                'B3': V4,
                'P3': V1,
                'B4': V2,
                'P4': V3,
                'B5': V4}
    
    ppot.make_pulse(pulse_length=pulse_length, pulse_shape=voltages, pulse_name='pulse1')


    xi = -210e-9
    yi = 0
    xf = 210e-9
    yf = 0
    x_width = 100e-9
    y_width = 100e-9

    xrange = ((xi - 0.5*x_width, xi + 0.5*x_width),
            (xf - 0.5*x_width, xf + 0.5*x_width))

    yrange = ((yi - 0.5*y_width, yi + 0.5*y_width),
            (yf - 0.5*y_width, yf + 0.5*y_width))
    
    dt = 1e-15
    num_evals = 100
    pulse_resolution = 1e-11

    pulse_reso = round(pulse_resolution / dt)
    F = sch.TE_solver(ppot, xrange, yrange, dt=dt, pulse_reso=pulse_reso, num_evals=num_evals)

    plt.plot(F[0], F[1])
    plt.xlabel('Time (s)')
    plt.ylabel('Fidelity')
    plt.show()
