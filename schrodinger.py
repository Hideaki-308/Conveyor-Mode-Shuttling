import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eig
from numba import jit
from tqdm import tqdm
from pulse_potentials import Potential
from utility import inner_prod, generate_fructuation


@jit(nopython=True, cache=True)
def _get_potential(val, gpot):
    """
    Get the interpolated potential for a given set of voltages.
    Same as the _get_potential method in the Potential.
    This is for the numba jit compilation.

    Parameters
    ----------
    val : list
        List of gate voltages. The order of the list must match the order of gate names.
    """
    if val.ndim == 1:
        val_ext = val.reshape(-1, 1, 1)
        pot = np.sum(-val_ext * gpot, axis=0)
        return pot
    elif val.ndim == 2:
        val_ext = val[:, :, np.newaxis, np.newaxis]
        gpot_ext = gpot[np.newaxis, :, :, :]
        pot = np.sum(-val_ext * gpot_ext, axis=1)
        return pot

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
    pot = potential._get_potential(t)

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
    Solve the time-dependent Schrodinger equation with spin-orbit intaraction using the Split-Operator method.

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
    pls = potential.pulse.resolve(dt)

    # initialize the output arrays
    F = np.zeros((2, num_evals))  # fidelity
    result = np.zeros((num_evals, Ny, Nx), dtype=np.complex128)  # wavefunction

    # initial wavefunction
    _, psi_r = get_instantaneous_eigenfunction(potential, 0, xrange[0], yrange[0])

    # define the function of split operator method
    @jit(nopython=True, cache=True)
    def split_operator(eval_step, psi_r, exp_kk, gpot, pls_slice):
        for i_step in range(eval_step):
            # time evolution (sprit operator method)
            pot = _get_potential(pls_slice[i_step], gpot)
            exp_r = np.exp(-1j*pot*dt/hbar/2)
            psi_r = exp_r * psi_r
            psi_p = fft.fft2(psi_r)
            psi_p = exp_kk * psi_p
            psi_r = fft.ifft2(psi_p)
            psi_r = exp_r * psi_r

        return psi_r

    for i_eval in tqdm(range(0, num_evals), total=num_evals):
        ### Aquire the potential for the current evaluation step
        pls_slice = pls[i_eval*eval_step: (i_eval+1)*eval_step]
        
        # shuttling
        psi_r = split_operator(eval_step, psi_r, exp_kk, potential.gpot, pls_slice)
        
        # evaluation
        t = tlist[i_eval*eval_step + eval_step - 1]
        _, psi_r_inst = get_instantaneous_eigenfunction(potential, t, xr[i_eval], yr[i_eval])
        F[0][i_eval] = t
        F[1][i_eval] = np.abs(inner_prod(psi_r, psi_r_inst, dx, dy))**2  # fidelity
        result[i_eval] = psi_r
        if np.isnan(F[1][i_eval]):
            raise ValueError("Fidelity contains NaN values.")
    
    return F, result

def TE_solver_with_spin(potential, xrange, yrange, dt=0.1e-9, num_evals=50,
                        alpha=0., beta=0., theta=0., B=(0., 0., 0.),
                        g_sd=0., g_scale=0.):
    """
    Solve the time-dependent Schrodinger equation with spin-orbit interaction using the Split-Operator method.

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
    alpha : float, optional
        Rashba spin-orbit coupling constant. The default is 0.
    beta : float, optional
        Dresselhaus spin-orbit coupling constant. The default is 0.
    theta : float, optional
        The angle of shuttling direction from (100). Positive is counterclockwise.
        The default is 0.
    B : tuple, optional
        External magnetic field vector. The default is 0.
    g_factor_noise : float, optional
        Standard deviation of the g-factor noise. The default is 0.
    g_scale : float, optional
        The scale of fluctuation of the g-factor. The default is 0.

    Returns
    -------
    None.
    """

    # get constants
    ct = potential.consts
    hbar = ct.hbar
    g = ct.g_factor + generate_fructuation(potential, g_sd, g_scale, seed=42)

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
    PX = fft.ifftshift(PX)  # change the order to match the order of fft
    PY = fft.ifftshift(PY)


    @jit(nopython=True, cache=True)
    def positional_operator(pot, dt, hbar, B):
        # define the Pauli matrices
        si = np.array([[1, 0], [0, 1]])
        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.array([[1, 0], [0, -1]])
        shape = pot.shape
        U = np.zeros((shape[0], shape[1], 2, 2), dtype=np.complex128)
        for i in range(shape[0]):
            for j in range(shape[1]):
                ### H_r = pot(r) * si + g(r)*mu*(B[0] * sx + B[1] * sy + B[2] * sz)
                a = g[i,j] * ct.mu_B * np.array([B[0], B[1], B[2]])  # spin coefficients
                
                ### U = exp(-1j * H_r * dt / hbar)
                U_pot = np.exp(-1j * pot[i,j] * dt / hbar)
                a_norm = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
                sa = (a[0] * sx + a[1] * sy + a[2] * sz)/a_norm
                U_z = si * np.cos(a_norm * dt / hbar / 2) - 1j * sa * np.sin(a_norm * dt / hbar)

                U[i,j] = U_z * U_pot

        return U


    @jit(nopython=True, cache=True)
    def momentum_operator(PX, PY, dt, me, hbar, alpha, beta, theta):
        # define the Pauli matrices
        si = np.array([[1, 0], [0, 1]])
        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.array([[1, 0], [0, -1]])
        shape = PX.shape
        U = np.zeros((shape[0], shape[1], 2, 2), dtype=np.complex128)
        for i in range(shape[0]):
            for j in range(shape[1]):
                px = PX[i,j] * np.cos(theta) - PY[i,j] * np.sin(theta)
                py = PX[i,j] * np.sin(theta) + PY[i,j] * np.cos(theta)

                ### H_p = (px**2 + py**2)/(2*me) * si - (alpha*py + beta*px) * sx + (alpha*px + beta*py) * sy
                a = [- (alpha*py + beta*px), alpha*px + beta*py]  # spin coefficients
                KE = (px**2 + py**2)/(2*me)  # kinetic energy

                ### U = exp(-1j * H_p * dt / hbar)
                U_KE = np.exp(-1j * KE * dt / hbar)

                a_norm = np.sqrt(a[0]**2 + a[1]**2)
                sa = (a[0] * sy + a[1] * sx)/a_norm
                U_SO = si * np.cos(a_norm * dt / hbar) - 1j * sa * np.sin(a_norm * dt / hbar)

                U[i,j] = U_SO * U_KE

        return U
    
    exp_k = momentum_operator(PX, PY, dt, ct.me, hbar, alpha, beta, theta)
    

    # time variables
    tlist = np.arange(0, potential.pulse['length'], dt)
    # eval_steps = np.linspace(0, len(tlist)-1, num_evals, dtype=int)
    eval_step = len(tlist) // num_evals
    xr = np.linspace(xrange[0], xrange[1], num_evals)
    yr = np.linspace(yrange[0], yrange[1], num_evals)

    # get the voltages of the pulse
    pls = potential.pulse.resolve(dt)

    # initialize the output arrays
    F = np.zeros((2, num_evals))  # fidelity
    result = np.zeros((num_evals, Ny, Nx, 2, 1), dtype=np.complex128)  # wavefunction

    # initial wavefunction (space and spin)
    _, psi_r = get_instantaneous_eigenfunction(potential, 0, xrange[0], yrange[0])
    sx = np.array([[0, 1], [1, 0]]); sy = np.array([[0, -1j], [1j, 0]]); sz = np.array([[1, 0], [0, -1]])
    spinen, spineig = eig(B[0] * sx + B[1] * sy + B[2] * sz)
    spin_ground = spineig[:, np.argmin(spinen)].reshape((2, 1))
    psi_r = np.einsum('ij,kl->ijkl', psi_r, spineig)    

    # define the function of split operator method
    @jit(nopython=True, cache=True)
    def split_operator(eval_step, psi_r, exp_k, gpot, pls_slice):
        for i_step in range(eval_step):
            # time evolution (sprit operator method)
            pot = _get_potential(pls_slice[i_step], gpot)
            exp_r = positional_operator(pot, dt/2, hbar, B)
            psi_r = np.einsum('ijkl,ijlm->ijkm', exp_r, psi_r)
            psi_p = fft.fft2(psi_r, axes=(0, 1))
            psi_p = np.einsum('ijkl,ijlm->ijkm', exp_k, psi_p)
            psi_r = fft.ifft2(psi_p, axes=(0, 1))
            psi_r = np.einsum('ijkl,ijlm->ijkm', exp_r, psi_r)

        return psi_r

    ###### main loop ################################################################
    for i_eval in tqdm(range(0, num_evals), total=num_evals):
        ### aquire the potential for the current evaluation step
        pls_slice = pls[i_eval*eval_step: (i_eval+1)*eval_step]
        
        # shuttling
        psi_r = split_operator(eval_step, psi_r, exp_k, potential.gpot, pls_slice)
        
        # evaluation
        # t = tlist[i_eval*eval_step + eval_step - 1]
        # _, psi_r_inst = get_instantaneous_eigenfunction(potential, t, xr[i_eval], yr[i_eval])
        # F[0][i_eval] = t
        # F[1][i_eval] = np.abs(inner_prod(psi_r, psi_r_inst, dx, dy))**2  # fidelity
        result[i_eval] = psi_r
        if np.isnan(F[1][i_eval]):
            raise ValueError("Fidelity contains NaN values.")
    
    return result

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

    A = 0.2  # amplitude of the pulse (V)
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
    result = sch.TE_solver_with_spin(ppot, xrange, yrange, dt=dt, num_evals=num_evals,
                                     alpha=0.1, beta=0.1, theta=0., B=(0., 0., 1.),
                                    g_sd=3., g_scale=20e-9)

