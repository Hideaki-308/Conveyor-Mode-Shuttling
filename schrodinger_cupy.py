import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import scipy
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.linalg import eig
from numba import jit
from tqdm import tqdm
from pulse_potentials import Potential
from utility import inner_prod, generate_fructuation, partial_matrix_prod
import pathlib
from pyevtk.hl import gridToVTK


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
        pot = cp.sum(-val_ext * gpot, axis=0)
        return pot
    elif val.ndim == 2:
        val_ext = val[:, :, cp.newaxis, cp.newaxis]
        gpot_ext = gpot[cp.newaxis, :, :, :]
        pot = cp.sum(-val_ext * gpot_ext, axis=1)
        return pot
    
def check_unitary(U):
    UdU = U.conj().swapaxes(-1, -2) @ U
    I = cp.eye(2)
    I = cp.ones(U.shape[:-2])[..., cp.newaxis, cp.newaxis] * I

    return cp.allclose(I, UdU)

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
    pot = potential.get_potential(t)

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


def TE_solver_cupy(potential, xrange, yrange, dt=0.1e-9, num_evals=50,
                   alpha=0., beta=0., theta=0., B=(0., 0., 0.),
                   g_sd=0., g_scale=10e-9, save_path=None):
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

    # retrieve the parameters from the potential object
    ct = potential.consts
    hbar = ct.hbar
    muB = ct.muB
    g = ct.g_factor + generate_fructuation(potential, g_sd, g_scale, seed=42)
    g = cp.asarray(g)
    gpot = cp.asarray(potential.gpot)

    # get the grid parameters
    X = cp.asarray(potential.x); Y = cp.asarray(potential.y)
    x = X[0,:]; y = Y[:,0]
    Nx = len(x); Ny = len(y)
    dx = x[1] - x[0]; dy = y[1] - y[0]

    if save_path is not None:
        # define the numpy grid for saving the wavefunction
        X_np = cp.asnumpy(X).reshape((Ny, Nx, 1))
        Y_np = cp.asnumpy(Y).reshape((Ny, Nx, 1))
        Z_np = np.zeros((Ny, Nx, 1))
        if not pathlib.Path(save_path).exists():
            pathlib.Path(save_path).mkdir(parents=True)
    
    # pauli matrices
    si = cp.eye(2, dtype=cp.complex128)
    sx = cp.asarray([[0, 1], [1, 0]], dtype=cp.complex128)
    sy = cp.asarray([[0, -1j], [1j, 0]], dtype=cp.complex128)
    sz = cp.asarray([[1, 0], [0, -1]], dtype=cp.complex128)

    # define momentum coordinates
    dpx = 2*np.pi*hbar/(Nx*dx); dpy = 2*np.pi*hbar/(Ny*dy)
    px = cp.arange(-Nx/2, Nx/2) * dpx
    py = cp.arange(-Ny/2, Ny/2) * dpy
    PX, PY = cp.meshgrid(px, py)
    PX = cp.fft.ifftshift(PX)  # change the order to match the order of fft
    PY = cp.fft.ifftshift(PY)

    # time evolution operator in real space
    def positional_operator(pot, dt):
        shape = pot.shape
        U = cp.zeros((*shape, 2, 2), dtype=cp.complex128)

        ### H_r = pot(r) * si + g(r)*mu*(B[0] * sx + B[1] * sy + B[2] * sz)
        a = g[...,cp.newaxis] * muB * cp.array(B)  # spin coefficients
        a = a[..., cp.newaxis, cp.newaxis, :]
        a_norm = cp.linalg.norm(a, axis=-1)
        sa = (a[...,0] * sx + a[...,1] * sy + a[...,2] * sz) / (a_norm + 1e-20)
        U_z = si*cp.cos(a_norm*dt/hbar) - 1j*sa*cp.sin(a_norm*dt/hbar)
        U_pot = cp.exp(-1j*pot*dt/hbar)[...,cp.newaxis, cp.newaxis]
        U = U_z * U_pot

        return U
    
    # time evolution operator in momentum space
    def momentum_operator(PX, PY, dt, alpha, beta, theta):
        U = cp.zeros((Ny, Nx, 2, 2), dtype=cp.complex128)

        # convert the momentum to the crystal frame
        px = PX * np.cos(theta) - PY * np.sin(theta)
        py = PX * np.sin(theta) + PY * np.cos(theta)

        ### H_p = (px**2 + py**2)/(2*me) * si - (alpha*py + beta*px) * sx + (alpha*px + beta*py) * sy
        a = cp.array([- (alpha*py + beta*px), alpha*px + beta*py])
        a = a[:,:,:,cp.newaxis,cp.newaxis]
        a_norm = cp.linalg.norm(a, axis=0)
        sa = (a[0] * sx + a[1] * sy) / (a_norm + 1e-20)
        U_SO = si*cp.cos(a_norm*dt/hbar) - 1j*sa*cp.sin(a_norm*dt/hbar)

        KE = (px**2 + py**2)/(2*ct.me)
        U_KE = cp.exp(-1j*KE*dt/hbar)[:,:,cp.newaxis, cp.newaxis]
        U = U_SO * U_KE

        return U
    
    exp_k = momentum_operator(PX, PY, dt, alpha, beta, theta)
    # if not check_unitary(exp_k):
    #     raise ValueError('The momentum operator is not unitary.')

    # time variables
    tlist = cp.arange(0, potential.pulse['length'], dt)
    tlist_chank = cp.array_split(tlist, num_evals)
    eval_step = len(tlist) // num_evals  # time steps per each evaluation
    xr = cp.linspace(xrange[0], xrange[1], num_evals)
    yr = cp.linspace(yrange[0], yrange[1], num_evals)

    # initialize the output
    F = cp.zeros((2, num_evals))  # fidelity
    result = cp.zeros((num_evals, Ny, Nx, 2, 1), dtype=cp.complex128)  # wavefunction

    # initial wavefunction
    _, psi_r = get_instantaneous_eigenfunction(potential, 0, xrange[0], yrange[0])
    psi_r = cp.asarray(psi_r)
    spinen, spineig = cp.linalg.eigh(B[0] * sx + B[1] * sy + B[2] * sz)
    spin_ground = spineig[:, cp.argmin(spinen)].reshape((2, 1))
    psi_r = cp.einsum('ij,kl->ijkl', psi_r, spin_ground)

    ### main loop ################################################################
    for i_eval in tqdm(range(0, num_evals), total=num_evals):
        pls_slice = potential.pulse.resolve(tlist_chank[i_eval])
        pot = _get_potential(pls_slice, gpot)
        exp_r = positional_operator(pot, dt/2)
        # if not check_unitary(exp_r):
        #     raise ValueError('The positional operator is not unitary.')
        if not cp.all(cp.isfinite(exp_r)):
            print(exp_r)
            raise ValueError('unexpected value is detected in the exp_r.')

        for i_t in range(len(tlist_chank[i_eval])):
            psi_r = cp.matmul(exp_r[i_t], psi_r)
            psi_k = cp.fft.fft2(psi_r, axes=(0, 1))
            psi_k = cp.matmul(exp_k, psi_k)
            psi_r = cp.fft.ifft2(psi_k, axes=(0, 1))
            psi_r = cp.matmul(exp_r[i_t], psi_r)

        if not cp.all(cp.isfinite(psi_r)):
            print(psi_r)
            raise ValueError('unexpected value is detected in the wavefunction.')
        result[i_eval] = psi_r

        if save_path is not None:
            wf_up = cp.asnumpy(cp.abs(psi_r[:,:, 0, :])**2)
            wf_dn = cp.asnumpy(cp.abs(psi_r[:,:, 1, :])**2)
            data_dict = {'spin_up': wf_up, 'spin_down': wf_dn}
            gridToVTK(save_path + f'/wavefunction_{i_eval}', X_np, Y_np, Z_np, pointData=data_dict)

    return result


if __name__ == '__main__':
    import numpy as np
    import cupy as cp
    import matplotlib.pyplot as plt
    import pathlib
    import pulse_potentials as pp
    # import schrodinger as sch
    import schrodinger_cupy as sch
    from constants import Constants
    from pulse_potentials import Potential


    script_dir = pathlib.Path().resolve()
    data_dir = script_dir / 'output' / 'pot'

    consts = Constants('Si/SiGe')
    gate_names = ['C1', 'C2', 'B1', 'B2', 'B3', 'B4', 'B5', 'P1', 'P2', 'P3', 'P4']
    ppot = Potential(data_dir, gate_names, consts)

    T = 1 / 1e9  # period of the pulse (s)
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

    from time import time

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

    dt_list = [1e-15]
    num_evals = 1000
    alpha = 1e-11
    beta = 1e-11
    theta = 0
    Mfield = (0, 0, 0.1)
    g_sd = 2
    g_scale = 10e-9

    for dt in dt_list:
        print(f'dt = {dt:.0e}')
        t0 = time()
        result = sch.TE_solver_cupy(ppot, xrange, yrange, dt=dt, num_evals=num_evals,
                                    alpha=alpha, beta=beta, theta=theta, B=Mfield,
                                    g_sd=g_sd, g_scale=g_scale)
        print(f'Elapsed time: {time() - t0:.2f} s')