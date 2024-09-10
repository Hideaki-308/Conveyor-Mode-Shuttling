import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from numba import jit


class Potential:
    def __init__(self, fdir, gate_names, consts):
        self.fdir = fdir
        self.gate_names = gate_names
        self.consts = consts
        self.x = None
        self.y = None
        self.pulse = None

        # assign index to each gate
        self.gate_idx = {g: i for i, g in enumerate(gate_names)}

        self.load_files(self.fdir, self.gate_names)
        
    def load_files(self, fdir, gate_names):
        self.x = np.loadtxt(fdir/'x.txt', delimiter=',') * 1e-9
        self.y = np.loadtxt(fdir/'y.txt', delimiter=',') * 1e-9
        self.gpot = np.zeros((len(gate_names)+1, self.x.shape[0], self.x.shape[1]))
        self.gpot[-1] = np.loadtxt(fdir/'zero.txt', delimiter=',') * self.consts.e

        for gate in gate_names:
            self.gpot[self.gate_idx[gate]] = np.loadtxt(fdir/f'{gate}_m1.txt', delimiter=',') * self.consts.e
    
    def get_potential(self, arg):
        """
        Get the interpolated potential from a given set of voltages or time.

        Parameters
        ----------
        arg : float or dict
            If float, it is the time. If dict, it is the gate voltages.

        Returns
        -------
        pot : np.ndarray
            Interpolated potential
        """
        if isinstance(arg, (int, float)):
            if self.pulse is None:
                raise ValueError("Pulse not defined.")
            return self._get_potential(self.pulse(arg), self.gpot)
        elif isinstance(arg, dict):
            val = np.zeros(len(self.gate_names) + 1)
            val[-1] = -1
            for gate in arg.keys():
                val[self.gate_idx[gate]] = arg[gate]
            return self._get_potential(val, self.gpot)


    @staticmethod
    @jit(nopython=True, cache=True)
    def _get_potential(val, gpot):
        """
        Get the interpolated potential for a given set of voltages.

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
        
        # # this is equivalent to the above code
        # pot = gpot[-1].copy()
        # for i, v in enumerate(val):
        #     pot += -v * gpot[i]
        
        # return pot
    
    def make_pulse(self, pulse_length, pulse_shape, pulse_name='pulse'):
        """
        Make a pulse object.

        Parameters
        ----------
        pulse_name : str
            Name of the pulse.
        pulse_length : float
            Length of the pulse.
        pulse_shape : dict
            Dictionary of control variables and their values (function of t).
        dt : float
            Time step for the pulse.
        
        Returns
        -------
        pulse : dict
            Dictionary of the pulse.
        """

        pulse = Pulse(pulse_length, pulse_shape, self.gate_names, pulse_name)
        self.pulse = pulse

        return pulse
    
    def set_pulse(self, pulse):
        self.pulse = pulse

    def plot_potential(self, val, linecut=False):
        """
        Plot the potential for a given time.

        Parameters
        ----------
        t : float, optional
            Time.
        val : float or dict
            Value of the gate voltages. If float, it is the time.
        linecut : bool, optional
            Plot a linecut at y=0, by default False.
        """
        
        pot = self.get_potential(val)
        
        plt.figure(figsize=(9, 3))
        if not linecut:
            plt.imshow(pot, origin='lower', extent=[self.x.min()*1e9, self.x.max()*1e9, self.y.min()*1e9, self.y.max()*1e9])
            plt.xlabel('x (nm)')
            plt.ylabel('y (nm)')
            plt.colorbar()
            plt.tight_layout()
            plt.show()
        else:
            plt.plot(self.x, pot[np.argmin(np.abs(self.y))])
            plt.show()
    


class Pulse(dict):
    def __init__(self, pulse_length, pulse_shape, gate_names, pulse_name='pulse'):
        """
        Make a pulse object.

        Parameters
        ----------
        pulse_name : str
            Name of the pulse.
        pulse_length : float
            Length of the pulse.
        pulse_shape : dict
            Dictionary of control variables and their values (can be function of t).
        gate_names : list
            List of gate names.
        
        Returns
        -------
        pulse : dict
            Dictionary of the pulse.
        """
        self['name'] = pulse_name
        self['length'] = pulse_length
        self['shape'] = pulse_shape
        # assign index to each gate
        self.gate_idx = {g: i for i, g in enumerate(gate_names)}
    
    def __call__(self, t):
        """
        Calculate the potential for the current pulse.

        Parameters
        ----------
        t : float
            Time.
        
        Returns
        -------
        potential : np.ndarray
            Interpolated potential.
        """
        voltages = np.zeros(len(self['shape']) + 1)
        voltages[-1] = -1
        for g, v in self['shape'].items():
            if isinstance(v, (int, float)):
                voltages[self.gate_idx[g]] = v
            elif callable(v):
                voltages[self.gate_idx[g]] = v(t)
            else:
                raise ValueError(f"Invalid value for {g}.")
        
        return voltages
    
    def plot_pulse(self, gate_names=None):
        """
        Plot the pulse shape as a voltage applied to each gates.

        Parameters
        ----------
        gate_names : list, optional
            If not None, plot only the specified gates, by default None.
        """
        if gate_names is None:
            gate_names = self['shape'].keys()
        
        tlist = np.linspace(0, self['length'], 101)
        for g in gate_names:
            if isinstance(self['shape'][g], (int, float)):
                plt.plot(tlist, np.full_like(tlist, self['shape'][g]), label=g)
            else:
                plt.plot(tlist, self['shape'][g](tlist), label=g)
        plt.xlabel('t')
        plt.ylabel('V')
        plt.legend()
        plt.show()

    def resolve(self, dt):
        """
        Resolve the pulse shape to a list of voltages.

        Parameters
        ----------
        dt : float
            Time step.
        
        Returns
        -------
        pulse : np.ndarray
            2D array of voltages. First index is gate index and second index is time index.
        """
        if 'voltages' in self.keys():
            return self['voltages']

        tlist = np.arange(0, self['length'], dt)

        self['voltages'] = np.zeros((len(tlist), len(self['shape'])+1))
        self['voltages'][:,-1] = -np.ones_like(tlist)
        for g, i in self.gate_idx.items():
            if callable(self['shape'][g]):
                self['voltages'][:,i] = np.array([self['shape'][g](t) for t in tlist])
            else:
                self['voltages'][:,i] = np.full_like(tlist, self['shape'][g])

        return self['voltages']
