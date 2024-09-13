from scipy import constants
import numpy as np

class Constants:
    
    def __init__(self, material_system="vacuum"):
        '''
        
        Keyword Arguments
        -----------------
        material_system : string, optional
            String specifying which material system the constant class is for.
            Currently allowed systems are: ["vacuum","Si/SiO2", "Si/SiGe", "GaAs"].
            Default is vacuum.

        Returns
        -------
        None.

        '''
        # Default units are SI [International system]
        self.units = "SI"
        
        ### Mathematical constants ###

        # Pi
        self.pi = constants.pi         

        ### Physical constants ###

        # Planck's constant [J*s]
        self.h = constants.h
        # Reduced Planck's constant [J*s]              
        self.hbar = constants.hbar   
        # Electron charge [C]
        self.e = constants.e
        # Free electron mass [kg]
        self.m0 = constants.m_e
        # Speed of light [m/s]
        self.c = constants.c
        # Bohr magneton [J/T]
        self.muB = constants.physical_constants['Bohr magneton'][0]
        # Vacuum permitivity [F/m]
        self.eps0 = constants.epsilon_0
        # Boltzmann constant [J/K]
        self.kB = constants.physical_constants['Boltzmann constant'][0]




        # Material system constants
        # Supported material systems include [Si/SiO2, Si/SiGe, GaAs, air]
        self.material_system = material_system
        
        if material_system == "Si/SiO2":
            self.epsR = 7.8                 # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0*0.191         # Effective mass [kg]
            self.mu_2 = 2.2 * (1e-9)**2     # Quadratic Stark shift coefficient [m^2/V^2]
        elif material_system == "Si/SiGe":
            self.epsR = 12.375              # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0*0.191         # Effective mass [kg]
            self.mu_2 = None                # Quadratic Stark shift coefficient [m^2/V^2]
        elif material_system == "GaAs":
            self.epsR = 12.4                # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0*0.067         # Effective mass [kg]
            self.mu_2 = None                # Quadratic Stark shift coefficient [m^2/V^2]
        elif material_system.lower() == "vacuum" or "air":
            self.epsR = 1                   # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0               # Effective mass [kg]
            self.mu_2 = None                # Quadratic Stark shift coefficient [m^2/V^2]
        else:
            # If no or unrecognized material system specified assume "air"
            self.epsR = 1                   # Dielectric constant
            self.eps = self.eps0*self.epsR  # Permitivity [F/m]
            self.me = self.m0               # Effective mass [kg]
            self.mu_2 = None                # Quadratic Stark shift coefficient [m^2/V^2]
            print("WARNING: Material system either not recognized or defined.\n"+
                  "Assuming ""vacuum"" as the material system.\n"+
                  "Allowed material systems are: [""Si/SiO2"", ""Si/SiGe"","+
                                                 " ""GaAs"", ""vacuum""]")   
            