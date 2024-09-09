#%%
import pathlib
import numpy as np
import itertools

from qtcad.device import constants as ct
from qtcad.device import analysis as an
from qtcad.device.mesh3d import Mesh, SubMesh
from qtcad.device import materials as mt
from qtcad.device import Device, SubDevice
from qtcad.device import io
from qtcad.device.poisson_linear import Solver, SolverParams
#%%
xy_saved = False

def slice2txt(mesh, array, z, dirname, filename, resx=256, resy=128):
    global xy_saved
    x,y,_ = mesh.glob_nodes.T

    xlist = np.linspace(np.min(x), np.max(x), resx, dtype='float64')
    ylist = np.linspace(np.min(y), np.max(y), resy, dtype='float64')
    ylist[0] += 0.1e-9
    ylist[-1] -= 0.1e-9
    # save x,y coordinates if not exist
    if not xy_saved:
        xx, yy = np.meshgrid(xlist, ylist)
        np.savetxt(dirname/"x.txt", xx * 1e9, delimiter=",")
        np.savetxt(dirname/"y.txt", yy * 1e9, delimiter=",")
        xy_saved = True
    
    dataset = np.zeros((resy,resx), dtype='f2')
    print('Writing to ', dirname/filename)
    for i, yi in enumerate(ylist):
        begin = (np.min(x), yi, z)
        end   = (np.max(x), yi, z)
        posx,EC = an.linecut(mesh, array, begin, end, method="pyvista", res=resx-1)
        posx = posx + np.min(x)
        dataset[i] = list(EC)
        
    np.savetxt(dirname/filename, dataset, delimiter=",")
# %%
# Input file
script_dir = pathlib.Path(__file__).parent.resolve()
path_mesh = str(script_dir/'meshes'/'moving_potential.msh2')
path_geo = script_dir / 'meshes' / 'moving_potential.geo_unrolled'
output_dir = script_dir / 'output' / 'pot' 
if not output_dir.exists():
    output_dir.mkdir()

# Load the mesh
scaling = 1e-6
mesh = Mesh(scaling,path_mesh)
# mesh.show()

# Create device
d = Device(mesh, conf_carriers='e')
d.set_temperature(0.1)

# Work function of Ti/Pt gates at midgap of Si
work_function = mt.Si.chi + mt.Si.Eg/2

# 2DEG depth
two_deg_depth = -57.0 * 1e-9


mt.SiGe.set_alloy_composition(0.3)

# Set up materials in heterostructure stack
d.new_region("insulator1", mt.Al2O3_ideal)
d.new_region("insulator2", mt.Al2O3_ideal)
d.new_region("cap", mt.Si)
d.new_region("spacer1", mt.SiGe)
d.new_region("spacer1_1DEC", mt.SiGe)
d.new_region("qw", mt.Si)
d.new_region("qw_1DEC", mt.Si)
d.new_region("spacer2", mt.SiGe)
d.new_region("spacer2_1DEC", mt.SiGe)

# Add band shifts
d.add_to_ref_potential(-0.15, region='qw')
d.add_to_ref_potential(-0.15, region='qw_1DEC')

d.new_insulator("cap", mt.Si)

# set the extra potential for the defects
if False:
    defect_pos = np.array([[0.0, 70e-9, two_deg_depth]])
    defect_charge = 1.0
    def charge_density(x, y, z):
        return defect_charge * np.exp(-((x-defect_pos[0,0])**2 + (y-defect_pos[0,1])**2 + (z-defect_pos[0,2])**2)/1e-18)
    d.set_charge_density(charge_density)

# Define the dot region as a list of region labels
dot_region = ["spacer1_1DEC", "qw_1DEC", "spacer2_1DEC"]
submesh = SubMesh(mesh, dot_region)
d.set_dot_region(submesh)
# %%
# resolution
resx = 128
resy = 64


gate_names = ['C1', 'C2']
gate_names += [f'B{i}' for i in range(1, 6)]
gate_names += [f'P{i}' for i in range(1, 5)]

# Define the gate regions and set to zero voltage
for g in gate_names:
    d.new_gate_bnd(g, 0, work_function)

# solve the poisson equation in zero-voltage condition
solver_params = SolverParams({'tol': 0.001})
solver_params.verbose = False
poisson_solver = Solver(d, solver_params)
poisson_solver.solve()

# Save the zero-voltage potential
outfile = 'zero.txt'
subdevice = SubDevice(d, submesh)
slice2txt(subdevice.mesh, subdevice.cond_band_edge()/ct.e, two_deg_depth, output_dir, outfile, resx=resx, resy=resy)
zero_pot = subdevice.cond_band_edge()/ct.e

for g in gate_names:
    d.set_applied_potential(g, -1.0)
    poisson_solver.solve()

    outfile = f'{g}_m1.txt'
    subdevice = SubDevice(d, submesh)
    slice2txt(subdevice.mesh, subdevice.cond_band_edge()/ct.e - zero_pot, two_deg_depth, output_dir, outfile, resx=resx, resy=resy)

    d.set_applied_potential(g, 0.0)
