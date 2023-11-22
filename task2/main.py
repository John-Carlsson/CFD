# MTF073 Computational Fluid Dynamics
# Task 2: convection-diffusion
# Håkan Nilsson, 2023
# Department of Mechanics and Maritime Sciences
# Division of Fluid Dynamics
# Note that this is not efficient code. It is for educational purposes!

# The code assumes that the folder with data is in the same path as this file

# Clear all variables when running entire code:
from IPython import get_ipython
get_ipython().run_line_magic('reset', '-sf')
# Packages needed
import numpy as np
import matplotlib.pyplot as plt
# Close all plots when running entire code:
plt.close('all')
# Set default font size in plots:
plt.rcParams.update({'font.size': 12})
import sys # For sys.exit()
import os # For saving plots

#===================== Inputs =====================

# Geometric and mesh inputs (mesh is read from file)
grid_type = 'coarse' # Either 'coarse' or 'fine'
caseID    =   20       # Your case number to solve

# Solver inputs
nIter  = 2000    # set maximum number of iterations
resTol = 0.001 # set convergence criteria for residuals
solver = 'Gauss-Seidel'  # Either Gauss-Seidel or TDMA

# Physical properties
rho    =   1    # Density
k      =   1    # Thermal conductivity 
Cp     =   200  # Specific heat
gamma  =   k/Cp # Calculated diffusion coefficient

#====================== Code ======================

# Read grid and velocity data:
match caseID:
    case 1 | 2 | 3 | 4 | 5:
        grid_number = 1
    case 6 | 7 | 8 | 9 | 10:
        grid_number = 2
    case 11 | 12 | 13 | 14 | 15:
        grid_number = 3
    case 16 | 17 | 18 | 19 | 20:
        grid_number = 4
    case 21 | 22 | 23 | 24 | 25:
        grid_number = 5
    case _:
        sys.exit("No mesh for chosen caseID!")
path = 'data/grid%d/%s_grid' % (grid_number,grid_type)
pointXvector = np.genfromtxt('%s/xc.dat' % (path)) # x node coordinates
pointYvector = np.genfromtxt('%s/yc.dat' % (path)) # y node coordinates
u_datavector = np.genfromtxt('%s/u.dat' % (path))  # u velocity at the nodes
v_datavector = np.genfromtxt('%s/v.dat' % (path))  # v veloctiy at the nodes

# Preparation of "nan", to fill empty slots in consistently numbered arrays.
# This makes it easier to check in Variable Explorer that values that should
# never be set are never set (or used). Plots simply omit nan values.
nan = float("nan")

# Allocate arrays (nan used to make clear where values need to be set)
# Note that some arrays could actually be 1D since they only have a variation
# in one direction, but they are kept 2D so the indexing is similar for all.
mI     = len(pointXvector);          # Number of mesh points X direction
mJ     = len(pointYvector);          # Number of mesh points X direction
nI     = mI + 1;                     # Number of nodes in X direction, incl. boundaries
nJ     = mJ + 1;                     # Number of nodes in Y direction, incl. boundaries
pointX = np.zeros((mI,mJ))*nan       # X coords of the mesh points, in points
pointY = np.zeros((mI,mJ))*nan       # Y coords of the mesh points, in points
nodeX  = np.zeros((nI,nJ))*nan       # X coords of the nodes, in nodes
nodeY  = np.zeros((nI,nJ))*nan       # Y coords of the nodes, in nodes
dx_PE  = np.zeros((nI,nJ))*nan       # X distance to east node, in nodes
dx_WP  = np.zeros((nI,nJ))*nan       # X distance to west node, in nodes
dy_PN  = np.zeros((nI,nJ))*nan       # Y distance to north node, in nodes
dy_SP  = np.zeros((nI,nJ))*nan       # Y distance to south node, in nodes
dx_we  = np.zeros((nI,nJ))*nan       # X size of the control volume, in nodes
dy_sn  = np.zeros((nI,nJ))*nan       # Y size of the control volume, in nodes
aE     = np.zeros((nI,nJ))*nan       # Array for east coefficient, in nodes
aW     = np.zeros((nI,nJ))*nan       # Array for wect coefficient, in nodes
aN     = np.zeros((nI,nJ))*nan       # Array for north coefficient, in nodes
aS     = np.zeros((nI,nJ))*nan       # Array for south coefficient, in nodes
aP     = np.zeros((nI,nJ))*nan       # Array for central coefficient, in nodes
Su     = np.zeros((nI,nJ))*nan       # Array for source term for temperature, in nodes
Sp     = np.zeros((nI,nJ))*nan       # Array for source term for temperature, in nodes
T      = np.zeros((nI,nJ))*nan       # Array for temperature, in nodes
De     = np.zeros((nI,nJ))*nan       # Diffusive coefficient for east face, in nodes
Dw     = np.zeros((nI,nJ))*nan       # Diffusive coefficient for west face, in nodes
Dn     = np.zeros((nI,nJ))*nan       # Diffusive coefficient for north face, in nodes
Ds     = np.zeros((nI,nJ))*nan       # Diffusive coefficient for south face, in nodes
Fe     = np.zeros((nI,nJ))*nan       # Convective coefficients for east face, in nodes
Fw     = np.zeros((nI,nJ))*nan       # Convective coefficients for west face, in nodes
Fn     = np.zeros((nI,nJ))*nan       # Convective coefficients for north face, in nodes
Fs     = np.zeros((nI,nJ))*nan       # Convective coefficients for south face, in nodes
P      = np.zeros((nI,nJ))*nan       # Array for TDMA, in nodes
Q      = np.zeros((nI,nJ))*nan       # Array for TDMA, in nodes
u      = u_datavector.reshape(nI,nJ) # Values of x-velocity, in nodes
v      = v_datavector.reshape(nI,nJ) # Values of y-velocity, in nodes
res    = []                          # Array for appending residual each iteration
# Set wall velocities to exactly zero:
u[u == 1e-10] = 0
v[v == 1e-10] = 0

# Set point coordinates:
for i in range(0, mI):
    for j in range(0, mJ):
        pointX[i,j] = pointXvector[i]
        pointY[i,j] = pointYvector[j]

# Calculate length and height:
L = pointX[mI-1,0] - pointX[0,0]
H = pointY[0,mJ-1] - pointY[0,0]

# Calculate node coordinates (same for equidistant and non-equidistant):
# Internal nodes:
for i in range(0, nI):
    for j in range(0, nJ):
        if i > 0 and i < nI-1:
            nodeX[i,j] = 0.5*(pointX[i,0] + pointX[i-1,0])
        if j > 0 and j < nJ-1:
            nodeY[i,j] = 0.5*(pointY[0,j] + pointY[0,j-1])
# Boundary nodes:
nodeX[0,:]  = pointX[0,0]  # Note: corner points needed for contour plot
nodeY[:,0]  = pointY[0,0]  # Note: corner points needed for contour plot
nodeX[-1,:] = pointX[-1,0] # Note: corner points needed for contour plot
nodeY[:,-1] = pointY[0,-1] # Note: corner points needed for contour plot

# Calculate distances
# Keep 'nan' where values are not needed!
for i in range(1, nI-1):
    for j in range(1, nJ-1):
        dx_PE[i,j] = nodeX[i+1,j] - nodeX[i,j]
        dx_WP[i,j] = nodeX[i,j] - nodeX[i-1,j]
        dy_PN[i,j] = nodeY[i,j+1] - nodeY[i,j]
        dy_SP[i,j] = nodeY[i,j] - nodeY[i,j-1]
        dx_we[i,j] = pointX[i,j] - pointX[i-1,j]
        dy_sn[i,j] = pointY[i,j] - pointY[i,j-1]

# Initialize dependent variable array and Dirichlet boundary conditions
# Note that a value is needed in all nodes for contour plot
# Default:
for i in range(0,nI):
    for j in range(0,nJ):
        # ADD CODE HERE
        T[i,j] = 0

# Inlets (found by velocity into domain), walls (found by zero velocity):
for i in range(nI):
    j = nJ-1
    # ADD CODE HERE
    j = 0
    # ADD CODE HERE
for j in range(nJ):
    i = nI-1
    # ADD CODE HERE
    i = 0
    # ADD CODE HERE

# Set default/constant source terms:
for i in range(1,nI-1):
    for j in range(1,nJ-1):
        pass

# Calculate constant diffusive (D) and convective (F) coefficients
def interpolate(y1,y2,x1,x2):
    """" Interpolates
    Args:
        y1 : First point y value
        x1 : First cell width/height
        y2 : First point y value
        x2 : Second cell width/height

    Returns:
        Interpolated value
    """
    dx = x2/2 + x1/2
    return  y1 + (y2-y1)/dx * x1/2


for i in range(1,nI-1):
    for j in range(1,nJ-1):
        De[i,j] = gamma/dx_PE[i,j] * dy_sn[i,j]
        Dw[i,j] = gamma/dx_WP[i,j] * dy_sn[i,j]
        Dn[i,j] = gamma/dy_PN[i,j] * dx_we[i,j]
        Ds[i,j] = gamma/dy_SP[i,j] * dx_we[i,j]
        
        fxe = 0
        fxw = 0
        fyn = 0
        fys = 0
        
        Fe[i,j] = rho*interpolate(u[i,j],u[i+1,j],dx_we[i,j],dx_we[i+1,j]) * dy_sn[i,j]
        Fw[i,j] = rho*interpolate(u[i,j],u[i-1,j],dx_we[i,j],dx_we[i-1,j]) * dy_sn[i,j]
        Fn[i,j] = rho*interpolate(u[i,j],u[i,j+1],dx_we[i,j],dx_we[i,j+1]) * dx_we[i,j]
        Fs[i,j] = rho*interpolate(u[i,j],u[i,j-1],dx_we[i,j],dx_we[i,j-1]) * dx_we[i,j]

# Calculate constant Hybrid scheme coefficients (not taking into account boundary conditions)
for i in range(1,nI-1):
    for j in range(1,nJ-1):
        fxe = # ADD CODE HERE
        fxw = # ADD CODE HERE
        fyn = # ADD CODE HERE
        fys = # ADD CODE HERE
        
        aE[i,j] = max([-Fe,(De-Fe/2), 0])
        aW[i,j] = max([Fw,(Dw+Fw/2), 0])
        aN[i,j] = max([-Fn,(Dn-Fn/2), 0])
        aS[i,j] = max([Fs,(Ds+Fs/2), 0])
        
# At outlets (found by velocity out of domain), set homogeneous Neumann
for j in range(1,nJ-1):
    i = nI-2
    # ADD CODE HERE
    i = 1
    # ADD CODE HERE
for i in range(1,nI-1):
    j = nJ-2
    # ADD CODE HERE
    j = 1
    # ADD CODE HERE

# (Homogeneous) Neumann walls (found by zero velocity):
for j in range(1,nJ-1):
    # East wall:
    i = nI-2
    # ADD CODE HERE
    # West wall:
    i = 1
    # ADD CODE HERE
for i in range(1,nI-1):
    # North wall:
    j = nJ-2
    # ADD CODE HERE
    # South wall:
    j = 1
    # ADD CODE HERE

for i in range(1,nI-1):
    for j in range(1,nJ-1):       
        aP[i,j] = # ADD CODE HERE

# The following loop is for the linear solver.
for iter in range(nIter): 
    # Solve for T using Gauss-Seidel
    if solver == 'Gauss-Seidel':
        # One direction
        # ADD CODE HERE
        
        # Other direction
        # ADD CODE HERE
    
    # Solve for T using TDMA
    if solver == 'TDMA':
        # One direction
        # ADD CODE HERE
        
        # Other direction
        # ADD CODE HERE

    # Copy T to walls where (non-)homogeneous Neumann is applied:
    # Note that specified heat flux is positive INTO computational domain!
    for j in range(1,nJ-1):
        # East wall:
        i = nI-2
        # ADD CODE HERE
        # West wall:
        i = 1
        # ADD CODE HERE
    for i in range(1,nI-1):
        # North wall:
        j = nJ-2
        # ADD CODE HERE
        # South wall:
        j = 1
        # ADD CODE HERE
    
    # Copy T to outlets (where homogeneous Neumann should always be applied):
    for j in range(1,nJ-1):
        i = 1
        # ADD CODE HERE
        i = nI-2
        # ADD CODE HERE
    for i in range(1,nI-1):
        j = 1
        # ADD CODE HERE
        j = nJ-2
        # ADD CODE HERE

    # Set cornerpoint values to average of neighbouring boundary points
    T[0,0]   = # ADD CODE HERE
    T[-1,0]  = # ADD CODE HERE
    T[0,-1]  = # ADD CODE HERE
    T[-1,-1] = # ADD CODE HERE

    # Compute and print residuals (taking into account normalization):
    # Non-normalized residual:
    r0 = 0
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
           r0 += # ADD CODE HERE

    # Global convective and diffusive heat rates (divided by cp), for normalization:
    Fin = 0
    Fout = 0
    Din = 0
    Dout = 0
    for i in range(nI):
        j = nJ-2
        # ADD CODE HERE 
        j = 1
        # ADD CODE HERE
    for j in range(nJ):
        i = nI-2
        # ADD CODE HERE  
        i = 1
        # ADD CODE HERE
    F =  Fin + Din

    r = r0/F
    print('iteration: %5d, res = %.5e' % (iter, r))
            
    # Compute residuals (taking into account normalization)
    # Append residual at present iteration to list of all residuals, for plotting:
    res.append(r)    

    # Stop iterations if converged:
    if r < resTol:
        break

#================ Post-processing section ================

# Global heat rate imbalance
glob_imbal = # ADD CODE HERE
print('Global heat rate imbalance: %.2g%%' %(100 * glob_imbal))

#================ Plotting section ================
# (only examples, more plots might be needed)

if not os.path.isdir('Figures'):
    os.makedirs('Figures')

# Plot mesh
plt.figure()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Computational mesh')
plt.axis('equal')
plt.vlines(pointX[:,0],pointY[0,0],pointY[0,-1],colors = 'k',linestyles = 'dashed')
plt.hlines(pointY[0,:],pointX[0,0],pointX[-1,0],colors = 'k',linestyles = 'dashed')
plt.plot(nodeX, nodeY, 'ro')
plt.show()
plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_mesh.png')

# Plot velocity vectors
plt.figure()
plt.quiver(nodeX.T, nodeY.T, u.T, v.T)
plt.title('Velocity vectors')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.show()
plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_velocityVectors.png')

# Plot temperature contour
plt.figure()
plt.contourf(nodeX.T, nodeY.T, T.T)
tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
cbar=plt.colorbar(tempmap)
cbar.set_label('$[K]$')
plt.title('Temperature $[K]$')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
plt.tight_layout()
plt.show()
plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_temperatureDistribution.png')

# Plot heat flux vectors NORMAL TO WALL boundary face centers ONLY (not in corners)
# Use temperature gradient just inside domain (note difference to set heat flux)
qX = np.zeros((nI,nJ))*nan # Array for heat flux in x-direction, in nodes
qY = np.zeros((nI,nJ))*nan # Array for heat flux in y-direction, in nodes
for j in range(1,nJ-1):
    i = 0
    # ADD CODE HERE
    i = nI-1
    # ADD CODE HERE
for i in range(1,nI-1):
    j = 0
    # ADD CODE HERE
    j = nJ-1
    # ADD CODE HERE
plt.figure()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Wall-normal heat flux vectors\n (from internal temperature gradient)')
plt.axis('equal')
tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
cbar=plt.colorbar(tempmap)
cbar.set_label('Temperature $[K]$')
plt.quiver(nodeX, nodeY, qX, qY, color="black")
plt.xlim(-0.5*L, 3/2*L)
plt.ylim(-0.5*H, 3/2*H)
plt.tight_layout()
plt.show()
plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_wallHeatFlux.png')

# Plot residual convergence
plt.figure()
plt.title('Residual convergence')
plt.xlabel('Iterations')
plt.ylabel('Residual [-]')
resLength = np.arange(0,len(res),1)
plt.plot(resLength, res)
plt.grid()
plt.yscale('log')
plt.show()
plt.savefig('Figures/Case_'+str(caseID)+'_'+grid_type+'_residualConvergence.png')
