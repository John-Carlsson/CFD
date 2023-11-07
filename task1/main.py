# MTF073 Computational Fluid Dynamics
# Task 1: 2D diffusion
# Håkan Nilsson, 2023
# Department of Mechanics and Maritime Sciences
# Division of Fluid Dynamics
# Note that this is not efficient code. It is for educational purposes!

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
import os # For saving plots

#===================== Inputs =====================

# Geometric and mesh inputs

L = 1 # Length of the domain in X direction
H = 2 # Length of the domain in Y direction
mI = 3 # Number of mesh points X direction.
mJ = 3 # Number of mesh points Y direction.
mesh_type = 'equidistant' # Set 'equidistant' or 'non-equidistant'

# Solver inputs

nIter  =  # set maximum number of iterations
resTol =  # set convergence criteria for residuals

#====================== Code ======================

# Preparation of "nan", to fill empty slots in consistently numbered arrays.
# This makes it easier to check in Variable Explorer that values that should
# never be set are never set (or used). Plots simply omit nan values.
nan = float("nan")

# Allocate arrays (nan used to make clear where values need to be set)
# Note that some arrays could actually be 1D since they only have a variation
# in one direction, but they are kept 2D so the indexing is similar for all.
nI = mI + 1                    # Number of nodes in X direction, incl. boundaries
nJ = mJ + 1                    # Number of nodes in Y direction, incl. boundaries
pointX = np.zeros((mI,mJ))*nan # X coords of the mesh points
pointY = np.zeros((mI,mJ))*nan # Y coords of the mesh points
nodeX  = np.zeros((nI,nJ))*nan # X coords of the nodes
nodeY  = np.zeros((nI,nJ))*nan # Y coords of the nodes
dx_PE  = np.zeros((nI,nJ))*nan # X distance to east node
dx_WP  = np.zeros((nI,nJ))*nan # X distance to west node
dy_PN  = np.zeros((nI,nJ))*nan # Y distance to north node
dy_SP  = np.zeros((nI,nJ))*nan # Y distance to south node
dx_we  = np.zeros((nI,nJ))*nan # X size of the control volume
dy_sn  = np.zeros((nI,nJ))*nan # Y size of the control volume
aE     = np.zeros((nI,nJ))*nan # Array for east coefficient, in nodes
aW     = np.zeros((nI,nJ))*nan # Array for wect coefficient, in nodes
aN     = np.zeros((nI,nJ))*nan # Array for north coefficient, in nodes
aS     = np.zeros((nI,nJ))*nan # Array for south coefficient, in nodes
aP     = np.zeros((nI,nJ))*nan # Array for central coefficient, in nodes
Su     = np.zeros((nI,nJ))*nan # Array for source term for temperature, in nodes
Sp     = np.zeros((nI,nJ))*nan # Array for source term for temperature, in nodes
T      = np.zeros((nI,nJ))*nan # Array for temperature, in nodes
k      = np.zeros((nI,nJ))*nan # Array for conductivity, in nodes
k_e    = np.zeros((nI,nJ))*nan # Array for conductivity at east face
k_w    = np.zeros((nI,nJ))*nan # Array for conductivity at west face
k_n    = np.zeros((nI,nJ))*nan # Array for conductivity at north face
k_s    = np.zeros((nI,nJ))*nan # Array for conductivity at south face
res    = []                    # Array for appending residual each iteration

# Generate mesh and compute geometric variables

if mesh_type == 'equidistant':
    # Calculate mesh point coordinates:
    for i in range(0, mI):
        for j in range(0, mJ):
            pointX[i,j] = i*L/(mI - 1)
            pointY[i,j] = j*H/(mJ - 1)
elif mesh_type == 'non-equidistant':


    # ADD CODE HERE, corresponding the above but for non-equidistant mesh
    # Note that all entries of the array must be filled (no nan)
    
# Calculate node coordinates (same for equidistant and non-equidistant):
# Internal nodes:
for i in range(0, nI):
    for j in range(0, nJ):
        if i > 0 and i < nI-1:
            nodeX[i,j] = 0.5*(pointX[i,0] + pointX[i-1,0])
        if j > 0 and j < nJ-1:
            nodeY[i,j] = 0.5*(pointY[0,j] + pointY[0,j-1])
# Boundary nodes:
nodeX[0,:] = 0   # Note: corner points needed for contour plot
nodeY[:,0] = 0   # Note: corner points needed for contour plot
nodeX[-1,:] = L # Note: corner points needed for contour plot
nodeY[:,-1] = H # Note: corner points needed for contour plot

# Calculate distances
# Keep 'nan' where values are not needed!
for i in range(1, nI-1):
    for j in range(1, nJ-1):
        dx_PE[i,j] = # ADD CODE HERE
        dx_WP[i,j] = # ADD CODE HERE
        dy_PN[i,j] = # ADD CODE HERE
        dy_SP[i,j] = # ADD CODE HERE
        dx_we[i,j] = # ADD CODE HERE
        dy_sn[i,j] = # ADD CODE HERE

# Initialize dependent variable array and Dirichlet boundary conditions
# Note that a value is needed in all nodes for contour plot
# ADD CODE HERE

# The following loop is for the linear solver. In the present implementation
# it includes updates of everything that influences the linear system every
# iteration. That may not be ideal. It may be beneficial to converge (iterate)
# the linear solver somewhat before doing the updates. Students that are
# interested can investigate this matter.
for iter in range(nIter):
    
    # Update conductivity arrays k, k_e, k_w, k_n, k_s, according to your case:
    # (could be moved to before iteration loop if independent of solution,
    # but keep here if you want to easily test different cases)
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE.
        
    # Update source term array according to your case:
    # (could be moved to before iteration loop if independent of solution,
    # but keep here if you want to easily test different cases)
    # Keep 'nan' where values are not needed!
    # ADD CODE HERE.
    
    # Calculate coefficients:
    # (could be moved to before iteration loop if independent of solution)
    # Keep 'nan' where values are not needed!
    # Inner node neighbour coefficients:
    # (not caring about special treatment at boundaries):
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            aE[i,j] = # ADD CODE HERE
            aW[i,j] = # ADD CODE HERE
            aN[i,j] = # ADD CODE HERE
            aS[i,j] = # ADD CODE HERE
    # Modifications of aE and aW inside east and west boundaries:
    for j in range(1,nJ-1):
        i = nI-2 #East
        # ADD CODE HERE IF NECESSARY
        i=1 #West
        # ADD CODE HERE IF NECESSARY
    # Modifications of aN and aS inside north and south boundaries:
    for i in range(1,nI-1):
        j = nJ-2 # North
        # ADD CODE HERE IF NECESSARY
        j=1 # South
        # ADD CODE HERE IF NECESSARY

    # Inner node central coefficients:
    for i in range(1,nI-1):
        for j in range(1,nJ-1):
            aP[i,j] = # ADD CODE HERE

    # Solve for T using Gauss-Seidel:
    # ADD CODE HERE
    
    # Copy T to boundaries (and corners) where homegeneous Neumann is applied:
    # ADD CODE HERE
    
    # Compute and print residuals (taking into account normalization):
    r = # ADD CODE HERE
    print('iteration: %5d, res = %.5e' % (iter, r))
    
    # Append residual at present iteration to list of all residuals, for plotting:
    res.append(r)
    
    # Stop iterations if converged:
    if r < resTol:
        break
    
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
plt.vlines(pointX[:,0],0,H,colors = 'k',linestyles = 'dashed')
plt.hlines(pointY[0,:],0,L,colors = 'k',linestyles = 'dashed')
plt.plot(nodeX, nodeY, 'ro')
plt.tight_layout()
plt.show()
plt.savefig('Figures/mesh.png')

# Plot temperature contour
plt.figure()
plt.title('Temperature distribution')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.axis('equal')
tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
cbar=plt.colorbar(tempmap)
cbar.set_label('Temperature [K]')
plt.tight_layout()
plt.show()
plt.savefig('Figures/temperatureDistribution.png')

# Plot residual convergence
plt.figure()
plt.title('Residual convergence')
plt.xlabel('Iterations')
plt.ylabel('Residuals [-]')
resLength = np.arange(0,len(res),1)
plt.plot(resLength, res)
plt.grid()
plt.yscale('log')
plt.show()
plt.savefig('Figures/residualConvergence.png')

# Plot heat flux vectors in nodes (not at boundaries)
qX = np.zeros((nI,nJ))*nan # Array for heat flux in x-direction, in nodes
qY = np.zeros((nI,nJ))*nan # Array for heat flux in y-direction, in nodes
for i in range(1,nI-1):
    for j in range(1,nJ-1):
            qX[i,j] = # ADD CODE HERE
            qY[i,j] = # ADD CODE HERE
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Heat flux')
plt.axis('equal')
tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
plt.quiver(nodeX, nodeY, qX, qY, color="black")
cbar=plt.colorbar(tempmap)
cbar.set_label('Temperature [K]')
plt.tight_layout()
plt.show()
plt.savefig('Figures/heatFlux.png')

# Plot heat flux vectors NORMAL TO WALL boundary face centers ONLY (not in corners)
# Use temperature gradient just inside domain (note difference to set heat flux)
qX = np.zeros((nI,nJ))*nan # Array for heat flux in x-direction, in nodes
qY = np.zeros((nI,nJ))*nan # Array for heat flux in y-direction, in nodes
for j in range(1,nJ-1):
    i = 0
    qX[i,j] = # ADD CODE HERE
    qY[i,j] = # ADD CODE HERE
    i = nI-1
    qX[i,j] = # ADD CODE HERE
    qY[i,j] = # ADD CODE HERE
for i in range(1,nI-1):
    j = 0
    qX[i,j] = # ADD CODE HERE
    qY[i,j] = # ADD CODE HERE
    j = nJ-1
    qX[i,j] = # ADD CODE HERE
    qY[i,j] = # ADD CODE HERE
plt.figure()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('Wall-normal heat flux \n (from internal temperature gradient)')
plt.axis('equal')
tempmap=plt.contourf(nodeX.T,nodeY.T,T.T,cmap='coolwarm',levels=30)
cbar=plt.colorbar(tempmap)
cbar.set_label('Temperature [K]')
plt.quiver(nodeX, nodeY, qX, qY, color="black")
plt.xlim(-0.5*L, 3/2*L)
plt.ylim(-0.5*H, 3/2*H)
plt.tight_layout()
plt.show()
plt.savefig('Figures/Case_'+str(caseID)+'_wallHeatFlux.png')