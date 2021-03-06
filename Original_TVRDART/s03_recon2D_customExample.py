#-----------------------------------------------------------------------
#Copyright 2016 Centrum Wiskunde & Informatica, Amsterdam
#National Research Institute for Mathematics and Computer Science in the Netherlands
#Author: Dr. Xiaodong ZHUGE
#Contact: x.zhuge@cwi.nl/zhugexd@hotmail.com
#
#
#This file is part of the Python implementation of TVR-DART algorithm (Total Variation 
#Regularized Discrete Algebraic Reconstruction Technique), a robust and automated
#reconsturction algorithm for performing discrete tomography
#
#References: 
# [1] X. Zhuge, W.J. Palenstijn, K.J. Batenburg, "TVR-DART: 
# A More Robust Algorithm for Discrete Tomography From Limited Projection Data 
# With Automated Gray Value Estimation," IEEE Transactions on Imaging Processing, 
# 2016, vol. 25, issue 1, pp. 455-468
# [2] X. Zhuge, H. Jinnai, R.E. Dunin-Borkowski, V. Migunov, S. Bals, P. Cool,
# A.J. Bons, K.J. Batenburg, "Automated discrete electron tomography - Towards
# routine high-fidelity reconstruction of nanomaterials," Ultramicroscopy 2016
#
#This Python implementaton of TVR-DART is a free software: you can use 
#it and/or redistribute it under the terms of the GNU General Public License as 
#published by the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This Python implementaton is distributed in the hope that it will 
#be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#the GNU General Public License can be found at
#<http://www.gnu.org/licenses/>.
#
#------------------------------------------------------------------------------
# 
# 2D Example script of applying TVR-DART to obtain two-dimensional discrete 
# tomographic reconstruction using a single slice of a Lanthanide-based 
# inorganic nanotube dataset recorded using a direct electron detector
# Reference to the data:
# V. Migunov, H. Ryll, X. Zhuge, M. Simson, L. Struder, K. J. Batenburg, 
# L. Houben, R. E. Dunin-Borkowski, Rapid low dose electron tomography using 
# a direct electron detection camera, Scientific Reports, 5:14516, 
# 2015. doi: 10.1038/srep14516
# 
#------------------------------------------------------------------------------
import TVRDART
import astra
import numpy as np

###Extra added
import tifffile
import pyqtgraph as pq

# Read data (-log has been performed beforehand)
Custom_Im = tifffile.imread('1Nx128Nchan3Nclass3.tiff')
data = np.load('nanotube2d.npy')
pq.image(Custom_Im)
#pq.image(data)
#input()

################################
##### Make projection data #####
################################
NAngles = 25
M = Custom_Im.shape[0]
N = Custom_Im.shape[1]
energies = 1
FixedAttenuations = [[0, 0.2, 0.5, 0.8]]#, [0, 0.5, 0.7, 0.9]]
materials = 4

Angles = np.linspace(0,np.pi,NAngles,False)
    
vol_geom = astra.create_vol_geom(M, N)
proj_geom = astra.create_proj_geom('parallel', 1, M, Angles)
proj_id = astra.create_projector('cuda', proj_geom, vol_geom)



projData = np.zeros((energies, NAngles, M)) #Contains the (projection) data

def createPhantomAtEnergy(M, N, SegArray, FA):
    AttProf = np.zeros((M, N))
    for i in range(0,materials):
        ConcProfile = np.zeros((M, N))
        ConcProfile[SegArray == i] = 1
        AttProf += ConcProfile*FA[i]
    return AttProf

#Create phantom at energy level
for e in range(0, energies):
    P = createPhantomAtEnergy(M, N, Custom_Im, FixedAttenuations[e])
    ###pq.image(P)
    
           
    #Compute forward projection and store the "data"
    sinogram_id, sinogram = astra.create_sino(P, proj_id)
    #Optional: add poisson noise to the projection data
    #if(addNoise == True):
    #    print("Adding noise...")
    #    sinogram = astra.add_noise_to_sino(sinogram, noiseInt)
    projData[e,:,:] = sinogram

pq.image(projData)
#input()
################################

Allrecsirt = np.zeros((energies, M, N))

for e in range(0, energies):

    data = projData[e]

    [Nan,Ndetx] = data.shape
    #angles = np.linspace(-50,50,Nan,True) * (np.pi/180)

    # Intensity-offset correction
    #print('Intensity-offset correction...')
    #offset = -0.00893
    #data -= offset

    # Setting reconstruction geometry
    print('Configure projection and volume geometry...')
    Nx = Ndetx
    Nz = Ndetx
    # create projection geometry and operator
    proj_geom = astra.create_proj_geom('parallel', 1.0, Ndetx, Angles)
    vol_geom = astra.create_vol_geom(Nz,Nx)
    proj_id = astra.create_projector('cuda',proj_geom,vol_geom)
    W = astra.OpTomo(proj_id)

    # Configuration of TVR-DART parameters
    Ngv = 4 # number of material composition in the specimen (including vacuum)
    K = 4*np.ones(Ngv-1) # sharpness of soft segmentation function
    lamb = 10 # weight of TV
    Niter = 50 # number of iterations

    # Initial reconstruction and normalization
    print('Initial reconstruction...')
    import SIRT
    recsirt = SIRT.recon(data, 50, proj_geom, vol_geom, 'cuda')
    pq.image(recsirt)
    print(np.max(recsirt))
    input()
    
    sf = np.max(recsirt)
    sf = 1.0
    data = data/sf
    p = data.reshape(Nan*Ndetx)
    recsirt = recsirt/sf

    Allrecsirt[e,:,:] = recsirt

pq.image(Allrecsirt)
input()


    
# Automatic parameter estimation
print('Parameter estimation...')
#gvMC = [np.linspace(0, 1, Ngv,True) for e in range(0,energies)]
gvMC = [np.linspace(0, 1, Ngv,True) for e in range(0,energies)]
print("gvMC::", gvMC)
#gv = [0, 0.2, 0.5, 0.8]
#param0MC = [MC_TVRDART.gv2param(gvMC[e],K) for e in range(0,energies)]          #### MAKEN ALS EEN GEFLATTENDE LIJST/ARRAY EN VERDER GOED SLICEN PER CHANNEL!!! 

param0MC = np.empty((Ngv-1+len(K))*energies)
for e in range(0,energies):
    start = e*(Ngv-1+len(K))
    stop = (e+1)*(Ngv-1+len(K))
    param0MC[start:stop] = TVRDART.gv2param(gvMC[e],K)

### param0MC = [MC_TVRDART.gv2param(gvMC[e],K) for e in range(0,energies)]

print(param0MC)
print(param0MC.shape)
input()
print("Start values", gvMC, param0MC)
Segrec,param_esti = TVRDART.joint(W, p, Allrecsirt[0,:,:], param0MC, lamb)
#Segrec,param_esti = TVRDART.joint(W, p, recsirt, param0 ,lamb)
#print("Estemiate??", gv)
[gv,K] = TVRDART.param2gv(param_esti)

#raise Exception('pause')

print("ESTIMATION??", param_esti)

# Reconstruction with estimated parameters
print('Reconstruction with estimated parameters...')
Segrec,rec = TVRDART.recon(W,p, recsirt, param_esti, lamb, Niter)
gv = gv*sf
recsirt = recsirt*sf
Segrec = Segrec*sf;

print("VALUES", sf, gv)
pq.image(recsirt)
pq.image(Segrec)
input()

#-----------------------------------------------------------------------------
# Plots
import pylab
pylab.gray()
pylab.figure(1)
pylab.imshow(recsirt)
pylab.colorbar()
pylab.title('SIRT')

pylab.figure(2)
pylab.imshow(Segrec)
pylab.colorbar()
pylab.title('TVR-DART')

# Save results
print('Saving results...')
np.save('TVRDART2Dreconstruction_CustomIm.npy',Segrec)
pq.image(P)
pq.image(Segrec)
input()
