###########################################################
            #"""import python libraries"""
###########################################################
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as numpy
from copy import deepcopy
import h5py
import opengm
import time 
from math import sqrt
import networkx as nx
import graphviz
from networkx.drawing.nx_agraph import graphviz_layout
###########################################################
#####################load   data##########################
###########################################################
f = h5py.File("sep_1072240.mat", "r")
Y = numpy.array(f["img"].value)
f.close()

#"""Response variable Y = 1066*307200"""
#Y =  img.reshape((img.shape[0], -1), order='F')
Y =  Y.reshape((Y.shape[0], -1))

#chech how a single image looks like
imgplot = plt.imshow(Y[0,:].reshape(640,480))
plt.show()

###########################################################
#####################define basis marix####################
###########################################################
#"""define observation variable X = 1066 points"""
X = numpy.linspace(0,1065,1066)

#"""define number of splines and order of splines"""
no_of_splines = 20
order_of_spline = 3

#"""define knots"""
knots = numpy.linspace(0, 1, 1 + no_of_splines - order_of_spline) #Return evenly spaced numbers over a specified interval.
difference = numpy.diff(knots[:2])[0]   #difference of first two knots


#"""scale the values of X to be in the range of 0 & 1 by normalization"""
x = (numpy.ravel(deepcopy(X)) - X[0]) / float(X[-1] - X[0]) # x = numpy.array([[1, 2, 3], [4, 5, 6]]) print(numpy.ravel(x)) gives [1 2 3 4 5 6]

#"""delete X since saves memory"""
del X
x = numpy.r_[x, 0., 1.] # append 0 and 1 in order to get derivatives for extrapolation
x = numpy.r_[x] #numpy.r_[numpy.array([1,2,3]), 0, 0, numpy.array([4,5,6])] gives array([1, 2, 3, 0, 0, 4, 5, 6])  append 0 and 1 in order to get derivatives for extrapolation
x = numpy.atleast_2d(x).T #View inputs as arrays with at least two dimensions
n = len(x)


#"""define corner knots on left and right of original knots"""
corner = numpy.arange(1, order_of_spline + 1) * difference
new_knots = numpy.r_[-corner[::-1], knots, 1 + corner]
new_knots[-1] += 1e-9 # want last knot inclusive


#""" prepare Haar Basis""" 
basis = (x >= new_knots[:-1]).astype(numpy.int) * (x < new_knots[1:]).astype(numpy.int)
basis[-1] = basis[-2][::-1] # force symmetric bases at 0 and 1


#""" De-boor recursion"""
maxi = len(new_knots) - 1
for m in range(2, order_of_spline + 2):
    maxi -= 1
    #""" Avoid division by 0 """
    mask_l = new_knots[m - 1 : maxi + m - 1] != new_knots[:maxi]
    mask_r = new_knots[m : maxi + m] != new_knots[1 : maxi + 1]
    #""" left sub-basis function"""
    left_numerator = (x - new_knots[:maxi][mask_l]) * basis[:, :maxi][:, mask_l]
    left_denominator = new_knots[m-1 : maxi+m-1][mask_l] - new_knots[:maxi][mask_l]
    left = numpy.zeros((n, maxi))
    left[:, mask_l] = left_numerator/left_denominator
    #""" right sub-basis function"""
    right_numerator = (new_knots[m : maxi+m][mask_r]-x) * basis[:, 1:maxi+1][:, mask_r]
    right_denominator = new_knots[m:maxi+m][mask_r] - new_knots[1 : maxi+1][mask_r]
    right = numpy.zeros((n, maxi))
    right[:, mask_r] = right_numerator/right_denominator
    #""" track previous bases and update"""
    prev_bases = basis[-2:]
    basis = left + right
    

#"""finally create a sparse basis matrix in compressed sparse column format"""
#"""helpful in arithmetic operations, saves memory"""
basis = basis[:-2]     # get rid of the added values at 0, and 1


###########################################################
#####################penalise spline#######################
###########################################################
D = numpy.identity(basis.shape[1])
#matrix representation of second order difference operator 
D_k = numpy.diff(D,n=2,axis=-1)  
#define smoothing parameter
lambda_param = 0.0001
#estimate the coefficients
a = numpy.linalg.solve(numpy.dot(basis.T,basis)+lambda_param*numpy.dot(D_k,D_k.T),numpy.dot(basis.T,Y))



###########################################################
##############build graphical model########################
###########################################################
dimx= 640
dimy= 480
numLabels = 10
numVar=dimx*dimy
numberOfStates=numpy.ones(numVar,dtype=opengm.index_type)*numLabels
gm=opengm.graphicalModel(numberOfStates)
sigma = numpy.full((no_of_splines,),0.1) #offset

#add the offset in the coefficients
a_offset = numpy.zeros((no_of_splines,numVar,numLabels))
t0=time.time()
for i in range(numVar):
    for l in range(numLabels):
	    a_offset[:,i,l] = a[:,i] + (l-4.5)*sigma

t1=time.time()
print t1-t0

def fast_norm(x):
   ###Returns a norm of a 1-D array/vector `x`.    
   ###   Turns out it's much faster than linalg.norm in case of 1-D arrays.
   ###   Measured with up to ~80% increase in speed.
   return sqrt(x.dot(x.conj()))
   
   
#calculate unary energies for each pixel	 
unary_energy = numpy.zeros((numVar,numLabels))

t0=time.time()
for i in range(numVar):
    for l in range(numLabels):
	    unary_energy[i,l] = fast_norm(Y[:,i] - basis.dot(a_offset[:,i,l])) / sqrt(1066) 

t1=time.time()
print t1-t0

#reshape unary energies
unary_energy = unary_energy.reshape(dimx,dimy,numLabels)

#add unary function and factors
for x in range(dimx):
	for y in range(dimy):
	    #add unary function to graphical model
		fid=gm.addFunction(unary_energy[x,y,:])
		#add unary factor to the graphical model
		gm.addFactor(fid,x*dimy+y)
		
#reshape a_offset
a_offset = a_offset.reshape(no_of_splines,dimx,dimy,numLabels)


t0=time.time()	 
#add binary function and factors
for x in range(dimx):
    for y in range(dimy):
        pairwise_energy = numpy.ones(numLabels*numLabels,dtype=numpy.float32).reshape(numLabels,numLabels)
        if (x + 1 < dimx):
            for l in range(numLabels):
                for k in range(numLabels):
                    pairwise_energy[l,k] = fast_norm(a_offset[:,x,y,l] - a_offset[:,x+1,y,k]) / sqrt(no_of_splines)
            pair_id = gm.addFunction(pairwise_energy)
            variableIndex0 = y + x * dimy
            variableIndex1 = y + (x + 1) * dimy
            gm.addFactor(pair_id, [variableIndex0, variableIndex1])
        if (y + 1 < dimy):
            for l in range(numLabels):
                for k in range(numLabels):
                    pairwise_energy[l,k] = fast_norm(a_offset[:,x,y,l] - a_offset[:,x,y+1,k]) / sqrt(no_of_splines)
            pair_id = gm.addFunction(pairwise_energy)	
            variableIndex0 = y + x * dimy
            variableIndex1 = (y + 1) + x * dimy
            gm.addFactor(pair_id, [variableIndex0, variableIndex1])         

			
t1=time.time()
print t1-t0 

#save the dataset
opengm.hdf5.saveGraphicalModel(gm,'model.h5','gm')
#load the dataset
opengm.hdf5.loadGraphicalModel(gm,'model.h5','gm')

#############################################################################
#########TRWS Inference#################
#############################################################################
inf_trws=opengm.inference.TrwsExternal(gm)
t0=time.time()
inf_trws.infer()
t1=time.time()
print t1-t0
# get the result states
argmin=inf_trws.arg()
#Evaluate the minimum energy
print "energy ",gm.evaluate(argmin)
#Evaluate the bound
bound = inf_trws.bound()
result=argmin.reshape(dimx,dimy)
# plot final result
imgplot = plt.imshow(result)
plt.title('TRWS')
plt.show()


##############################################################################
####### coefficients after optimal labelling from inference
##############################################################################
a_inf = numpy.zeros((no_of_splines,numVar))
for i in range(numVar):
	a_inf[:,i] = a[:,i] + (argmin[i]-4.5)*sigma
	
mse_spline = numpy.zeros((1066,))
mse_inf    = numpy.zeros((1066,))	
for i,j in range(numVar,1066):
    mse_spline[j] = ((Y[:,i] - basis.dot(a[:,i])) ** 2).mean()
    mse_inf[j]    = ((Y[:,i] - basis.dot(a_inf[:,i])) ** 2).mean()
	
x = numpy.linspace(0, 1065, 1066)
matplotlib.pyplot.figure(figsize=(10,8))
matplotlib.pyplot.plot(x, mse_spline, label = 'MSE (Spline)', linewidth =1.0 ,color='r')
matplotlib.pyplot.plot(x, mse_inf,    label = 'MSE (After Inference)', linewidth =1.0 ,color='b')
matplotlib.pyplot.xlabel(r'Time points (X)',fontweight='bold',fontsize=10)
matplotlib.pyplot.ylabel(r'Mean Square Error (MSE)',fontweight='bold', fontsize=10)
matplotlib.pyplot.legend(prop={'size': 12})
matplotlib.pyplot.title(r'Univariate Regression vs Spatial Regularizer (Offset = 0.11)', fontsize=15)
matplotlib.pyplot.show()

