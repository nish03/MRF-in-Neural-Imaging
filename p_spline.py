# -*- coding: utf-8 -*-
"""
Created on Fri Oct 06 09:23:16 2017

@author: aa
"""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""import python libraries"""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as numpy
from copy import deepcopy
import h5py
import matplotlib.pyplot 

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""load   data"""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
f = h5py.File("sep_1072240.mat", "r")
img = numpy.array(f["img"].value)
f.close()

"""Response variable Y = 1066*307200"""
#Y =  img.reshape((img.shape[0], -1), order='F')
Y =  img.reshape((img.shape[0], -1))
"""delete img since covering a lot of memory"""
del img


"""to calculate run time"""
import time
start_time = time.time()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""define basis matrix""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""define observation variable X = 1066 points"""
X = numpy.linspace(0,1065,1066)

"""define number of splines and order of splines"""
no_of_splines = 200
order_of_spline = 3

"""define knots"""
knots = numpy.linspace(0, 1, 1 + no_of_splines - order_of_spline) #Return evenly spaced numbers over a specified interval.
difference = numpy.diff(knots[:2])[0]   #difference of first two knots


"""scale the values of X to be in the range of 0 & 1 by normalization"""
x = (numpy.ravel(deepcopy(X)) - X[0]) / float(X[-1] - X[0]) # x = numpy.array([[1, 2, 3], [4, 5, 6]]) print(numpy.ravel(x)) gives [1 2 3 4 5 6]

"""delete X since saves memory"""
del X
x = numpy.r_[x, 0., 1.] # append 0 and 1 in order to get derivatives for extrapolation
x = numpy.r_[x] #numpy.r_[numpy.array([1,2,3]), 0, 0, numpy.array([4,5,6])] gives array([1, 2, 3, 0, 0, 4, 5, 6])  append 0 and 1 in order to get derivatives for extrapolation
x = numpy.atleast_2d(x).T #View inputs as arrays with at least two dimensions
n = len(x)


"""define corner knots on left and right of original knots"""
corner = numpy.arange(1, order_of_spline + 1) * difference
new_knots = numpy.r_[-corner[::-1], knots, 1 + corner]
new_knots[-1] += 1e-9 # want last knot inclusive


""" prepare Haar Basis""" 
basis = (x >= new_knots[:-1]).astype(numpy.int) * (x < new_knots[1:]).astype(numpy.int)
basis[-1] = basis[-2][::-1] # force symmetric bases at 0 and 1


""" De-boor recursion"""
maxi = len(new_knots) - 1
for m in range(2, order_of_spline + 2):
    maxi -= 1
    """ Avoid division by 0 """
    mask_l = new_knots[m - 1 : maxi + m - 1] != new_knots[:maxi]
    mask_r = new_knots[m : maxi + m] != new_knots[1 : maxi + 1]
    """ left sub-basis function"""
    left_numerator = (x - new_knots[:maxi][mask_l]) * basis[:, :maxi][:, mask_l]
    left_denominator = new_knots[m-1 : maxi+m-1][mask_l] - new_knots[:maxi][mask_l]
    left = numpy.zeros((n, maxi))
    left[:, mask_l] = left_numerator/left_denominator
    """ right sub-basis function"""
    right_numerator = (new_knots[m : maxi+m][mask_r]-x) * basis[:, 1:maxi+1][:, mask_r]
    right_denominator = new_knots[m:maxi+m][mask_r] - new_knots[1 : maxi+1][mask_r]
    right = numpy.zeros((n, maxi))
    right[:, mask_r] = right_numerator/right_denominator
    """ track previous bases and update"""
    prev_bases = basis[-2:]
    basis = left + right
    

"""finally create a sparse basis matrix in compressed sparse column format"""
"""helpful in arithmetic operations, saves memory"""
basis = basis[:-2]     # get rid of the added values at 0, and 1


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""Penalise the spline"""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
D = numpy.identity(basis.shape[1])
#matrix representation of second order difference operator 
D_k = numpy.diff(D,n=2,axis=-1)  
#define smoothing parameter
lambda_param = 1
#estimate the coefficients
a = numpy.linalg.solve(numpy.dot(basis.T,basis)  + lambda_param * numpy.dot(D_k,D_k.T), numpy.dot(basis.T, Y))
#estimate Y =  Y_hat
Y_hat = basis.dot(a)
s = numpy.sum((Y-Y_hat)**2)
#inverse of numpy.dot(basis.T,basis)  + lambda_param * numpy.dot(D_k,D_k.T)
Q = numpy.linalg.inv(numpy.dot(basis.T,basis) + lambda_param * numpy.dot(D_k,D_k.T))
#diagonal elements  of hat matrix 
t = numpy.sum(numpy.diag(Q.dot(numpy.dot(basis.T,basis))))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""Performance using GCV, AIC and MSE"""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
GCV = s / (basis.shape[0] - t)**2
AIC = -2*numpy.log(numpy.exp(-(Y - Y_hat)**2/(2)) / (2 * numpy.pi)**0.5 ).sum() + 2*t
MSE =  ((Y - Y_hat) ** 2).mean()


"""total time taken to run the code"""
print("--- %s seconds ---" % (time.time() - start_time))


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""Verification of fitting of p-splines using plots"""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

x = numpy.linspace(0, 1065, 1066)
matplotlib.pyplot.figure(figsize=(10,8))
matplotlib.pyplot.scatter(x, Y[:,0], label = 'Original Y', marker = 'o', s=10)
matplotlib.pyplot.plot(numpy.delete(x,[0,1065]), numpy.delete(Y_hat[:,0],[0,1065]), label = 'Estimate Y', linewidth =1.0 ,color='r')
matplotlib.pyplot.xlabel(r'Time points (X)',fontweight='bold',fontsize=10)
matplotlib.pyplot.ylabel(r'Response variable (Y)',fontweight='bold', fontsize=10)
matplotlib.pyplot.legend(prop={'size': 12})
matplotlib.pyplot.title(r'$\lambda = 1$, # knots = 200', fontsize=15)
matplotlib.pyplot.show()


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""principal component analysis"""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#define covariance matrix
covariance_matrix = numpy.cov(a)
import numpy.linalg

"""Compute eigenvalues and corresponding eigenvectors from covariance matrix"""
eigen_values, eigen_vectors = numpy.linalg.eig(covariance_matrix)   

"""sort the eigenvectors by decreasing eigenvalues"""
"""This is done to drop lowest eigenvectors since they are less informative 
   about data."""
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(numpy.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda x: x[0], reverse=True)


"""choose p eigenvectors with largest eigenvalues"""
total = sum(eigen_values)
#from explained_variance see how many p required for >96% as sum
explained_variance = [(i / total)*100 for i in sorted(eigen_values, reverse=True)]
#first two or three principal components itself normally has more than 96% information 
#check the value by explained_variance[0] >= 95% or not 

"""Reducing the 200 features to a p dimensional feature subspace , 
   by choosing the "top eigenvectors which represent 95% of info 
   with the highest eigenvalues to construct our 200*p eigenvector matrix """
eigenvector_matrix = eigen_pairs[0][1].reshape(200,1)

"""Project onto the new feature space"""
T_cropped = a.T.dot(eigenvector_matrix) 
T_cropped = numpy.absolute(T_cropped)

"""Discretization of T_cropped"""
bins = numpy.linspace(numpy.amin(T_cropped),numpy.amax(T_cropped),255) 
discretized_T_cropped = numpy.digitize(T_cropped, bins) 
