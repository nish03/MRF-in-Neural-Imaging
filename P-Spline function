# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:19:36 2017

@author: Nishant
"""
import numpy as np
import scipy as sp
from copy import deepcopy
import h5py



"""import data"""
f = h5py.File("D:/Tu dresden/4th Sem/Present/After literature review/Datasets/sep_1072240.mat", "r")
img = np.array(f["img"].value)
f.close()

"""Response variable Y = 1066*307200"""
Y =  img.reshape((img.shape[0], -1), order='F')

"""delete img since covering a lot of memory"""
del img

"""define observation variable X = 1066 points"""
X = np.linspace(0,1065,1066)

"""define number of splines and order of splines"""
no_of_splines = 1065
order_of_spline = 3

"""define knots"""
knots = np.linspace(0, 1, 1 + no_of_splines - order_of_spline) #Return evenly spaced numbers over a specified interval.
difference = np.diff(knots[:2])[0]   #difference of first two knots


"""scale the values of X to be in the range of 0 & 1 by normalization"""
x = (np.ravel(deepcopy(X)) - X[0]) / float(X[-1] - X[0]) # x = np.array([[1, 2, 3], [4, 5, 6]]) print(np.ravel(x)) gives [1 2 3 4 5 6]

"""delete X since saves memory"""
del X
x = np.r_[x, 0., 1.] # append 0 and 1 in order to get derivatives for extrapolation
x = np.r_[x] #np.r_[np.array([1,2,3]), 0, 0, np.array([4,5,6])] gives array([1, 2, 3, 0, 0, 4, 5, 6])  append 0 and 1 in order to get derivatives for extrapolation
x = np.atleast_2d(x).T #View inputs as arrays with at least two dimensions
n = len(x)


"""define corner knots on left and right of original knots"""
corner = np.arange(1, order_of_spline + 1) * difference
new_knots = np.r_[-corner[::-1], knots, 1 + corner]
new_knots[-1] += 1e-9 # want last knot inclusive


""" prepare Haar Basis""" 
basis = (x >= new_knots[:-1]).astype(np.int) * (x < new_knots[1:]).astype(np.int)
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
    left = np.zeros((n, maxi))
    left[:, mask_l] = left_numerator/left_denominator
    """ right sub-basis function"""
    right_numerator = (new_knots[m : maxi+m][mask_r]-x) * basis[:, 1:maxi+1][:, mask_r]
    right_denominator = new_knots[m:maxi+m][mask_r] - new_knots[1 : maxi+1][mask_r]
    right = np.zeros((n, maxi))
    right[:, mask_r] = right_numerator/right_denominator
    """ track previous bases and update"""
    prev_bases = basis[-2:]
    basis = left + right
    

"""finally create a sparse basis matrix in compressed sparse column format"""
"""helpful in arithmetic operations, saves memory"""
basis = basis[:-2]     # get rid of the added values at 0, and 1
basis_matrix = sp.sparse.csc_matrix(basis)

"""delete unrequired variables to save memory"""
del basis,corner, difference, knots, left, left_denominator, left_numerator,  x
del m, mask_l, mask_r, maxi, n, new_knots, order_of_spline, prev_bases, right, right_denominator, right_numerator

""""""""""""""""""""""""""""""""""""""
"""Now estimate spline coefficients"""
""""""""""""""""""""""""""""""""""""""
"""n = number of observations in basis matrix, m=number of features in basis matrix"""
"""initialise a matrix 307200 which will store pixel coefficients"""
coefficients_pixels = []

"""define type of penalty to use and the value of lambda as smoothing parameter"""
lambda_param = [0.6]  
"""create and initialize penalty matrix and multiply the matrix with lambda_param"""
Ps = []
p = sp.sparse.eye(no_of_splines).tocsc()
Ps.append(p)
Penalty_matrix = tuple([np.multiply(P, lam) for lam, P in zip(lambda_param, Ps)])
Penalty_matrix = sp.sparse.block_diag(Penalty_matrix)
Penalty_matrix = Penalty_matrix.todense()
"""delete unrequired variables to save memory"""
del lam,lambda_param,Ps,p 
"""Perform cholesky decomposition on (Penalty_matrix) if being positive definite"""
""" decomposes positive-definite matrix into the product of lower triangular matrix""" 
"""and its conjugate transpose, for efficient numerical solutions, saving computation time and storage space and to increase regularisation""" 
"""regularization. Test for positive definite: gaussianwaves.com/2013/04/tests-for-positive-definiteness-of-a-matrix/ """
cholesky_matrix = np.linalg.cholesky(Penalty_matrix) #Cholesky decomposition.
del Penalty_matrix, no_of_splines
 
"""following is the iteration for each pixel to determine the coefficient
    for each pixels till 307200 pixels,link_function = 'identity', distribution = 'normal'  """
for i in range(307200):
    #initialise coefficients m =1065(splines) n=1066(time points) no_splines = 1065 min_n_m = 1065"""
    coefficients = np.ones((1065,1)) * np.sqrt(np.finfo(np.float64).eps) #is all together then coefficient will be 1065*307200
    #Initialise inverse of a diagonal matrix (D_inv). D is a Diagonal matrix with singular values"""
    #singular values are eigenvalues/eigenvectors 1<= value <= min(basis_matrix.shape) hence taking highest value which is 1064
    D_inverse = np.zeros((1064, 1064)).T
    #Define and initialise weights with ones for each response"""
    weights = np.ones_like(Y[:,i]).astype('f') #replaces values in Y[:,i] by 1. [123] becomes [111]
    #multiply the model matrix by the spline basis coefficients (bZ)
    linear_predictor = basis_matrix.dot(coefficients).flatten()
    #create weight matrix
    W = sp.sparse.diags((np.ones_like(linear_predictor)**2 * np.ones_like(linear_predictor) * weights ** -1)**-0.5)
    #pseudo data for iterations
    pseudo_data = W.dot(linear_predictor + (Y[:,i] - linear_predictor) * np.ones_like(linear_predictor))
    # common matrix product
    WB = weights.dot(basis_matrix)
    #QR decomposition
    Q, R = np.linalg.qr(WB[0].todense()) 
    #singular value decomposition
    #U, D, Vt = np.linalg.svd(np.vstack([R, cholesky_matrix.T])) , k=eigenvalues
    U, D, Vt = sp.sparse.linalg.svds(sp.sparse.vstack([sp.sparse.csc_matrix(R), sp.sparse.csc_matrix(cholesky_matrix.T)]),k=1064)
    # mask out small singular values (RANK deficiency taken care of)
    svd_mask = D <= (D.max() * np.sqrt(np.finfo(np.float64).eps))
    # invert the singular values and fill the D_inverse as D^-1
    np.fill_diagonal(D_inverse, D**-1)
    # keep only top portion of U
    U1 = U[:1065,:]
    B = Vt.T.dot(D_inverse).dot(U1.T).dot(Q.T)
    #Estimate coefficients. A means to flatten in column-major order 
    coefficients_new = B.dot(pseudo_data).A.flatten() 
    coefficients_pixels.append(coefficients_new)
    
"""delete the unrequired variables"""
del D_inverse,weights,linear_predictor,W,pseudo_data,WB,Q,R,U,D,Vt,svd_mask,U1,B,coefficients,cholesky_matrix,i,coefficients_new

"""convert coefficients_pixels from a list to numpy array"""
coefficients_pixels = np.asarray(coefficients_pixels)
        
    


