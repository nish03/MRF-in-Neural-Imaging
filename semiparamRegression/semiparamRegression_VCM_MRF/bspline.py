###########################################################
#####################define basis matrix####################
###########################################################
import numpy
from copy import deepcopy

def createBasis(noKnots = 40,orderSpline = 3,lengthTimeseries = 1024):
    #define observation variable X = 1024 points
    X = numpy.linspace(0,lengthTimeseries-1,lengthTimeseries)
    #define number of splines and order of splines
    #define knots
    knots = numpy.linspace(0, 1, 1 + noKnots - orderSpline) #Return evenly spaced numbers over a specified interval.
    difference = numpy.diff(knots[:2])[0]   #difference of first two knots
    #scale the values of X to be in the range of 0 & 1 by normalization
    x = (numpy.ravel(deepcopy(X)) - X[0]) / float(X[-1] - X[0]) # x = numpy.array([[1, 2, 3], [4, 5, 6]]) print(numpy.ravel(x)) gives [1 2 3 4 5 6]
    x = numpy.r_[x, 0., 1.] # append 0 and 1 in order to get derivatives for extrapolation
    x = numpy.r_[x] #numpy.r_[numpy.array([1,2,3]), 0, 0, numpy.array([4,5,6])] gives array([1, 2, 3, 0, 0, 4, 5, 6])  append 0 and 1 in order to get derivatives for extrapolation
    x = numpy.atleast_2d(x).T #View inputs as arrays with at least two dimensions
    n = len(x)
    #define corner knots on left and right of original knots
    corner = numpy.arange(1, orderSpline + 1) * difference
    new_knots = numpy.r_[-corner[::-1], knots, 1 + corner]
    new_knots[-1] += 1e-9 # want last knot inclusive
    #prepare Haar Basis"""
    basis = (x >= new_knots[:-1]).astype(numpy.int) * (x < new_knots[1:]).astype(numpy.int)
    basis[-1] = basis[-2][::-1] # force symmetric bases at 0 and 1
    #De-boor recursion
    maxi = len(new_knots) - 1
    for m in range(2, orderSpline + 2):
        maxi -= 1
        #Avoid division by 0
        mask_l = new_knots[m - 1 : maxi + m - 1] != new_knots[:maxi]
        mask_r = new_knots[m : maxi + m] != new_knots[1 : maxi + 1]
        #left sub-basis function
        left_numerator = (x - new_knots[:maxi][mask_l]) * basis[:, :maxi][:, mask_l]
        left_denominator = new_knots[m-1 : maxi+m-1][mask_l] - new_knots[:maxi][mask_l]
        left = numpy.zeros((n, maxi))
        left[:, mask_l] = left_numerator/left_denominator
        #right sub-basis function
        right_numerator = (new_knots[m : maxi+m][mask_r]-x) * basis[:, 1:maxi+1][:, mask_r]
        right_denominator = new_knots[m:maxi+m][mask_r] - new_knots[1 : maxi+1][mask_r]
        right = numpy.zeros((n, maxi))
        right[:, mask_r] = right_numerator/right_denominator
        #track previous bases and update
        #prev_bases = basis[-2:]
        basis = left + right


    #finally create a sparse basis matrix in compressed sparse column format
    #helpful in arithmetic operations and saves memory"""
    basis = basis[:-2]     # get rid of the added values at 0, and 1
    return basis
