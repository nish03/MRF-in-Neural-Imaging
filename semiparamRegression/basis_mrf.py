import numpy
from copy import deepcopy

def basis_mrf(num_knots,noTimepoints):
    X = numpy.linspace(0, noTimepoints - 1, noTimepoints)
    no_of_splines = num_knots
    order_of_spline = 3
    knots = numpy.linspace(0, 1, 1 + no_of_splines - order_of_spline)
    difference = numpy.diff(knots[:2])[0]
    x = (numpy.ravel(deepcopy(X)) - X[0]) / float(X[-1] - X[0])
    x = numpy.r_[x, 0., 1.]
    x = numpy.r_[x]
    x = numpy.atleast_2d(x).T
    n = len(x)
    corner = numpy.arange(1, order_of_spline + 1) * difference
    new_knots = numpy.r_[-corner[::-1], knots, 1 + corner]
    new_knots[-1] += 1e-9
    basis = (x >= new_knots[:-1]).astype(numpy.int) * (x < new_knots[1:]).astype(numpy.int)
    basis[-1] = basis[-2][::-1]
    maxi = len(new_knots) - 1
    for m in range(2, order_of_spline + 2):
        maxi -= 1
        mask_l = new_knots[m - 1 : maxi + m - 1] != new_knots[:maxi]
        mask_r = new_knots[m : maxi + m] != new_knots[1 : maxi + 1]
        left_numerator = (x - new_knots[:maxi][mask_l]) * basis[:, :maxi][:, mask_l]
        left_denominator = new_knots[m-1 : maxi+m-1][mask_l] - new_knots[:maxi][mask_l]
        left = numpy.zeros((n, maxi))
        left[:, mask_l] = left_numerator/left_denominator
        right_numerator = (new_knots[m : maxi+m][mask_r]-x) * basis[:, 1:maxi+1][:, mask_r]
        right_denominator = new_knots[m:maxi+m][mask_r] - new_knots[1 : maxi+1][mask_r]
        right = numpy.zeros((n, maxi))
        right[:, mask_r] = right_numerator/right_denominator
        prev_bases = basis[-2:]
        basis = left + right
    
    basis = basis[:-2] 
    return basis