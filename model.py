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
#####################define basis matrix####################
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
spline_coeff = numpy.linalg.solve(numpy.dot(basis.T,basis)+lambda_param*numpy.dot(D_k,D_k.T),numpy.dot(basis.T,Y))
Y_hat_spline = basis.dot(spline_coeff)
s = numpy.sum((Y-Y_hat_spline)**2)
#inverse of numpy.dot(basis.T,basis)  + lambda_param * numpy.dot(D_k,D_k.T)
Q = numpy.linalg.inv(numpy.dot(basis.T,basis) + lambda_param * numpy.dot(D_k,D_k.T))
#diagonal elements  of hat matrix 
t = numpy.sum(numpy.diag(Q.dot(numpy.dot(basis.T,basis))))


###########################################################
#####################AIC, GCV, MSE##########################
###########################################################
GCV = s / (basis.shape[0] - t)**2
AIC = -2*numpy.log(numpy.exp(-(Y - Y_hat_spline)**2/(2)) / (2 * numpy.pi)**0.5 ).sum() + 2*t
MSE =  ((Y - Y_hat_spline) ** 2).mean()


#"""total time taken to run the code"""
print("--- %s seconds ---" % (time.time() - start_time))

###################################################################
###################MSE Error evaluation############################
###################################################################
#average time courses of Y_hat_spline
Y_avg_spline = numpy.zeros(1066)
for i in range(1066):
       Y_avg_spline[i] = (Y_hat_spline[i,:].mean())
	   
MSE_spline  = numpy.zeros(1066)
for i in range(1066):
       MSE_spline[i] = (Y[i,:].mean() - Y_hat_spline[i,:].mean())**2

##################################################################################################
############################Principal component analysis for #####################################
###########selecting first p (p is less than knots) and neglecting the first 'p'##################
##################################################################################################
#define covariance matrix
covariance_matrix = numpy.cov(spline_coeff)
import numpy.linalg
#Compute eigenvalues and corresponding eigenvectors from covariance matrix"""
eigen_values, eigen_vectors = numpy.linalg.eig(covariance_matrix)   
#sort the eigenvectors by decreasing eigenvalues"""
#This is done to drop lowest eigenvectors since they are less informative about data."""
# Make a list of (eigenvalue, eigenvector) tuples
eigen_pairs = [(numpy.abs(eigen_values[i]), eigen_vectors[:,i]) for i in range(len(eigen_values))]
# Sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda x: x[0], reverse=True)
#choose p eigenvectors with largest eigenvalues"""
total = sum(eigen_values)
#from explained_variance see how many p required for >96% as sum
explained_variance = [(i / total)*100 for i in sorted(eigen_values, reverse=True)]
#first two or three principal components itself normally has more than 96% information 
#check the value by explained_variance[0] >= 95% or not 

#Reducing the 20 knots to a p=10 dimensional feature subspace , 
#   by choosing the "top eigenvectors(except first one) 
#   with the highest eigenvalues to construct our 20*p eigenvector matrix """
#eigenvector_matrix = eigen_pairs[0][1].reshape(20,1)
eigenvector_matrix = numpy.hstack((eigen_pairs[0][1].reshape(20,1),eigen_pairs[1][1].reshape(20,1),eigen_pairs[2][1].reshape(20,1),eigen_pairs[3][1].reshape(20,1),eigen_pairs[4][1].reshape(20,1),eigen_pairs[5][1].reshape(20,1),eigen_pairs[6][1].reshape(20,1),eigen_pairs[7][1].reshape(20,1),eigen_pairs[8][1].reshape(20,1),eigen_pairs[9][1].reshape(20,1)))
#"""Project onto the new feature space 307200*10"""
pca_coeff = spline_coeff.T.dot(eigenvector_matrix) 
#T_cropped = numpy.absolute(T_cropped)
imgplot = plt.imshow(pca_coeff.reshape(640,480))
plt.show()

#################################################################################
#####################Discretization using K Means clustering#####################
#################################################################################
#determining the number of clusters
from sklearn.cluster import KMeans
from scipy.spatial import distance
#compute BIC
def compute_bic(kmeans,X):
    """
    Computes the BIC metric for a given clusters

    Parameters:
    -----------------------------------------
    kmeans:  List of clustering object from scikit learn

    X     :  multidimension np array of data points

    Returns:
    -----------------------------------------
    BIC value
    """
    # assign centers and labels
    centers = [kmeans.cluster_centers_]
    labels  = kmeans.labels_
    #number of clusters
    m = kmeans.n_clusters
    # size of the clusters
    n = numpy.bincount(labels)
    #size of data set
    N, d = X.shape
    #compute variance for all clusters beforehand
    cl_var = (1.0 / (N - m) / d) * sum([sum(distance.cdist(X[numpy.where(labels == i)], [centers[0][i]], 
             'euclidean')**2) for i in range(m)])
    const_term = 0.5 * m * numpy.log(N) * (d+1)
    BIC = numpy.sum([n[i] * numpy.log(n[i]) -
               n[i] * numpy.log(N) -
             ((n[i] * d) / 2) * numpy.log(2*numpy.pi*cl_var) -
             ((n[i] - 1) * d/ 2) for i in range(m)]) - const_term
    return(BIC)

ks = range(1,50)
K_means = [KMeans(n_clusters = i, init="k-means++").fit(pca_coeff) for i in ks]
BIC = [compute_bic(kmeansi,pca_coeff) for kmeansi in K_means]
print BIC


plt.plot(ks, BIC, 'bx-')
plt.xlabel('k')
plt.ylabel('BIC Value')
plt.title('BIC Method')
plt.show()

#compute AIC
from sklearn import mixture
range_n_clusters = range(1, 20)
aic_list = []
for n_clusters in range_n_clusters:
     model = mixture.GaussianMixture(n_components=n_clusters, init_params='kmeans')
     model.fit(pca_coeff)
     aic_list.append(model.aic(pca_coeff))
plt.plot(range_n_clusters, aic_list, marker='o')
plt.show()
  
#elbow method
distortions = []
K = range(1,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(pca_coeff)
    kmeanModel.fit(pca_coeff)
    distortions.append(sum(numpy.amin(distance.cdist(pca_coeff, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / pca_coeff.shape[0])


plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('Elbow Method')
plt.show()


#X-means clustering
from  pyclustering.cluster import xmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
pca_coeff_list = pca_coeff.tolist()   #xmeans takes only list so converting 2d array into list of list check by type(pca_coeff_list) and type(pca_coeff.tolist()[0])
#initialize them using K-Means++ method
initial_centers = kmeans_plusplus_initializer(pca_coeff_list, 2).initialize()  #this also has to be a list
xmeans_instance = xmeans.xmeans(pca_coeff_list, initial_centers, ccore = True)
# run cluster analysis
xmeans_instance.process()
# obtain results of clustering
clusters = xmeans_instance.get_clusters()
centers  = xmeans_instance.get_centers()
#obtain the number of clusters
len(clusters)




#finally perform kmeans clustering on choosen k
from sklearn.cluster import KMeans
from scipy.spatial import distance
kmeans = KMeans(n_clusters=16)
kmeans.fit(pca_coeff)
#learn the labels and the means
labels = kmeans.predict(pca_coeff)
means  = kmeans.cluster_centers_
imgplot = plt.imshow(labels.reshape(640,480))
plt.show()

##################################################################################
######################Start MRF using Opengm graphical model######################
##################################################################################
dimx= 640
dimy= 480
numLabels = 16
numVar=dimx*dimy
numberOfStates=numpy.ones(numVar,dtype=opengm.index_type)*numLabels
gm=opengm.graphicalModel(numberOfStates)

threshold_l1 = numpy.repeat(6,1066)
threshold_l2 = numpy.repeat(36,1066)

def fast_norm(x):
   ###   Returns a norm of a 1-D array/vector `x`.    
   ###   Turns out it's much faster than linalg.norm in case of 1-D arrays.
   ###   Measured with up to ~80% increase in speed.
   return sqrt(x.dot(x.conj()))

#inverse PCA of means
means_inv_PCA = eigenvector_matrix.dot(means.T)

#calculate unary energies
unary_energy = numpy.zeros((numVar,numLabels),dtype=numpy.float32)
t0=time.time()
for i in range(numVar):
    for l in range(numLabels):
	    unary_energy[i,l] = fast_norm(Y[:,i] - basis.dot(means_inv_PCA[:,l]))                                                                       #L2 norm
		#unary_energy[i,l] = sum(abs(Y[:,i] - basis.dot(means_inv_PCA[:,l])))                                                                       #L1 norm
		#unary_energy[i,l] = max(abs(Y[:,i] - basis.dot(means_inv_PCA[:,l])))                                                                       #max norm
		#unary_energy[i,l] = sum ( numpy.minimum ( abs(Y[:,i] - basis.dot(means_inv_PCA[:,l])) , threshold_l1 ) )                                   #trunc L1 norm
		#unary_energy[i,l] = sqrt( sum ( numpy.minimum ( (Y[:,i] - basis.dot(means_inv_PCA[:,l]))**2 , threshold_l2 ) ) )                           #trunc L2 norm
		#unary_energy[i,l]  =  sum ( abs(Y[:,i] - basis.dot(means_inv_PCA[:,l])) / ( abs(Y[:,i]) + abs(basis.dot(means_inv_PCA[:,l])) ) )           #canberra distance

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
		
t0=time.time()	    
#add binary function and factors
for x in range(dimx):
    for y in range(dimy):
        pairwise_energy = numpy.ones(numLabels*numLabels,dtype=numpy.float32).reshape(numLabels,numLabels)
        if (x + 1 < dimx):
            for l in range(numLabels):
                for k in range(numLabels):
                    pairwise_energy[l,k] = numpy.abs(l - k)
            pair_id = gm.addFunction(pairwise_energy)
            variableIndex0 = y + x * dimy
            variableIndex1 = y + (x + 1) * dimy
            gm.addFactor(pair_id, [variableIndex0, variableIndex1])
        if (y + 1 < dimy):
            for l in range(numLabels):
                for k in range(numLabels):
                    pairwise_energy[l,k] = numpy.abs(l - k)
            pair_id = gm.addFunction(pairwise_energy)	
            variableIndex0 = y + x * dimy
            variableIndex1 = (y + 1) + x * dimy
            gm.addFactor(pair_id, [variableIndex0, variableIndex1])         

			
t1=time.time()
print t1-t0 

############################################################################
#################################Inference##################################
############################################################################
imgplot=[]
class PyCallback(object):
  def appendLabelVector(self,labelVector):
     #save the labels at each iteration, to examine later.
     labelVector=labelVector.reshape(self.shape)
     imgplot.append([labelVector])
  def __init__(self,shape,numLabels):
     self.shape=shape
     self.numLabels=numLabels
     matplotlib.interactive(True)
  def checkEnergy(self,inference):
     gm=inference.gm()
     #the arg method returns the (class) labeling at each pixel.
     labelVector=inference.arg()
     #evaluate the energy of the graph given the current labeling.
     print "energy ",gm.evaluate(labelVector)
     self.appendLabelVector(labelVector)
  def begin(self,inference):
     print "beginning of inference"
     self.checkEnergy(inference)
  def end(self,inference):
     print "end of inference"
  def visit(self,inference):
     self.checkEnergy(inference)
	 
#TRWS inference
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
print "bound", inf_trws.bound()
result=argmin.reshape(dimx,dimy)
# plot final result
imgplot = plt.imshow(result)
plt.title('TRWS')
plt.show()

#BP Inference
inf_bp=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(steps=200,damping=0.5))
callback=PyCallback((640,480),numLabels)
visitor=inf_bp.pythonVisitor(callback,visitNth=1)
t0=time.time()
inf_bp.infer(visitor)
t1=time.time()
print t1-t0
# get the result states
argmin=inf_bp.arg()
#Evaluate the minimum energy
print "energy ",gm.evaluate(argmin)
#Evaluate the bound
print "bound", inf_trws.bound()
result=argmin.reshape(dimx,dimy)
# plot final result
imgplot = plt.imshow(result)
plt.title('BP')
plt.show()


##############################################################################
########################Inverse discretization################################
##############################################################################
centroid_labels = numpy.zeros((numVar,10))
centroid_labels = [means[i,:] for i in argmin]
centroid_labels = numpy.asarray(centroid_labels)

##############################################################################
######################## Inverse PCA #########################################
##############################################################################
inv_pca_coeff = eigenvector_matrix.dot(centroid_labels.T)

##############################################################################
######################## Estimating  Y_hat ###################################
##############################################################################
Y_hat_mrf = basis.dot(inv_pca_coeff)


##############################################################################
######################### MSE and time series Evaluation #####################
##############################################################################
Y_avg_MRF = numpy.zeros(1066)
for i in range(1066):
       Y_avg_MRF[i] = (Y_hat_mrf[i,:].mean())
MSE_MRF = numpy.zeros(1066)
for i in range(1066):
    MSE_MRF[i] =  (Y[i,:].mean() - Y_hat_mrf[i,:].mean()) ** 2


##############################################################################
######################## plot MSE of spline and MRF ##########################
##############################################################################
#first plot MSE before and after inference
x = numpy.linspace(0, 1065, 1066)
matplotlib.pyplot.figure(figsize=(10,8))
matplotlib.pyplot.plot(x, MSE_spline, label = 'MSE_spline', linewidth = 1.0, color='r')
matplotlib.pyplot.plot(x, MSE_MRF, label = 'MSE_MRF', linewidth =1.0 ,color='b')
matplotlib.pyplot.xlabel(r'Time points (X)',fontweight='bold',fontsize=10)
matplotlib.pyplot.ylabel(r'Mean square error',fontweight='bold', fontsize=10)
matplotlib.pyplot.legend(prop={'size': 12})
matplotlib.pyplot.title(r'Mean Square error for time points for all pixels', fontsize=15)
matplotlib.pyplot.show()

#plot averaged time courses after spline vs after MRF
x = numpy.linspace(0, 1065, 1066)
matplotlib.pyplot.figure(figsize=(10,8))
matplotlib.pyplot.plot(x, Y_avg_spline, label = 'Y_avg_spline', linewidth = 1.0, color='r')
matplotlib.pyplot.plot(x, Y_avg_MRF, label = 'Y_avg_MRF', linewidth =1.0 ,color='b')
matplotlib.pyplot.xlabel(r'Time points (X)',fontweight='bold',fontsize=10)
matplotlib.pyplot.ylabel(r'Y_estimate',fontweight='bold', fontsize=10)
matplotlib.pyplot.legend(prop={'size': 12})
matplotlib.pyplot.title(r'Y_estimate after spline vs after MRF', fontsize=15)
matplotlib.pyplot.show()
