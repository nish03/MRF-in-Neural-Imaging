#################################################################
###  Intraoperative data model for brain tumor imaging
#################################################################
###While some tumors such as meningiomas can be easily
###segmented, others like gliomas and glioblastomas are
###much more difficult to localize.These tumors are often diffused,
###poorly contrasted, and extend tentacle-like structures that make
###them difficult to segment. Another fundamental difficulty
###with segmenting these brain tumors is that they can appear
###anywhere in the brain, in almost any shape and size.
#################################################################

import numpy as numpy
from copy import deepcopy
import h5py
import opengm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot 
import matplotlib.image
import matplotlib.cm as cm
import time 


img = discretized_T_cropped.reshape(640,480)
#convert the values in the range of 0 to 1  0=black 1=white
img = numpy.asarray(img).astype(float)/256

#plot img to see the starting point of the image
imgplot = matplotlib.pyplot.imshow(img,cmap = cm.Greys_r)
matplotlib.pyplot.show()


##Now start coding for graphical model
shape = img.shape
dimx=img.shape[0]
dimy=img.shape[1]
numLabels = 5
numVar=dimx*dimy
numberOfStates=numpy.ones(numVar,dtype=opengm.index_type)*numLabels
gm=opengm.graphicalModel(numberOfStates)

#thresholds for glioma and glioblastoma sub-regions(not meningiomas because they can be easily segmented)
threshold5 = 0.796875  #(204 of 256)  #necrotic core (dead tissues)
threshold4 = 0.5859378 #(150 of 256)  #enhancing core
threshold3 = 0.410     #(105 of 256)  #non-enhancing core
threshold2 = 0.1758    #(45  of 256)  #edema (swelling)
threshold1 = 0.0156    #(4   of 256)  #normal tissue

#Adding unary function and factors
for x in range(dimx):
		for y in range(dimy):
			energy0 =  numpy.abs(img[x, y] - threshold1)
			energy1 =  numpy.abs(img[x, y] - threshold2)
			energy2 =  numpy.abs(img[x, y] - threshold3)
			energy3 =  numpy.abs(img[x, y] - threshold4)
			energy4 =  numpy.abs(img[x, y] - threshold5)
			f = numpy.array([energy0,energy1,energy2,energy3,energy4])
			#add unary function to graphical model
			fid = gm.addFunction(f)
			#add unary factor to graphical model
			gm.addFactor(fid,x*dimy+y)


#add spatial regularizer (Potts, squared distance, truncated squared distance) use one of them
beta = 0.1
#potts
f_pairwise = opengm.PottsFunction([numLabels,numLabels],0.0,beta)
#squared distance
f_pairwise=opengm.differenceFunction(shape=[numLabels,numLabels],norm=2,weight=beta,truncate=None)
#truncated squared distance
f_pairwise=opengm.TruncatedSquaredDifferenceFunction([numLabels,numLabels],10,beta)


#add function to grapical model
potts_fid = gm.addFunction(f_pairwise)

for x in range(dimx):
  for y in range(dimy):
     if (x + 1 < dimx):
       variableIndex0 = y + x * dimy
       variableIndex1 = y + (x + 1) * dimy
       gm.addFactor(potts_fid, [variableIndex0, variableIndex1])
     if (y + 1 < dimy):
       variableIndex0 = y + x * dimy
       variableIndex1 = (y + 1) + x * dimy
       gm.addFactor(potts_fid, [variableIndex0, variableIndex1])
			 

opengm.hdf5.saveGraphicalModel(gm,'model.h5','gm')

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
	 
	 
#############################################################################
#########TRWS Inference#################
#############################################################################
inf_trws=opengm.inference.TrwsExternal(gm,parameter=opengm.InfParam(steps=200))
t0=time.time()
inf_trws.infer()
t1=time.time()
print t1-t0
# get the result states
argmin=inf_trws.arg()
#Evaluate the minimum energy
print "energy ",gm.evaluate(argmin)
result=argmin.reshape(shape)
# plot final result
imgplot = matplotlib.pyplot.imshow(result)
matplotlib.pyplot.title('TRWS')
matplotlib.pyplot.show()


#############################################################################
#########Belief propagation inference#################
#############################################################################
inf_bp=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(steps=200,damping=0.5))
callback=PyCallback(shape,numLabels)
visitor=inf_bp.pythonVisitor(callback,visitNth=1)
t0=time.time()
inf_bp.infer(visitor)
t1=time.time()
print t1-t0
# get the result states
argmin=inf_bp.arg()
#Evaluate the minimum energy
print "energy ",gm.evaluate(argmin)
result=argmin.reshape(shape)
# plot final result
imgplot = matplotlib.pyplot.imshow(result)
matplotlib.pyplot.title('BP')
matplotlib.pyplot.show()

