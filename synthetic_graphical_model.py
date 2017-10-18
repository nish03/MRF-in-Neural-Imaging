###################################################################
######## MRF Inpainting
###################################################################
###Image inpainting is a restoration task where given a
###noisy input image with missing pixels in certain regions, 
###the goal is to denoise the image and fill in missing pixel values
###################################################################

import numpy
import opengm
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot 
import matplotlib.image
import matplotlib.cm as cm
import time

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
        
norm      = 2
weight    = 5.0
numLabels = 50   # use 256 for full-model (slow)
# Read image
img   = numpy.array(numpy.squeeze(matplotlib.image.imread('Penguin-input.png')),dtype=opengm.value_type)
shape = img.shape
inpaintPixels=numpy.where(img==0)
# normalize and flatten image
iMin    = numpy.min(img)
iMax    = numpy.max(img)
imgNorm = ((img[:,:]-iMin)/(iMax-iMin))*float(numLabels)
imgFlat = imgNorm.reshape(-1).astype(numpy.uint64)
# Set up Grapical Model:
numVar = int(img.size)
gm = opengm.gm([numLabels]*numVar,operator='adder')
gm.reserveFunctions(numLabels,'explicit')
numberOfPairwiseFactors=shape[0]*(shape[1]-1) + shape[1]*(shape[0]-1)
gm.reserveFactors(numVar-len(inpaintPixels[0]) + numberOfPairwiseFactors)
	
# Set up unaries:
# - create a range of all possible labels
allPossiblePixelValues=numpy.arange(numLabels)
pixelValueRep    = numpy.repeat(allPossiblePixelValues[:,numpy.newaxis],numLabels,1)
# - repeat [0,1,2,3,...,253,254,255] numVar times
labelRange = numpy.arange(numLabels,dtype=opengm.value_type)
labelRange = numpy.repeat(labelRange[numpy.newaxis,:], numLabels, 0)
unaries = numpy.abs(pixelValueRep - labelRange)**norm
# - add unaries to the graphical model
fids=gm.addFunctions(unaries.astype(opengm.value_type))
# add unary factors to graphical model
if inpaintPixels is None:
    for l in xrange(numLabels):
        whereL = numpy.where(imgFlat == l)
        gm.addFactors(fids[l], whereL[0].astype(opengm.index_type))
else:
        # get vis of inpaint pixels
    ipX = inpaintPixels[0]
    ipY = inpaintPixels[1]
    ipVi = ipX * shape[1] + ipY
    for l in xrange(numLabels):
        whereL = numpy.where(imgFlat == l)
        notInInpaint = numpy.setdiff1d(whereL[0], ipVi)
        gm.addFactors(fids[l], notInInpaint.astype(opengm.index_type))
			
# add one of the second order function
#try squared difference function
f=opengm.differenceFunction(shape=[numLabels,numLabels],norm=2,weight=weight,truncate=None)
#try potts function
f=opengm.PottsFunction([numLabels,numLabels],0.0,5.0)
#try truncated squared difference function
f=opengm.TruncatedSquaredDifferenceFunction([numLabels,numLabels],weight,40.0)

fid=gm.addFunction(f)
vis2Order=opengm.secondOrderGridVis(shape[0],shape[1],True)
# add all second order factors
gm.addFactors(fid,vis2Order)


#############################################################################
#########TRWS Inference#################
#############################################################################
inf_trws=opengm.inference.TrwsExternal(gm,parameter=opengm.InfParam(steps=60))
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
matplotlib.pyplot.show()
matplotlib.interactive(False)
# Two subplots, the axes array is 1-d
f, axarr = matplotlib.pyplot.subplots(1,2)
axarr[0].imshow(img, cmap = cm.Greys_r)
axarr[0].set_title('Input Image')
axarr[1].imshow(result, cmap = cm.Greys_r)
axarr[1].set_title('Solution')
matplotlib.pyplot.show()


#############################################################################
#########Belief propagation inference#################
#############################################################################
inf_bp=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(steps=60))
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
matplotlib.pyplot.show()
matplotlib.interactive(False)
# Two subplots, the axes array is 1-d
f, axarr = matplotlib.pyplot.subplots(1,2)
axarr[0].imshow(img, cmap = cm.Greys_r)
axarr[0].set_title('Input Image')
axarr[1].imshow(result, cmap = cm.Greys_r)
axarr[1].set_title('Solution')
matplotlib.pyplot.show()

