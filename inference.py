import opengm
import matplotlib.pyplot
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
	 
###########################################
#Inference with BeliefPropogation algorithm
###########################################
#without damping
inf_bp=opengm.inference.BeliefPropagation(gm)
#with damping
inf_bp=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(damping=0.05))

callback=PyCallback(shape,numLabels)
visitor=inf_bp.pythonVisitor(callback,visitNth=1)
import time
t0=time.time()
inf_bp.infer(visitor)
t1=time.time()
print t1-t0
# get the result states
argmin=inf_bp.arg()
#Evaluate the minimum energy
print "energy ",gm.evaluate(argmin)
#finally put the result in 2d grid in 480*640
result=argmin.reshape(dimx,dimy)
imgplot = matplotlib.pyplot.imshow(result)
matplotlib.pyplot.show()

###########################################
#Inference with TRWS algorithm
###########################################
inf_trws=opengm.inference.TrwsExternal(gm)
import time
t0=time.time()
inf_trws.infer()
t1=time.time()
print t1-t0
# get the result states
argmin=inf_trws.arg()
#Evaluate the minimum energy
print "energy ",gm.evaluate(argmin)
#finally put the result in 2d grid in 480*640
result=argmin.reshape(dimx,dimy)
imgplot = matplotlib.pyplot.imshow(result)
matplotlib.pyplot.show()


###########################################
#Inference with GraphCut algorithm
###########################################
inf_graphcut=opengm.inference.GraphCut(gm)
callback=PyCallback(shape,numLabels)
visitor=inf_trbp.pythonVisitor(callback,visitNth=1)
import time
t0=time.time()
inf_graphcut.infer(visitor)
t1=time.time()
print t1-t0
# get the result states
argmin=inf_graphcut.arg()
#Evaluate the minimum energy
print "energy ",gm.evaluate(argmin)
#finally put the result in 2d grid in 480*640
result=argmin.reshape(dimx,dimy)
imgplot = matplotlib.pyplot.imshow(result)
matplotlib.pyplot.show()


###########################################
#Inference with TreeReweightedBp algorithm
###########################################
#without damping
inf_trbp=opengm.inference.TreeReweightedBp(gm)
#with damping
inf_trbp=opengm.inference.TreeReweightedBp(gm,parameter=opengm.InfParam(damping=0.05))
callback=PyCallback(shape,numLabels)
visitor=inf_trbp.pythonVisitor(callback,visitNth=1)
import time
t0=time.time()
inf_trbp.infer(visitor)
t1=time.time()
print t1-t0
# get the result states
argmin=inf_trbp.arg()
#finally put the result in 2d grid in 480*640
result=argmin.reshape(dimx,dimy)
imgplot = matplotlib.pyplot.imshow(result)
matplotlib.pyplot.show()
