
import numpy
import opengm

#################################################################
######                    Synthetic model  
######Grid = 100*100, Number of Labels= 20, Coupling strength = 0.5
######Truncation threshold = 10
######Unary potential = random 
######Spatial regularizers/functions used: 1) Potts
######                           2) Squared Difference
######                           3) Truncated Absolute Difference
######                           4) Truncated Squared Difference
#################################################################
numLabels=20
shape=(100,100)
unaries=numpy.random.rand(shape[0],shape[1],numLabels)

###Potts function as spatial regularizer
potts=opengm.PottsFunction([numLabels,numLabels],0.0,0.5)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)

###Squared difference as spatial regularizer.Use TruncatedSquaredDifference
### but trick is to keep threshold high like 400 so that it becomes squared distance
tsd=opengm.TruncatedSquaredDifferenceFunction([numLabels,numLabels],400,0.5)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=tsd)

###Truncated Absolute Difference as spatial regularizer
tad=opengm.TruncatedAbsoluteDifferenceFunction([numLabels,numLabels],10,0.5)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=tad)

###Truncated Squared Difference as spatial regularizer
tsd=opengm.TruncatedSquaredDifferenceFunction([numLabels,numLabels],10,0.5)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=tsd)



##############################################################
######Synthetic model  Grid = 40*40 #####
##############################################################
numLabels=2
shape=(40,40)
unaries=numpy.random.rand(shape[0],shape[1],numLabels)
potts=opengm.PottsFunction([numLabels,numLabels],0.0,0.5)
gm=opengm.grid2d2Order(unaries=unaries,regularizer=potts)




#################################################################
###               Intraoperative data model
#################################################################
img = numpy.reshape(discretized_T_cropped, (-1, 640))
#convert the values in the range of 0 to 1  0=black 1=white
img = numpy.asarray(img).astype(float)/255

#plot img to see the original starting point of the image
imgplot = matplotlib.pyplot.imshow(img)
matplotlib.pyplot.show()

##Now start coding for graphical model
shape = img.shape
dimx=img.shape[0]
dimy=img.shape[1]

numVar=dimx*dimy
numLabels=2
beta=0.3

numberOfStates=numpy.ones(numVar,dtype=opengm.index_type)*numLabels
gm=opengm.graphicalModel(numberOfStates,operator='adder')

#Adding unary function and factors
for y in range(dimy):
   for x in range(dimx):
      f=numpy.ones(2,dtype=numpy.float32)
      f[0]=img[x,y]
      f[1]=1.0-img[x,y]
      fid=gm.addFunction(f)
      gm.addFactor(fid,(x*dimy+y,))

#Adding binary function and factors
vis=numpy.ones(5,dtype=opengm.index_type)

#Potts as binary function
f=numpy.ones(pow(numLabels,2),dtype=numpy.float32).reshape(numLabels,numLabels)*beta
for l in range(numLabels):
   f[l,l]=0  
fid=gm.addFunction(f)

#add binary factors
for y in range(dimy):   
   for x in range(dimx):
      if(x+1<dimx):
         #vi as tuple (list and numpy array can also be used as vi's)
         gm.addFactor(fid,numpy.array([x*dimy+y,(x+1)*dimy+y],dtype=opengm.index_type))
      if(y+1<dimy):
         #vi as list (tuple and numpy array can also be used as vi's)
         gm.addFactor(fid,[x*dimy+y,x*dimy+(y+1)])


opengm.hdf5.saveGraphicalModel(gm,'model.h5','gm')
