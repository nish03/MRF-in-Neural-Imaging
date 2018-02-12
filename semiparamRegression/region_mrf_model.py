import vigra
from vigra import graphs
import numpy
from sklearn import mixture
import opengm

def region_mrf_model(Z):
    Z = numpy.reshape(Z,[640, 480])
    Z = numpy.float32(Z)
    superpixelDiameter = 2                # super-pixel size
    slicWeight = 1                        # SLIC color - spatial weight
    labels, n_segments = vigra.analysis.slicSuperpixels(Z, slicWeight, superpixelDiameter) 
    labels = vigra.analysis.labelImage(labels) - 1
    gridGraph = graphs.gridGraph(Z.shape)
    rag = graphs.regionAdjacencyGraph(gridGraph, labels)
    nodeFeatures = rag.accumulateNodeFeatures(Z)
    nodeFeatures = nodeFeatures.reshape(-1,1)
    nCluster   = 2
    g = mixture.GaussianMixture(n_components=nCluster)
    g.fit(nodeFeatures)
    clusterProb = g.predict_proba(nodeFeatures)
    probs = numpy.clip(clusterProb, 0.00001, 0.99999)
    superpixel_unaries = -1.0*numpy.log(probs)   #define superpixel_unaries
    superpixel_regularizer = opengm.differenceFunction(shape=[nCluster,nCluster],norm=1,weight=1.0/nCluster,truncate=None) #define superpixel regularizer
    gm = opengm.graphicalModel([nCluster]*n_segments)
    fids = gm.addFunctions(superpixel_unaries)   #superpixel wise unary factors
    gm.addFactors(fids,numpy.arange(n_segments))
    rag_edges = rag.uvIds()
    rag_edges = numpy.sort(rag_edges,axis=1)
    fid = gm.addFunction(superpixel_regularizer)  #superpixel wise pairwise factors
    gm.addFactors(fid, numpy.sort(rag_edges, axis=1))
    inf_trws=opengm.inference.TrwsExternal(gm, parameter=opengm.InfParam(steps=50))  #Perform inference
    visitor=inf_trws.timingVisitor()
    inf_trws.infer(visitor)
    argmin=inf_trws.arg()
    argImg = rag.projectNodeFeaturesToGridGraph(argmin.astype(numpy.uint32))
    return argImg