import numpy as np
import operator

def createDataSet():
    group=np.array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    datasetSize=dataSet.shape[0]    # get the number of samples
    X=np.tile(inX,(datasetSize,1))  # tile the sample and make the size equal to that of the  training dataset
    diffMat=X-dataSet
    diffMat=diffMat**2 #to square each element in the array
    sqDistance=diffMat.sum(axis=1)
    distances=np.sqrt(sqDistance)

    sortedDistanceIndices=distances.argsort() # sort the value of distances array in an ascending manner and return the corresponding index

    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistanceIndices[i]] #get the ith element in the sortedDistances which indicate the row number of original dataset
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1

    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True) # return a list of key-value tuple

    return sortedClassCount[0][0] #return the first item in the first tuple




if __name__=="__main__":

    group,labels=createDataSet()
    print(group)
    print(labels)

    inX=[0.2,0.3]

    pred_label=classify0(inX,group,labels,2)
    print(pred_label)