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

def file2matrix(filename):
    with open(filename) as f:
        fileData=[data.strip().split('\t') for data in f.readlines()]
        returnMat=np.array(fileData)[:,:3].astype(float)
        classLabelVector=np.array(fileData)[:,3].astype(float).flatten()
        return returnMat,classLabelVector

def autoNorm(dataset):
    minVal=dataset.min(0)
    maxVal=dataset.max(0)
    rangeVal=maxVal-minVal
    normVal=(dataset-minVal)/rangeVal
    return minVal,maxVal,normVal


def datingClassTest():
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    hoRatio = 0.1
    dataNum = len(datingDataMat)
    trainingNum = dataNum - int(dataNum * 0.1)
    testingNum = int(dataNum * 0.1)

    indx = list(range(datingDataMat.shape[0]))
    np.random.shuffle(indx)
    datingDataMat = datingDataMat[indx]
    datingLabels = datingLabels[indx]
    minVal, maxVal, normVal = autoNorm(datingDataMat)

    X_train, y_train, X_test, y_test = normVal[:trainingNum], datingLabels[:trainingNum], + \
        normVal[trainingNum:], datingLabels[trainingNum:]
    correctNum = 0
    for indx, testData in enumerate(X_test):
        y_pred = classify0(testData, X_train, y_train, 3)
        y_true = y_test[indx]
        if y_pred == y_true:
            correctNum += 1

    acc = correctNum / testingNum
    return X_train, y_train, X_test, y_test, acc





if __name__=="__main__":

    group,labels=createDataSet()
    print(group)
    print(labels)

    inX=[0.2,0.3]

    pred_label=classify0(inX,group,labels,2)
    # print(pred_label)

    returnMat,classLabelVector=file2matrix('datingTestSet.txt')
    print(returnMat.shape)
    print(classLabelVector)

    minVal,maxVal,normVal=autoNorm(returnMat)
    print('minVal',minVal)
    print('maxVal',maxVal)
    print('normVal',normVal)